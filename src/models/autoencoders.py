from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np
import torch as T
from pytorch_lightning import LightningModule

import wandb
from mattstools.mattstools.loss import kld_to_norm, masked_dist_loss
from mattstools.mattstools.modules import IterativeNormLayer
from mattstools.mattstools.numpy_utils import undo_log_squash
from mattstools.mattstools.plotting import plot_latent_space, plot_multi_hists_2
from mattstools.mattstools.torch_utils import (
    get_loss_fn,
    get_sched,
    reparam_trick,
    to_np,
)
from mattstools.mattstools.transformers import (
    FullTransformerVectorDecoder,
    FullTransformerVectorEncoder,
)
from src.jet_utils import locals_to_jet_pt_mass
from src.torch_jets import torch_locals_to_jet_pt_mass


class TransformerVAE(LightningModule):
    """An transformer based autoencoder for point cloud data using
    transformers."""

    def __init__(
        self,
        *,
        inpt_dim: list,
        n_nodes: int,
        n_classes: int,
        lat_dim: int,
        loss_name: str,
        kld_weight: float,
        kld_warmup_steps: int,
        reg_loss_name: str,
        reg_loss_weight: float,
        reg_loss_warmup_steps: int,
        normaliser_config: Mapping,
        encoder_config: Mapping,
        decoder_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
    ) -> None:
        """
        Args:
            inpt_dim: Number of edge, node and high level features
            n_nodes: Number of nodes used as inputs
            n_classes: Number of classes
            lat_dim: Dimension size of the latent space
            loss_name: Loss function to use in reconstuction
            kld_weight: Loss weight used for the KLD term in the autoencoder
            kld_warmup_steps: Number of steps for the KLD term to reach full strength
            reg_loss_name: Name of the loss function for mass and pt regression
            reg_loss_weight: Loss weight used for the regression term
            reg_loss_warmup_steps: Number of steps for the reg term to reach full strength
            normaliser_config: Config for the IterativeNormLayer
            encoder_config: Config for the TransformerVectorEncoder
            decoder_config: Config for the TransformerVectorDecoder
            optimizer: Partially initialised optimiser
            scheduler: How the sceduler should be used
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.edge_dim = inpt_dim[0]
        self.node_dim = inpt_dim[1]
        self.high_dim = inpt_dim[2]
        self.n_classes = n_classes
        self.loss_fn = get_loss_fn(loss_name)
        self.lat_dim = lat_dim
        self.kld_weight = kld_weight
        self.kld_warmup_steps = kld_warmup_steps
        self.reg_loss_fn = get_loss_fn(reg_loss_name)
        self.reg_loss_weight = reg_loss_weight
        self.reg_loss_warmup_steps = reg_loss_warmup_steps
        self.val_step_outs = []

        # The layers which normalise the input data
        self.node_norm = IterativeNormLayer(self.node_dim, **normaliser_config)
        if self.edge_dim:
            self.edge_norm = IterativeNormLayer(self.edge_dim, **normaliser_config)
        if self.high_dim:
            self.high_norm = IterativeNormLayer(self.high_dim, **normaliser_config)

        # The transformer encoding model
        self.encoder = FullTransformerVectorEncoder(
            inpt_dim=self.node_dim,
            outp_dim=self.lat_dim * 2,
            edge_dim=self.edge_dim,
            ctxt_dim=self.high_dim,
            **encoder_config,
        )

        # Initialise the transformer generator model
        self.generator = FullTransformerVectorDecoder(
            inpt_dim=self.lat_dim,
            outp_dim=self.node_dim,
            ctxt_dim=self.high_dim,
            **decoder_config,
        )

    def forward(
        self,
        edges: T.Tensor,
        nodes: T.Tensor,
        high: T.Tensor,
        adjmat: T.BoolTensor,
        mask: T.BoolTensor,
        do_reverse: bool = True,
    ) -> tuple:
        """Pass inputs through the full autoencoder.

        Returns:
            reconstructions, latent samples, latent means, latent_stds
        """

        # Transformers create their attention matrices as: recv x send
        # This opposite to the GNNs adjmat which is: send x recv, rectify
        # Transformers also need the edges to be filled or None
        adjmat = adjmat.transpose(-1, -2)
        edges = edges.transpose(-2, -3) if edges.nelement() > 0 else None

        # Transformers also need the attention mask to not fully block all queries
        # Meaning that padded nodes need to be able to receive information
        # Even if the padded node will be ignored
        adjmat = adjmat | ~mask.unsqueeze(-1)

        # Pass the inputs through the normalisation layers
        nodes = self.node_norm(nodes, mask)
        if self.edge_dim:
            edges = self.edge_norm(edges, adjmat)
        if self.high_dim:
            high = self.high_norm(high)

        # Pass through the encoder
        latents = self.encoder(
            nodes, mask, ctxt=high, attn_bias=edges, attn_mask=adjmat
        )

        # Apply the reparameterisation trick
        latents, means, lstds = reparam_trick(latents)

        # Pass through the generator
        rec_nodes = self.generator(latents, mask, ctxt=high)

        # Undo the scaling (HERE BECAUSE LOSS TERMS DEPEND ON RAW DATA LIKE PT WEIGHT)
        rec_nodes = self.node_norm.reverse(rec_nodes, mask)

        return rec_nodes, latents, means, lstds

    def generate_from_noise(
        self, mask: T.BoolTensor, high: T.Tensor | None = None
    ) -> T.Tensor:
        """Generate random graphs using noise as the latent vector.

        Still requires the conditional labels and the requested mask!
        """
        latents = T.randn((len(mask), self.lat_dim), device=self.device)
        if self.high_dim:
            high = self.high_norm(high)
        nodes = self.generator(latents, mask, ctxt=high)
        nodes = self.node_norm.reverse(nodes, mask)
        return nodes

    def _shared_step(self, sample: tuple, loss_reduction: str = "mean") -> T.Tensor:
        # Unpack the tuple
        edges, nodes, high, adjmat, mask, label = sample

        # Get the reconstructions from the vae
        rec_nodes, _latents, means, lstds = self.forward(
            edges, nodes, high, adjmat, mask, do_reverse=False
        )

        # Distribution matching loss for reconstructed nodes
        rec_loss = masked_dist_loss(
            self.loss_fn, nodes, mask, rec_nodes, mask, reduce=loss_reduction
        )

        # Latent space loss for encodings
        lat_loss = kld_to_norm(means, lstds, reduce=loss_reduction)

        # Reconstructed mass and pt loss
        inp_pt_mass = torch_locals_to_jet_pt_mass(nodes, mask, clamp_etaphi=1)
        rec_pt_mass = torch_locals_to_jet_pt_mass(rec_nodes, mask, clamp_etaphi=1)
        reg_loss = self.reg_loss_fn(inp_pt_mass, rec_pt_mass)
        if loss_reduction == "mean":
            reg_loss = reg_loss.mean()

        # Combine the losses with their respective terms
        total_loss = (
            rec_loss
            + lat_loss * self._get_kld_weight()
            + reg_loss * self._get_reg_loss_weight()
        )
        return rec_nodes, means, rec_loss, lat_loss, reg_loss, total_loss

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        rec_nodes, means, rec_loss, lat_loss, reg_loss, total_loss = self._shared_step(
            sample
        )
        self.log("train/rec_loss", rec_loss)
        self.log("train/lat_loss", lat_loss)
        self.log("train/reg_loss", reg_loss)
        self.log("train/total_loss", total_loss)
        self.log("kld_weight", self._get_kld_weight())
        self.log("reg_loss_weight", self._get_reg_loss_weight())

        return total_loss

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        rec_nodes, means, rec_loss, lat_loss, reg_loss, total_loss = self._shared_step(
            sample
        )
        self.log("valid/rec_loss", rec_loss)
        self.log("valid/lat_loss", lat_loss)
        self.log("valid/reg_loss", reg_loss)
        self.log("valid/total_loss", total_loss)

        # Also generate samples from noise to get the generation loss
        # For memory reasons only do this for the first 100 batches
        if batch_idx < 100:
            _, nodes, high, _, mask, _ = sample
            gen_nodes = self.generate_from_noise(mask, high)

            # Add the variables required for plotting
            self.val_step_outs.append(to_np((nodes, rec_nodes, gen_nodes, mask, means)))

        return total_loss

    def predict_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        _, _, rec_loss, lat_loss, _, total_loss = self._shared_step(
            sample,
            loss_reduction="none",
        )
        return {
            "recon_loss": rec_loss,
            "latent_loss": lat_loss,
            "total_loss": total_loss,
        }

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("train/rec_loss", summary="min")
            wandb.define_metric("train/lat_loss", summary="min")
            wandb.define_metric("train/reg_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")
            wandb.define_metric("valid/rec_loss", summary="min")
            wandb.define_metric("valid/lat_loss", summary="min")
            wandb.define_metric("valid/reg_loss", summary="min")

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        # Use mattstools to initialise the scheduler (cyclic-epoch sync)
        sched = get_sched(
            self.hparams.scheduler.mattstools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.scheduler.lightning},
        }

    def _get_kld_weight(self) -> float:
        """Returns the KLD weight which is always a linear warmup to full
        strength."""
        return min(self.global_step / self.kld_warmup_steps, 1) * self.kld_weight

    def _get_reg_loss_weight(self) -> float:
        """Returns the KLD weight which is always a linear warmup to full
        strength."""
        return (
            min(self.global_step / self.reg_loss_warmup_steps, 1) * self.reg_loss_weight
        )

    def on_validation_epoch_end(self) -> None:
        """Makes several plots of the jets and how they are reconstructed.

        Assumes that the nodes are of the format: del_eta, del_phi,
        log_pt
        """

        # Unpack the list
        nodes = np.vstack([v[0] for v in self.val_step_outs]).astype("float")
        rec_nodes = np.vstack([v[1] for v in self.val_step_outs]).astype("float")
        gen_nodes = np.vstack([v[2] for v in self.val_step_outs]).astype("float")
        mask = np.vstack([v[3] for v in self.val_step_outs])
        means = np.vstack([v[4] for v in self.val_step_outs]).astype("float")

        # Clear the outputs
        self.val_step_outs.clear()

        # Create the plotting dir
        plot_dir = Path("./plots/")
        plot_dir.mkdir(parents=False, exist_ok=True)

        # Convert to total jet mass and pt
        jets = locals_to_jet_pt_mass(nodes, mask)
        rec_jets = locals_to_jet_pt_mass(rec_nodes, mask)
        gen_jets = locals_to_jet_pt_mass(gen_nodes, mask)

        # Image for the total jet variables
        jet_img = plot_multi_hists_2(
            data_list=[jets, rec_jets, gen_jets],
            data_labels=["Original", "Reconstructed", "Generated"],
            col_labels=["pt", "mass"],
            bins="quant",
            do_ratio_to_first=True,
            return_img=True,
            path=f"./plots/jets_{self.trainer.current_epoch}",
        )

        # Latent space plots, showing the spreads of the means
        lat_img = plot_latent_space(
            path=f"./plots/latents_{self.trainer.current_epoch}",
            latents=means,
            return_img=True,
        )

        # Undo the log squash for the plots
        nodes[..., -1] = undo_log_squash(nodes[..., -1])
        rec_nodes[..., -1] = undo_log_squash(rec_nodes[..., -1])
        gen_nodes[..., -1] = undo_log_squash(gen_nodes[..., -1])

        # Plot histograms for the constituent marginals
        cst_img = plot_multi_hists_2(
            data_list=[nodes[mask], rec_nodes[mask], gen_nodes[mask]],
            data_labels=["Original", "Reconstructed", "Generated"],
            col_labels=["del_eta", "del_phi", "pt"],
            return_img=True,
            do_ratio_to_first=True,
            do_err=True,
            path=f"./plots/csts_{self.trainer.current_epoch}",
            bins=[
                np.linspace(-1, 1, 50),
                np.linspace(-1, 1, 50),
                np.linspace(0, 500, 50),
            ],
            logy=True,
        )

        # Create the wandb table and add the data
        if wandb.run is not None:
            table = wandb.Table(columns=["csts", "jets", "latents"])
            table.add_data(
                wandb.Image(cst_img), wandb.Image(jet_img), wandb.Image(lat_img)
            )
            wandb.run.log({"table": table}, commit=False)
