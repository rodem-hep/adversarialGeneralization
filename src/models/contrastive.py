from functools import partial
from typing import Mapping

import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule

from mattstools.mattstools.cnns import DoublingConvNet
from mattstools.mattstools.torch_utils import get_sched
from mattstools.mattstools.transformers import FullTransformerVectorEncoder
from src.models.image_classifiers import ConvNet


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=1.0):
        super().__init__()
        self.logit_scale = logit_scale

    def forward(self, embedding_1, embedding_2, valid=False):
        device = embedding_1.device
        logits_1 = self.logit_scale * embedding_1 @ embedding_2.T
        logits_2 = self.logit_scale * embedding_2 @ embedding_1.T
        num_logits = logits_1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        loss = 0.5 * (
            F.cross_entropy(logits_1, labels) + F.cross_entropy(logits_2, labels)
        )
        return loss


class CLIPLossNorm(nn.Module):
    def __init__(
        self,
        logit_scale_init=np.log(1 / 0.07),
        logit_scale_max=np.log(100),
        logit_scale_min=np.log(0.01),
        logit_scale_learnable=True,
    ):
        super().__init__()
        if logit_scale_learnable:
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        else:
            self.logit_scale = torch.ones([], requires_grad=False) * logit_scale_init
        self.logit_scale_max = logit_scale_max
        self.logit_scale_min = logit_scale_min
        self.logit_scale_learnable = logit_scale_learnable

    def forward(self, embedding_1, embedding_2, valid=False):
        logit_scale = torch.clamp(
            self.logit_scale, max=self.logit_scale_max, min=self.logit_scale_min
        ).exp()
        if self.logit_scale_learnable:
            wandb.log({"logit_scale": logit_scale})
        device = embedding_1.device
        norm = (
            embedding_1.norm(dim=1, keepdim=True)
            @ embedding_2.norm(dim=1, keepdim=True).T
        )
        logits_1 = (logit_scale * embedding_1 @ embedding_2.T) / norm
        logits_2 = (logit_scale * embedding_2 @ embedding_1.T) / norm.T
        num_logits = logits_1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        loss = 0.5 * (
            F.cross_entropy(logits_1, labels) + F.cross_entropy(logits_2, labels)
        )
        return loss


class MSELoss(nn.Module):
    def __init__(self, logit_scale=1.0):
        super().__init__()

    def forward(self, embedding_1, embedding_2, valid=False):
        loss = ((embedding_1 - embedding_2) ** 2).mean()
        return loss


class CosineSim(nn.Module):
    def __init__(self, logit_scale=1.0):
        super().__init__()

    def forward(self, embedding_1, embedding_2, valid=False):
        similarities = 1 - (embedding_1 * embedding_2).sum(dim=1)
        return similarities


class CosineSimNorm(nn.Module):
    def __init__(self, logit_scale=1.0):
        super().__init__()

    def forward(self, embedding_1, embedding_2, valid=False):
        norm = embedding_1.norm(dim=1, keepdim=True) * embedding_2.norm(
            dim=1, keepdim=True
        )
        similarities = 1 - ((embedding_1 * embedding_2).sum(dim=1, keepdim=True) / norm)
        return similarities


def get_2embedding_loss(loss_name, loss_parameters={}):
    if loss_name == "CLIP":
        return CLIPLoss(**loss_parameters)
    elif loss_name == "CLIP_norm" or "contrastive_norm":
        return CLIPLossNorm(**loss_parameters)
    elif loss_name == "mse":
        return MSELoss(**loss_parameters)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


class CLRBase(LightningModule):
    def forward(self, x):
        return self.net1(x[0]), self.net2(x[1])

    def _shared_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        emb1, emb2 = self.forward(sample)
        return self.loss_fn(emb1, emb2).mean()

    def training_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        loss = self._shared_step(sample, batch_idx)
        self.log("train/total_loss", loss)
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> T.Tensor:
        loss = self._shared_step(sample, batch_idx)
        self.log("valid/total_loss", loss)
        return loss

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss")
            wandb.define_metric("valid/total_loss")

    def predict_step(self, sample: tuple, _batch_idx: int) -> None:
        """Single step which produces the tagger outputs for a single test
        batch Must be as a dictionary to generalise to models with multiple
        tagging methods."""
        emb1, emb2 = self.forward(sample)
        anomaly_score_batch = self.anomaly_score(emb1, emb2)
        return {"CosineSimNorm": anomaly_score_batch}

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


class ImagexImageCLR(CLRBase):
    def __init__(
        self,
        *,
        image_dim: int,
        emb_size: int,
        loss_name: str,
        convnet_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        loss_parameters: dict = {},
    ) -> None:
        super().__init__()
        self.net1 = ConvNet(
            image_dim=image_dim, n_outputs=emb_size, convnet_config=convnet_config
        )
        self.net2 = ConvNet(
            image_dim=image_dim, n_outputs=emb_size, convnet_config=convnet_config
        )
        self.save_hyperparameters(logger=False)

        self.loss_fn = get_2embedding_loss(loss_name, loss_parameters)
        self.anomaly_score = CosineSimNorm()


class ImagexImageResnetCLR(CLRBase):
    def __init__(
        self,
        *,
        loss_name: str,
        doublinconvnet_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        loss_parameters: dict = {},
    ) -> None:
        super().__init__()
        self.net1 = DoublingConvNet(**doublinconvnet_config)
        self.net2 = DoublingConvNet(**doublinconvnet_config)
        self.save_hyperparameters(logger=False)

        self.loss_fn = get_2embedding_loss(loss_name, loss_parameters)
        self.anomaly_score = CosineSimNorm()


class JIxPCResnetCLR(CLRBase):
    def __init__(
        self,
        *,
        loss_name: str,
        inpt_dim: tuple,
        emb_dim: int,
        jinet_config: Mapping,
        pcnet_config: Mapping,
        optimizer: partial,
        scheduler: Mapping,
        n_nodes: None = None,
        n_classes: None = None,
        loss_parameters: dict = {},
    ) -> None:
        ji_dim = inpt_dim[0]
        edge_dim = inpt_dim[1]
        node_dim = inpt_dim[2]
        high_dim = inpt_dim[3]

        super().__init__()
        # input size should be given without channels to doubling net
        self.jinet = DoublingConvNet(
            inpt_size=ji_dim[1:], outp_dim=emb_dim, **jinet_config
        )
        self.pcnet = FullTransformerVectorEncoder(
            inpt_dim=node_dim,
            outp_dim=emb_dim,
            ctxt_dim=high_dim,
            edge_dim=edge_dim,
            **pcnet_config,
        )
        self.save_hyperparameters(logger=False)

        self.loss_fn = get_2embedding_loss(loss_name, loss_parameters)
        self.anomaly_score = CosineSimNorm()

    def forward(self, x):
        (edges, nodes, high, adjmat, mask) = x[1:]

        # Transformers create their attention matrices as: recv x send
        # This opposite to the GNNs adjmat which is: send x recv, rectify
        # See mathews transformer classifier for same fix
        adjmat = adjmat.transpose(-1, -2)
        edges = edges.transpose(-2, -3)

        # Pass through the transformer and return

        return self.jinet(x[0]), self.pcnet(
            nodes, mask, ctxt=high, attn_mask=adjmat, attn_bias=edges
        )
