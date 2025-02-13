import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import h5py
from pathlib import Path

import logging
import hydra
import torch
from omegaconf import DictConfig
import numpy as np

from mattstools.mattstools.hydra_utils import reload_original_config

from franckstools.franckstools.swag import SWAGWrapper
from franckstools.franckstools.sam import SAMWrapper
from franckstools.franckstools.adversarial_attack import AdversarialWrapper

# from franckstools.franckstools.loss_landscapes import compute_gauss_newton_approximation
from torch.autograd.functional import hessian
from franckstools.franckstools.utils import power_iteration

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="hessianAnalysis.yaml",
)
def main(cfg: DictConfig) -> None:
    model = load_model(cfg)
    path = f"{cfg.tagger_path}/{cfg.tagger_name}"
    orig_cfg = reload_original_config(cfg, get_best=cfg.get_best, path=path)

    # Cycle through the datasets and create the dataloader
    for key, dataset in cfg.datamodule.data_conf.datasets.items():
        log.info(f"Instantiating the data module for {dataset}")
        orig_cfg.datamodule.data_conf.datasets = {"c0": dataset}

        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()

        computeLargestEigenvalueHessian(
            cfg, model, dataloader, dataset=dataset, isInputSpace=True
        )
        computeLargestEigenvalueHessian(
            cfg, model, dataloader, dataset=dataset, isInputSpace=False
        )

    return


def load_model(cfg):
    log.info("Loading run information")
    path = f"{cfg.tagger_path}/{cfg.tagger_name}"
    orig_cfg = reload_original_config(cfg, get_best=cfg.get_best, path=path)

    log.info("Loading best checkpoint")
    wrapperList = []
    if (
        hasattr(orig_cfg, "use_sharpness_aware_minimization")
        and orig_cfg.use_sharpness_aware_minimization
    ):
        wrapperList.append("SAM")
    if (
        hasattr(orig_cfg, "use_adversarial_samples")
        and orig_cfg.use_adversarial_samples
    ):
        wrapperList.append("Adversarial")
    if hasattr(orig_cfg, "use_SWAG") and orig_cfg.use_SWAG:
        wrapperList.append("SWAG")

    if len(wrapperList) > 0:
        log.info("Instantiating the underlying base model")
        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

        if hasattr(datamodule, "n_nodes"):
            model = hydra.utils.instantiate(
                orig_cfg.model,
                inpt_dim=datamodule.get_dims(),
                n_nodes=datamodule.n_nodes,
                n_classes=datamodule.n_classes,
            )
        else:
            model = hydra.utils.instantiate(orig_cfg.model)

        for wrapper in wrapperList:
            log.info(f"adding {wrapper} wrapper")
            if wrapper == "SWAG":
                model = SWAGWrapper(model)
            elif wrapper == "SAM":
                model = SAMWrapper(
                    model,
                    rho=orig_cfg.get("rho"),
                    method_type=orig_cfg.get("method_type"),
                    sparsity=orig_cfg.get("sparsity"),
                    num_samples=orig_cfg.get("num_samples"),
                    update_freq=orig_cfg.get("update_freq"),
                    drop_rate=orig_cfg.get("drop_rate"),
                    drop_strategy=orig_cfg.get("drop_strategy"),
                    growth_strategy=orig_cfg.get("growth_strategy"),
                    T_start=orig_cfg.get("T_start"),
                    T_end=orig_cfg.get("T_end"),
                )
            elif wrapper == "Adversarial":
                model = AdversarialWrapper(
                    model,
                    adversarial_scheduler_config=orig_cfg.adversarial_scheduler_config,
                    attack_type=orig_cfg.attack_type,
                    epsilon=orig_cfg.epsilon,
                    PGD_num_steps=orig_cfg.PGD_num_steps,
                    PGD_step_size=orig_cfg.PGD_step_size,
                )
            else:
                raise ValueError(f"Unknown wrapper {wrapper}")

        checkpoint = torch.load(orig_cfg.ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])

    else:
        log.info("Instantiating the model")
        model_class = hydra.utils.get_class(orig_cfg.model._target_)
        model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)

    orig_cfg.datamodule.predict_n_test = cfg.datamodule.predict_n_test
    orig_cfg.datamodule.data_conf.dataset_type = cfg.datamodule.data_conf.dataset_type

    if cfg.SWA_export:
        log.info("Initializing SWA export")
        model.init_SWA_export()
    elif cfg.SWAG_export:
        log.info("Initializing SWAG export")
        model.init_SWAG_export()

    model.eval()

    return model


def computeLargestEigenvalueHessian(cfg, model, dataloader, dataset, isInputSpace=True):
    def computeLoss(model, edges, nodes, high, adjmat, mask, label):
        outputs = model(edges, nodes, high, adjmat, mask)
        if hasattr(model, "parent"):
            loss = model.parent.loss_fn(outputs, label)
        else:
            loss = model.loss_fn(outputs, label)

        return loss

    def getHessian(
        model,
        dataloader,
        isInputSpace=True,
        isTransformerNetwork=False,
        doMean=True,
    ):
        hessians = []
        max_iter = 50 if isInputSpace else cfg.max_iter

        for edges, nodes, high, adjmat, mask, label in dataloader:
            if max_iter == 0:
                break

            def inputHessian(inputs, rest_inputs):
                inputs = inputs.requires_grad_(True)
                nodes = torch.cat((inputs, rest_inputs), dim=1)
                loss = computeLoss(model, edges, nodes, high, adjmat, mask, label)
                loss = loss.mean()

                grads_batch = torch.autograd.grad(
                    loss, inputs, retain_graph=True, create_graph=True
                )[0].squeeze()

                flattened_grads = torch.cat(
                    ([grad.flatten() for grad in grads_batch[0]])
                )

                hessianMatrix = torch.zeros(
                    flattened_grads.shape[0], flattened_grads.shape[0]
                )

                for batch_idx, grads in enumerate(grads_batch):
                    flattened_grads = torch.cat(([grad.flatten() for grad in grads]))

                    for idx, grad in enumerate(flattened_grads):
                        second_der = torch.autograd.grad(
                            grad,
                            inputs,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        second_der = second_der[0][batch_idx]
                        second_der = torch.cat(
                            ([grad.flatten() for grad in second_der])
                        )
                        hessianMatrix[idx, :] += second_der

                hessianMatrix = hessianMatrix / len(grads_batch)
                return hessianMatrix

            def weightHessian(weight_layer):
                loss = computeLoss(model, edges, nodes, high, adjmat, mask, label)
                loss = loss.mean()
                grads = torch.autograd.grad(
                    loss,
                    weight_layer.weight,
                    retain_graph=True,
                    create_graph=True,
                )[0].squeeze()
                flattened_grads = torch.cat(([grad.flatten() for grad in grads]))

                hessianMatrix = torch.zeros(
                    flattened_grads.shape[0], flattened_grads.shape[0]
                )

                for idx, grad in enumerate(flattened_grads):
                    second_der = torch.autograd.grad(
                        grad,
                        weight_layer.weight,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    second_der = torch.cat(([grad.flatten() for grad in second_der]))
                    hessianMatrix[idx, :] = second_der

                return hessianMatrix

            if isInputSpace:
                inputs = nodes[:, : cfg.n_const, :]
                rest_inputs = nodes[:, cfg.n_const :, :]
                hessianMatrix = inputHessian(inputs, rest_inputs)
            else:
                # Get the weights of the last layer
                if hasattr(model, "parent"):
                    if isTransformerNetwork:
                        weight_layer = model.parent.transformer.transformer.layers[
                            -1
                        ].output_proj
                    else:
                        weight_layer = model.parent.dense.output_block.block[-1]
                else:
                    if isTransformerNetwork:
                        weight_layer = model.transformer.transformer.layers[
                            -1
                        ].output_proj
                    else:
                        weight_layer = model.dense.output_block.block[-1]

                hessianMatrix = weightHessian(weight_layer)

            hessianMatrix = hessianMatrix.squeeze()
            hessians.append(hessianMatrix)

            max_iter -= 1

        if doMean:
            avg_hessian = torch.mean(torch.stack(hessians), dim=0)
            return avg_hessian
        else:
            return hessians

    log.info("Computing Hessian")
    isTransformer = cfg.tagger_name.split("_")[0] == "simpleTransformer"

    hessianMatrices = getHessian(
        model,
        dataloader,
        isInputSpace,
        isTransformerNetwork=isTransformer,
        doMean=False,
    )

    log.info("Computing largest eigenvalue")
    largest_eigenvalues = []
    for hessianMatrix in hessianMatrices:
        largest_eigenvalue, _ = power_iteration(hessianMatrix)
        largest_eigenvalue = np.abs(largest_eigenvalue)
        largest_eigenvalues.append(largest_eigenvalue)

    largest_eigenvalue = np.mean(largest_eigenvalues)
    largest_eigenvalue_std = np.std(largest_eigenvalues)

    log.info(f"Largest eigenvalue: {largest_eigenvalue} +- {largest_eigenvalue_std}")

    log.info("Saving results")
    output_dir = cfg.output_dir
    # create output directory if not exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    spaceSuffix = "input" if isInputSpace else "weight"
    with open(
        f"{output_dir}/{dataset}_hessian_largest_eigenvalue_{spaceSuffix}.txt", "w"
    ) as f:
        f.write(f"Largest eigenvalue: {largest_eigenvalue} +- {largest_eigenvalue_std}")

    return


if __name__ == "__main__":
    main()
