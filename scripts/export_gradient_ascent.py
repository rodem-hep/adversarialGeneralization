import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import h5py
from pathlib import Path

import logging
import hydra
import torch
from omegaconf import DictConfig

from mattstools.mattstools.hydra_utils import reload_original_config

from franckstools.franckstools.swag import SWAGWrapper
from franckstools.franckstools.sam import SAMWrapper
from franckstools.franckstools.adversarial_attack import AdversarialWrapper
from franckstools.franckstools.invariantRiskMinimization import IRMWrapper
from franckstools.franckstools.loss_landscapes import gradient_ascent_weight_tracing
from franckstools.franckstools.loss_landscapes import gradient_ascent_input_tracing


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="gradientAscent.yaml",
)
def main(cfg: DictConfig) -> None:
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
    if hasattr(orig_cfg, "use_IRM_mass") and orig_cfg.use_IRM_mass:
        wrapperList.append("IRM_mass")

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
            elif wrapper == "IRM_mass":
                model = IRMWrapper(
                    model,
                    n_environments=orig_cfg.n_environments,
                    #    weight_decay = orig_cfg.weight_decay,
                    #    penalty_weight = orig_cfg.penalty_weight,
                    #    penalty_anneal_iters = orig_cfg.penalty_anneal_iters,
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

    # WEIGHT tracing
    # Initialize a list to store all tensors
    loss_tensors = []
    std_tensors = []

    for key, dataset in cfg.datamodule.data_conf.datasets.items():
        log.info(f"Instantiating the data module for {dataset}")
        orig_cfg.datamodule.data_conf.datasets = {"c0": dataset}
        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()

        log.info("Instantiating the trainer")
        trainer = hydra.utils.instantiate(orig_cfg.trainer)

        log.info("Calculating losses")
        loss_tensor, std_tensor = gradient_ascent_weight_tracing(
            model, dataloader, cfg.num_steps, cfg.step_size
        )
        loss_tensor = torch.tensor(loss_tensor)
        std_tensor = torch.tensor(std_tensor)

        loss_tensors.append(loss_tensor)
        std_tensors.append(std_tensor)

    avg_loss_tensor = torch.stack(loss_tensors).mean(dim=0)
    avg_std_tensor = torch.stack(std_tensors).mean(dim=0)

    log.info("Saving results")
    torch.save(
        avg_loss_tensor,
        f"{cfg.output_dir}/{cfg.evaluated_on_dataset_type}_gradient_ascent_weight_loss_tensor.pt",
    )
    torch.save(
        avg_std_tensor,
        f"{cfg.output_dir}/{cfg.evaluated_on_dataset_type}_gradient_ascent_weight_std_tensor.pt",
    )

    # INPUT tracing
    # Initialize a list to store all tensors
    loss_tensors = []
    std_tensors = []

    for key, dataset in cfg.datamodule.data_conf.datasets.items():
        log.info(f"Instantiating the data module for {dataset}")
        orig_cfg.datamodule.data_conf.datasets = {"c0": dataset}
        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()

        log.info("Instantiating the trainer")
        trainer = hydra.utils.instantiate(orig_cfg.trainer)

        log.info("Calculating losses")
        loss_tensor, std_tensor = gradient_ascent_input_tracing(
            model, dataloader, cfg.num_steps, cfg.step_size
        )
        loss_tensor = torch.tensor(loss_tensor)
        std_tensor = torch.tensor(std_tensor)

        loss_tensors.append(loss_tensor)
        std_tensors.append(std_tensor)

    avg_loss_tensor = torch.stack(loss_tensors).mean(dim=0)
    avg_std_tensor = torch.stack(std_tensors).mean(dim=0)

    log.info("Saving results")
    torch.save(
        avg_loss_tensor,
        f"{cfg.output_dir}/{cfg.evaluated_on_dataset_type}_gradient_ascent_input_loss_tensor.pt",
    )
    torch.save(
        avg_std_tensor,
        f"{cfg.output_dir}/{cfg.evaluated_on_dataset_type}_gradient_ascent_input_std_tensor.pt",
    )


if __name__ == "__main__":
    main()
