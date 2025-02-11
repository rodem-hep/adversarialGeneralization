import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from pathlib import Path

import h5py
import hydra
import torch as T
from omegaconf import DictConfig

from mattstools.mattstools.hydra_utils import reload_original_config
from mattstools.mattstools.torch_utils import to_np

from franckstools.franckstools.swag import SWAGWrapper
from franckstools.franckstools.sam import SAMWrapper
from franckstools.franckstools.adversarial_attack import AdversarialWrapper
from franckstools.franckstools.invariantRiskMinimization import IRMWrapper

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="hessianAnalyis.yaml",
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(cfg, get_best=cfg.get_best)
    if cfg.adversarial_export:
        orig_cfg.model.use_adversarial_samples = True

    orig_cfg.datamodule.data_conf.dataset_type = cfg.datamodule.data_conf.dataset_type
    orig_cfg.datamodule.data_conf.datasets = cfg.datamodule.data_conf.datasets
    orig_cfg.datamodule.data_conf.rodem_predictions_path = None
    orig_cfg.datamodule.export_train = cfg.datamodule.export_train

    ##########
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

        checkpoint = T.load(orig_cfg.ckpt_path, map_location=T.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])

    else:
        log.info("Instantiating the model")
        model_class = hydra.utils.get_class(orig_cfg.model._target_)
        model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)
    ##########

    # log.info("Loading best checkpoint")
    # if (hasattr(orig_cfg, "use_SWAG") and orig_cfg.use_SWAG):
    #     log.info("Instantiating the underlying model, datamodule and SWAGWrapper")
    #     datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

    #     if hasattr(datamodule, "n_nodes"):
    #         model = hydra.utils.instantiate(
    #             orig_cfg.model,
    #             inpt_dim=datamodule.get_dims(),
    #             n_nodes=datamodule.n_nodes,
    #             n_classes=datamodule.n_classes,
    #         )
    #     else:
    #         model = hydra.utils.instantiate(orig_cfg.model)

    #     model = SWAGWrapper(model)
    #     checkpoint = T.load(orig_cfg.ckpt_path)
    #     model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     model_class = hydra.utils.get_class(orig_cfg.model._target_)
    #     model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)

    if cfg.adversarial_export:
        log.info("Initializing the adversarial prediction step")
        model.predict_step = model.predict_step_adversarial
        model.automatic_optimization = False

    elif cfg.SWA_export:
        log.info("Initializing SWA export")
        model.init_SWA_export()
    elif cfg.SWAG_export:
        log.info("Initializing SWAG export")
        model.init_SWAG_export()

    if orig_cfg.datamodule.data_conf.dataset_type == "TopTag":
        orig_cfg.datamodule.data_conf.path = (
            "/srv/beegfs/scratch/groups/rodem/datasets/TopTagging"
        )

    elif orig_cfg.datamodule.data_conf.dataset_type == "JetNet":
        orig_cfg.datamodule.data_conf.path = (
            "/srv/beegfs/scratch/groups/rodem/datasets/jetnet"
        )

    # Cycle through the datasets and create the dataloader
    for key, dataset in cfg.datamodule.data_conf.datasets.items():
        log.info(f"Instantiating the data module for {dataset}")
        orig_cfg.datamodule.data_conf.datasets = {"c0": dataset}
        datamodule = hydra.utils.instantiate(orig_cfg.datamodule)

        log.info("Instantiating the trainer")
        if cfg.adversarial_export:
            log.info("Initializing the adversarial trainer")
            trainer = hydra.utils.instantiate(orig_cfg.trainer, inference_mode=False)
        else:
            trainer = hydra.utils.instantiate(orig_cfg.trainer)

        log.info("Running the prediction loop")
        outputs = trainer.predict(model=model, datamodule=datamodule)

        log.info("Combining predictions across dataset")
        scores = list(outputs[0].keys())
        score_dict = {score: T.vstack([o[score] for o in outputs]) for score in scores}

        log.info("Saving outputs")
        output_dir = Path(orig_cfg.paths.full_path, "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_nature = "test"
        if cfg.datamodule.export_train:
            dataset_nature = "train"

        if cfg.adversarial_export:
            with h5py.File(
                output_dir
                / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_{dataset_nature}_adversarial.h5",
                mode="w",
            ) as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))
        elif cfg.SWA_export:
            with h5py.File(
                output_dir
                / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_{dataset_nature}_SWA.h5",
                mode="w",
            ) as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))
        elif cfg.SWAG_export:
            with h5py.File(
                output_dir
                / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_{dataset_nature}_SWAG.h5",
                mode="w",
            ) as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))
        else:
            with h5py.File(
                output_dir
                / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_{dataset_nature}.h5",
                mode="w",
            ) as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))


if __name__ == "__main__":
    main()
