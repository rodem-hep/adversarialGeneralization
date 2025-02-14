import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
import wandb

import hydra
import pytorch_lightning as pl
import torch as T
from omegaconf import DictConfig

import os
from pathlib import Path

from mattstools.mattstools.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)

from franckstools.franckstools.swag import SWAGWrapper
from franckstools.franckstools.adversarial_attack import AdversarialWrapper
from franckstools.franckstools.sam import SAMWrapper

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    wandb_key = open("/wandb/wandb.key", "r").read()
    wandb.login(key=wandb_key)

    log.info("Setting up full job config")
    if cfg.full_resume:
        cfg = reload_original_config(cfg)

    # Add log_mass to the coordinates if needed
    if hasattr(cfg, "append_log_mass") and cfg.append_log_mass:
        log.info("Appending log_mass to the coordinates")
        if "log_mass" not in cfg.datamodule.data_conf.coordinates.high:
            cfg.datamodule.data_conf.coordinates.high.append("log_mass")

    print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.precision:
        log.info(f"Setting matrix precision to: {cfg.precision}")
        T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    if hasattr(datamodule, "n_nodes"):
        model = hydra.utils.instantiate(
            cfg.model,
            inpt_dim=datamodule.get_dims(),
            n_nodes=datamodule.n_nodes,
            n_classes=datamodule.n_classes,
        )
    else:
        model = hydra.utils.instantiate(cfg.model)

    if hasattr(cfg, "use_adversarial_samples") and cfg.use_adversarial_samples:
        log.info("Instantiating Adversarial wrapper")
        model = AdversarialWrapper(
            model,
            adversarial_scheduler_config=cfg.adversarial_scheduler_config,
            attack_type=cfg.attack_type,
            epsilon=cfg.epsilon,
            PGD_num_steps=cfg.PGD_num_steps,
            PGD_step_size=cfg.PGD_step_size,
        )

    if (
        hasattr(cfg, "use_sharpness_aware_minimization")
        and cfg.use_sharpness_aware_minimization
    ):
        log.info("Instantiating SAM wrapper")
        model = SAMWrapper(
            model,
            rho=cfg.get("rho"),
            method_type=cfg.get("method_type"),
            sparsity=cfg.get("sparsity"),
            num_samples=cfg.get("num_samples"),
            update_freq=cfg.get("update_freq"),
            drop_rate=cfg.get("drop_rate"),
            drop_strategy=cfg.get("drop_strategy"),
            growth_strategy=cfg.get("growth_strategy"),
            T_start=cfg.get("T_start"),
            T_end=cfg.get("T_end"),
        )

    if hasattr(cfg, "use_SWAG") and cfg.use_SWAG:
        log.info("Instantiating SWAG wrapper")
        model = SWAGWrapper(
            model,
            max_samples_to_record=cfg.max_samples_to_record,
            number_epoch_before_new_record=cfg.number_epoch_before_new_record,
            scale=cfg.scale,
            isRecordCyclic=cfg.isRecordCyclic,
            cycle_period=cfg.cycle_period,
        )

        if hasattr(cfg, "transfer_ckpt_path") and cfg.transfer_ckpt_path != None:
            # If transfer checkpoint path contains ".last.ckpt". Replace it with best
            transfer_ckpt_path = str(
                sorted(
                    Path(cfg.transfer_ckpt_path).glob(f"best*.ckpt"),
                    key=os.path.getmtime,
                )[-1]
            )
            print(f"Transfering checkpoint from {transfer_ckpt_path}")
            model.transfer_checkpoint(transfer_ckpt_path)

    log.info(model)

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating the trainer")
    if (
        hasattr(cfg, "use_sharpness_aware_minimization")
        and cfg.use_sharpness_aware_minimization
        and cfg.method_type == "SSAMD"
    ):
        log.info(f"Using SSAMD, setting max_epochs to T_end = {cfg.T_end}")
        trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=loggers, max_epochs=cfg.T_end
        )
    else:
        trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=loggers
        )

    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
