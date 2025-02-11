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

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="export_jetClass.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(cfg, get_best=cfg.get_best)
    if cfg.adversarial_export:
        orig_cfg.model.use_adversarial_samples = True

    log.info("Loading best checkpoint")
    if (hasattr(orig_cfg, "use_SWAG") and orig_cfg.use_SWAG):
        log.info("Instantiating the underlying model, datamodule and SWAGWrapper")
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
            
        model = SWAGWrapper(model)
        checkpoint = T.load(orig_cfg.ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model_class = hydra.utils.get_class(orig_cfg.model._target_)
        model = model_class.load_from_checkpoint(orig_cfg.ckpt_path)

    if cfg.adversarial_export:
        log.info("Initializing the adversarial prediction step")
        model.predict_step = model.predict_step_adversarial
        model.automatic_optimization = False

    orig_cfg.datamodule.data_conf.dataset_type = cfg.datamodule.data_conf.dataset_type
            
    if orig_cfg.datamodule.data_conf.dataset_type != "JetClass":
        raise ValueError('export_jetClass.py requires dataset_type to be JetClass')

    #Change orig_cfg to use JetClass DataLoader
    orig_cfg.datamodule = cfg.datamodule

    # Cycle through the datasets and create the dataloader
    for dataset in cfg.datamodule.data_conf.iterator.processes:
        log.info(f"Instantiating the data module for {dataset}")
        orig_cfg.datamodule.data_conf.iterator.processes = {dataset:cfg.datamodule.data_conf.iterator.processes[dataset]}
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
        if cfg.adversarial_export:
            with h5py.File(output_dir / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_test_adversarial.h5", mode="w") as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))
        elif cfg.SWA_export:
            with h5py.File(output_dir / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_test_SWA.h5", mode="w") as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))
        elif cfg.SWAG_export:
            with h5py.File(output_dir / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_test_SWAG.h5", mode="w") as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))
        else:
            with h5py.File(output_dir / f"{cfg.datamodule.data_conf.dataset_type}_{dataset}_test.h5", mode="w") as file:
                for score in scores:
                    file.create_dataset(score, data=to_np(score_dict[score]))

if __name__ == "__main__":
    main()
