import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
import hydra

import torch

import matplotlib.pyplot as plt

from omegaconf import DictConfig
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="avgLossLandscape.yaml",
)
def main(cfg: DictConfig) -> None:
    log.info("Creating taggers list")
    taggers = []
    taggers.append(
        {
            "label": "default",
            "path": f"{cfg.exp_path}/taggers/supervised_{cfg.trained_on_dataset_type}_default",
            "name": f"{cfg.network_type}_{cfg.trained_on_dataset_type}_default",
            "plot_kwargs": {"color": "black", "linestyle": "dashed"},
        }
    )
    for i in range(len(cfg.method_types)):
        taggers.append(
            {
                "label": f"{cfg.method_types[i]}",
                "path": f"{cfg.exp_path}/taggers/supervised_{cfg.trained_on_dataset_type}_{cfg.method_types[i]}",
                "name": f"{cfg.network_type}_{cfg.trained_on_dataset_type}_{cfg.method_types[i]}",
                "plot_kwargs": {"color": cfg.color_list[i], "linestyle": "solid"},
            }
        )

    for tagger in taggers:
        log.info(f"{tagger['label']}")

        loss_tensor = torch.load(
            f'{tagger["path"]}/{tagger["name"]}/outputs/{cfg.evaluated_on_dataset_type}_avgLossIncrease.pt'
        )
        distances = torch.load(
            f'{tagger["path"]}/{tagger["name"]}/outputs/{cfg.evaluated_on_dataset_type}_distances.pt'
        )

        loss_mean = torch.mean(loss_tensor, dim=1)
        loss_std = torch.std(loss_tensor, dim=1)

        loss_mean = torch.add(loss_mean, -1 * torch.tensor(loss_mean.numpy()[0]))

        # Plot the results
        plt.plot(
            distances.numpy(),
            loss_mean.numpy(),
            color=tagger["plot_kwargs"]["color"],
            linestyle=tagger["plot_kwargs"]["linestyle"],
            label=tagger["label"],
        )
        plt.fill_between(
            distances.numpy(),
            np.maximum(loss_mean.numpy() - loss_std.numpy(), 0),
            loss_mean + loss_std,
            alpha=0.2,
            color=tagger["plot_kwargs"]["color"],
        )

    plt.legend()
    plt.xlabel("Weight Deviation")
    plt.ylabel("Zero-shifted Loss")
    # plt.grid(True)

    plt.savefig(
        f"{cfg.output_dir}/avgLossLandscape_TR_{cfg.trained_on_dataset_type}_EV_{cfg.evaluated_on_dataset_type}.pdf"
    )
    plt.close()


if __name__ == "__main__":
    main()
