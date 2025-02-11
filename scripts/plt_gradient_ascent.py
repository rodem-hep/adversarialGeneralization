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
    config_name="gradientAscent.yaml",
)
def main(cfg: DictConfig) -> None:
    markers = ["o", "^", "s", "p", "h"]  # circle, triangle, square, pentagon, hexagon

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

    # Plot WEIGHT tracing
    for i, tagger in enumerate(taggers):
        log.info(f"{tagger['label']}")

        loss_tensor = torch.tensor(
            torch.load(
                f'{tagger["path"]}/{tagger["name"]}/outputs/{cfg.evaluated_on_dataset_type}_gradient_ascent_weight_loss_tensor.pt'
            )
        )
        std_tensor = torch.tensor(
            torch.load(
                f'{tagger["path"]}/{tagger["name"]}/outputs/{cfg.evaluated_on_dataset_type}_gradient_ascent_weight_std_tensor.pt'
            )
        )
        steps = range(len(loss_tensor))

        loss_tensor = torch.add(loss_tensor, -1 * torch.tensor(loss_tensor.numpy()[0]))

        # Plot the results
        plt.errorbar(
            steps,
            loss_tensor.numpy(),
            yerr=std_tensor.numpy(),
            fmt=markers[i % len(markers)],
            color=tagger["plot_kwargs"]["color"],
            label=tagger["label"],
        )
        plt.plot(
            steps,
            loss_tensor.numpy(),
            color=tagger["plot_kwargs"]["color"],
            linestyle=tagger["plot_kwargs"]["linestyle"],
        )

    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("Zero-shifted Loss")
    # plt.grid(True)

    plt.savefig(
        f"{cfg.output_dir}/gradientAscentWeight_TR_{cfg.trained_on_dataset_type}_EV_{cfg.evaluated_on_dataset_type}.pdf"
    )
    plt.close()

    # Plot INPUT tracing
    for i, tagger in enumerate(taggers):
        log.info(f"{tagger['label']}")

        loss_tensor = torch.tensor(
            torch.load(
                f'{tagger["path"]}/{tagger["name"]}/outputs/{cfg.evaluated_on_dataset_type}_gradient_ascent_input_loss_tensor.pt'
            )
        )
        std_tensor = torch.tensor(
            torch.load(
                f'{tagger["path"]}/{tagger["name"]}/outputs/{cfg.evaluated_on_dataset_type}_gradient_ascent_input_std_tensor.pt'
            )
        )
        steps = range(len(loss_tensor))

        loss_tensor = torch.add(loss_tensor, -1 * torch.tensor(loss_tensor.numpy()[0]))

        # Plot the results
        plt.errorbar(
            steps,
            loss_tensor.numpy(),
            yerr=std_tensor.numpy(),
            fmt=markers[i % len(markers)],
            color=tagger["plot_kwargs"]["color"],
            label=tagger["label"],
        )
        plt.plot(
            steps,
            loss_tensor.numpy(),
            color=tagger["plot_kwargs"]["color"],
            linestyle=tagger["plot_kwargs"]["linestyle"],
        )

    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("Zero-shifted Loss")
    # plt.grid(True)

    plt.savefig(
        f"{cfg.output_dir}/gradientAscentInput_TR_{cfg.trained_on_dataset_type}_EV_{cfg.evaluated_on_dataset_type}.pdf"
    )
    plt.close()


if __name__ == "__main__":
    main()
