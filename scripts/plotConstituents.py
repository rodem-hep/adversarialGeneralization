import vector
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging

import numpy as np
import hydra
import torch as T
from omegaconf import DictConfig
import yaml

from src.datamodules.dataset import JetData

import matplotlib.pyplot as plt


log = logging.getLogger(__name__)


def plot_pt_ratio(cfg) -> None:
    max_n_constituents = 60
    n_samples = 10000

    # Load dataset configuration
    config_path = "/home/users/r/rothenf3/workspace/Jettagging/jettagging/configs/"

    with open(f"{config_path}/data_confs/evaluated_on.yaml", "r") as file:
        data_config = yaml.safe_load(file)

    dataset_types = [dataset["dataset_type"] for dataset in data_config]

    for dataset_type in dataset_types:
        # Create the empty list of pt_ratios
        pt_ratios = []

        cfg.datamodule.data_conf.dataset_type = dataset_type
        cfg.datamodule.data_conf.datasets.c0 = "QCD"
        cfg.datamodule.data_conf.datasets.c1 = "Hbb"

        # Create the JetData
        log.info(f"Instantiating the JetData for {max_n_constituents} constituents.")
        cfg.datamodule.data_conf.n_csts = max_n_constituents
        cfg.datamodule.data_conf.n_jets = n_samples

        dataset = JetData(dset="train", **cfg.datamodule.data_conf)
        pt_ratios = dataset.get_pT_ratio()

        # Plot the pt_ratios
        log.info(f"Plotting the pT ratios for {dataset_type} dataset.")
        plt.plot(range(max_n_constituents), pt_ratios, label=dataset_type)
    plt.xlabel("Number of Constituents")
    plt.ylabel("Cumulative transverse momentum pT Ratio")
    plt.legend(loc="lower right")
    # Save the plot
    log.info("Saving the plot")
    plt.savefig(f"{cfg.exp_path}/plots/pT_ratio.png")
    plt.close()

    return


def plot_constituent_distribution(cfg) -> None:
    log.info(f"Plotting constituent distributions")

    config_path = "/home/users/r/rothenf3/workspace/Jettagging/jettagging/configs/"

    with open(f"{config_path}/data_confs/evaluated_on.yaml", "r") as file:
        data_config = yaml.safe_load(file)
    dataset_types = [dataset["dataset_type"] for dataset in data_config]

    max_n_constituents = 100
    n_samples = 100000

    cfg.datamodule.data_conf.datasets.c0 = "QCD"
    cfg.datamodule.data_conf.datasets.c1 = "Hbb"
    cfg.datamodule.data_conf.n_csts = max_n_constituents
    cfg.datamodule.data_conf.n_jets = n_samples

    average_n_constituents = []

    for dataset_type in dataset_types:
        cfg.datamodule.data_conf.dataset_type = dataset_type

        # Create the JetData
        log.info(
            f"Instantiating the JetData for {max_n_constituents} constituents for {dataset_type} dataset."
        )
        dataset = JetData(dset="train", **cfg.datamodule.data_conf)
        csts_hist, bins = dataset.get_constituents_hist(max_events=n_samples)

        plt.stairs(csts_hist, bins, label=dataset_type)

        # Get average number of constituents
        average_n_constituents.append(
            np.mean(np.sum(np.append(csts_hist, 0) * (bins + 1)))
        )

    plt.xlabel("Number of Constituents")
    plt.ylabel("Normalized distribution")
    plt.legend()
    plt.savefig(f"{cfg.exp_path}/plots/constituent_distribution.png")
    plt.close()

    # Print average number of constituents
    for dataset_type, avg_n in zip(dataset_types, average_n_constituents):
        log.info(f"Average number of constituents for {dataset_type}: {avg_n}")
        log.info("------------------------------")

    return


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    if hasattr(cfg, "plotPTRatio") and cfg.plotPTRatio:
        plot_pt_ratio(cfg)

    if hasattr(cfg, "plotConstituentDistribution") and cfg.plotConstituentDistribution:
        plot_constituent_distribution(cfg)

    return


if __name__ == "__main__":
    main()
