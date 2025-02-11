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
    config_name="gradientAscentPathTracing.yaml",
)
def main(cfg: DictConfig) -> None:
    tagger_dir = Path(
        "/home/users/r/rothenf3/workspace/Jettagging/jettagging/jobs/taggers"
    )

    taggers = [
        #######################################
        {
            "path": tagger_dir / "supervised" / "dense_test" / "outputs",
            "label": "dense",
            "score_name": "output",
            "linestyle": "solid",
            "color": "black",
        },
        {
            "path": tagger_dir / "supervised_sam" / "dense_test_sam" / "outputs",
            "label": "dense SAM",
            "score_name": "output",
            "linestyle": "solid",
            "color": "blue",
        },
        # {
        #     "path": tagger_dir / "supervised_adversarial_frac0.8_eps_0.007" / "dense_test_adversarial_frac0.8_eps_0.007" / "outputs",
        #     "label": "dense adversarial",
        #     "score_name": "output",
        #     "linestyle": "solid",
        #     "color": "red",
        # },
    ]

    datasets = cfg.datasets

    for dataset in datasets:
        for tagger in taggers:
            log.info(f"{dataset},{tagger['label']} Largest eigenvalue: ")

            largest_eigenvalue = torch.load(
                f'{tagger["path"]}/{dataset}_hessian_largest_eigenvalue.pt'
            )
            log.info(f"{largest_eigenvalue}")
            # # Plot the results
            # plt.plot(sorted_eigenvalues)
            # plt.xlabel("Eigenvalue Index")
            # plt.ylabel("Eigenvalue Value")
            # plt.title("Eigenvalues of Hessian Matrix")

            # output_dir = root / "plots"
            # plt.savefig(f'{output_dir}/HessianEigenValues_{dataset}_{tagger["label"]}')
            # plt.close()


if __name__ == "__main__":
    main()

