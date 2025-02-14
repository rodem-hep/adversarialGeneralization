import numpy as np
import h5py
from pathlib import Path
import pyrootutils
import hydra
from omegaconf import DictConfig
import pandas as pd
from copy import deepcopy
from itertools import combinations
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)


@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="generate_hessian_latex_table.yaml",
)
def main(cfg: DictConfig) -> None:
    # Load the experiment configuration
    config_path = root / "configs"
    with open(f"{config_path}/exp_confs/adversarial.yaml", "r") as file:
        exp_config = yaml.safe_load(file)
        method_types = [method["experiment_name"] for method in exp_config]

    eigenvalue_loader = EigenValueLoader(path=cfg.output_dir, method_types=method_types)
    eigenvalue_loader.load_eigenvalues(
        trained_on_dataset_type=cfg.trained_on_dataset_type,
        evaluated_on_dataset_type=cfg.evaluated_on_dataset_type,
    )
    eigenvalues = eigenvalue_loader.get_eigenvalues()
    print(eigenvalues)

    # Generate LaTeX table
    latex_table = r"""
\begin{table}[h]
  \centering
  \caption{Largest Hessian eigenvalues for the different training methods and perturbation spaces \textcolor{darkgreen}{for models trained and evaluated on Pythia}. Lower values correlate with wider minimas.} 
  \label{tab:HessianEigenvalues}
  \resizebox{\linewidth}{!}{
      \begin{tabular}{lcccc}
          \toprule
          \textbf{Methods} & \multicolumn{2}{c}{\textbf{Feature-space}} & \multicolumn{2}{c}{\textbf{Weight-space}} \\
                            & \textbf{Hbb} & \textbf{QCD} & \textbf{Hbb} & \textbf{QCD} \\
          \midrule
"""

    # Find the smallest values
    min_values = {
        space: {key: float("inf") for key in ["Hbb", "QCD"]}
        for space in ["input", "weight"]
    }
    for spaces in eigenvalues.values():
        for space in ["input", "weight"]:
            for key, val in spaces[space].items():
                if val[0] < min_values[space][key]:
                    min_values[space][key] = val[0]

    for method, spaces in eigenvalues.items():
        latex_table += f"            \\textbf{{{method}}} & "
        latex_table += " & ".join(
            [
                f"$\\mathbf{{{val[0]}}} \\pm \\mathbf{{{val[1]}}}$"
                if val[0] == min_values[space][key]
                else f"${val[0]} \\pm {val[1]}$"
                for space in ["input", "weight"]
                for key, val in spaces[space].items()
            ]
        )
        latex_table += r" \\" + "\n"

    latex_table += r"""
            \bottomrule
        \end{tabular}
    }
\end{table}
"""

    output_dir = Path(cfg.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with open(
        f"{output_dir}/TR_{cfg.trained_on_dataset_type}_EV_{cfg.evaluated_on_dataset_type}_hessian_eigenvalues.txt",
        "w",
    ) as file:
        file.write(latex_table)

    print(latex_table)
    print("done")
    return


class EigenValueLoader:
    def __init__(
        self,
        path,
        method_types,
        datasets=["Hbb", "QCD"],
        spaces=["input", "weight"],
        rounding_precision=3,
    ):
        self.path = path
        self.method_types = method_types
        self.datasets = datasets
        self.spaces = spaces
        self.rounding_precision = rounding_precision

        self.eigenvalues = {}

    def load_eigenvalues(self, trained_on_dataset_type, evaluated_on_dataset_type):
        self.eigenvalues = {}
        for method_type in self.method_types:
            self.eigenvalues[method_type] = {}
            for space in self.spaces:
                self.eigenvalues[method_type][space] = {}
                for dataset in self.datasets:
                    eigenvalue, std = self.load_eigenvalue(
                        method_type=method_type,
                        trained_on_dataset_type=trained_on_dataset_type,
                        evaluated_on_dataset_type=evaluated_on_dataset_type,
                        dataset=dataset,
                        space=space,
                    )
                    self.eigenvalues[method_type][space][dataset] = (eigenvalue, std)

    def load_eigenvalue(
        self,
        method_type,
        trained_on_dataset_type,
        evaluated_on_dataset_type,
        dataset,
        space,
    ):
        assert space in ["input", "weight"]
        with open(
            f"{self.path}/{method_type}/TR_{trained_on_dataset_type}/EV_{evaluated_on_dataset_type}/{dataset}_hessian_largest_eigenvalue_{space}.txt",
            "r",
        ) as file:
            line = file.readline().strip()
            parts = line.split(" ")
            eigenvalue = float(parts[2])
            std = float(parts[4])

        rounded_eigenvalue = np.round(eigenvalue, self.rounding_precision)
        rounded_std = np.round(std, self.rounding_precision)

        return rounded_eigenvalue, rounded_std

    def get_eigenvalues(self):
        return deepcopy(self.eigenvalues)


if __name__ == "__main__":
    main()
