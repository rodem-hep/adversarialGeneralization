import numpy as np
import h5py
from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import hydra
from omegaconf import DictConfig
import pandas as pd
from copy import deepcopy
from itertools import combinations
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# from tabulate import tabulate
@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="plot_scores_array.yaml",
)
def main(cfg: DictConfig) -> None:
    # Save reference scores
    if cfg.plotReferenceScores:
        save_reference_scores(cfg, "AUC")
        save_reference_scores(cfg, "rejection")

    # Save method scores
    if cfg.plotScores:
        save_scores(cfg, "AUC")
        save_scores(cfg, "rejection")

    # Plot ratio heatmap
    if cfg.plotRatioHeatmap:
        plot_ratio_heatmap(cfg, "AUC", "RS3L0", "RS3L4")
        plot_ratio_heatmap(cfg, "rejection", "RS3L0", "RS3L4")

    return


def load_label_and_predictions(tag, evaluated_on_dataset_type, decor=False):
    tagger_path = Path(tag["path"], tag["name"], "outputs")
    tagger_score_name = tag["score_name"]

    predictions = []
    labels = []

    decor_suffix = "_decor" if decor else ""

    score_files = [
        f"{evaluated_on_dataset_type}_{class_name}_test{decor_suffix}.h5"
        for class_name in ["background", "signal"]
    ]

    for i, file_name in enumerate(score_files):
        file_path = tagger_path / file_name
        with h5py.File(file_path, "r") as f:
            predictions.append(f[tagger_score_name][:])
            labels += len(f[tagger_score_name]) * [i]

    predictions = np.concatenate(predictions)
    labels = np.array(labels)

    return labels, predictions


def get_score_value(labels, predictions, metric="AUC", signal_efficency=0.85):
    match metric:
        case "AUC":
            scores = roc_auc_score(labels, predictions)
        case "rejection":
            scores = rejection(labels, predictions, signal_efficency)
        case _:
            raise ValueError(f"Unknown metric {metric}")

    return scores


def false_positive_ratio(labels, predictions, signal_efficiency=0.85):
    fpr, tpr, _ = roc_curve(labels, predictions)

    signal_efficiency_index = np.abs(tpr - signal_efficiency).argmin()
    fpr = fpr[signal_efficiency_index]

    return fpr


def rejection(labels, predictions, signal_efficiency=0.85):
    fpr = false_positive_ratio(labels, predictions, signal_efficiency=signal_efficiency)

    return 1 / fpr


def load_configs(cfg):
    # Load train configuration
    with open(f"{cfg.config_path}/data_confs/trained_on.yaml", "r") as file:
        trained_on_data_config = yaml.safe_load(file)
        trained_on_dataset_types = [
            dataset["dataset_type"] for dataset in trained_on_data_config
        ]

    # Load evaluation configuration
    with open(f"{cfg.config_path}/data_confs/evaluated_on.yaml", "r") as file:
        evaluated_on_data_config = yaml.safe_load(file)
        evaluated_on_dataset_types = [
            dataset["dataset_type"] for dataset in evaluated_on_data_config
        ]

    # Load tagger list
    with open(
        f"{cfg.config_path}/taggers/generated/{cfg.exp_name}_{cfg.network_type}_trainedOnTaggerList.yaml",
        "r",
    ) as file:
        trained_tagger_list = yaml.safe_load(file)
    with open(
        f"{cfg.config_path}/taggers/generated/{cfg.exp_name}_{cfg.network_type}_evaluatedOnTaggerList.yaml",
        "r",
    ) as file:
        evaluated_tagger_list = yaml.safe_load(file)

    # Load the experiment configuration
    with open(f"{cfg.config_path}/exp_confs/{cfg.method_file_name}.yaml", "r") as file:
        exp_config = yaml.safe_load(file)
        method_types = [method["experiment_name"] for method in exp_config]

    return (
        trained_on_dataset_types,
        evaluated_on_dataset_types,
        method_types,
        trained_tagger_list,
        evaluated_tagger_list,
    )


def plot_ratio_heatmap(cfg, metric, dataset_type1, dataset_type2):
    (
        _,
        _,
        method_types,
        trained_tagger_list,
        _,
    ) = load_configs(cfg)

    method_types.insert(0, "optimal")

    # Initialize numpy array to store scores
    scores_array = np.zeros(
        (
            len(method_types),
            2,
            2,
        )
    )

    # TODO: This is not using dataset_type1 and dataset_type2 correctly
    tagger1 = deepcopy(trained_tagger_list[0])
    tagger2 = deepcopy(trained_tagger_list[1])

    for i in range(len(method_types)):
        if method_types[i] != "optimal":
            tagger1m = deepcopy(tagger1)
            tagger2m = deepcopy(tagger2)

            tagger1m["name"] = tagger1m["name"].replace("default", method_types[i])
            tagger1m["path"] = tagger1m["path"].replace("default", method_types[i])

            tagger2m["name"] = tagger2m["name"].replace("default", method_types[i])
            tagger2m["path"] = tagger2m["path"].replace("default", method_types[i])

        else:
            tagger1m = deepcopy(tagger2)
            tagger2m = deepcopy(tagger1)

        labelsT1E2, predictionsT1E2 = load_label_and_predictions(
            tagger1m, dataset_type2, False
        )
        labelsT2E1, predictionsT2E1 = load_label_and_predictions(
            tagger2m, dataset_type1, False
        )

        scores_array[i, 0, 0] = get_score_value(
            labelsT1E2, predictionsT1E2, metric=metric
        )
        scores_array[i, 1, 0] = get_score_value(
            labelsT2E1, predictionsT2E1, metric=metric
        )

        labelsT1E2, predictionsT1E2 = load_label_and_predictions(
            tagger1m, dataset_type2, True
        )
        labelsT2E1, predictionsT2E1 = load_label_and_predictions(
            tagger2m, dataset_type1, True
        )

        scores_array[i, 0, 1] = get_score_value(
            labelsT1E2, predictionsT1E2, metric=metric
        )
        scores_array[i, 1, 1] = get_score_value(
            labelsT2E1, predictionsT2E1, metric=metric
        )

    def plot_heatmap(decor=False):
        cmap = sns.color_palette("mako", as_cmap=True)
        decor_id = 1 if decor else 0
        decimals = ".3f"

        sns.set_theme(font_scale=2.0)
        plt.figure(figsize=(10, 10))

        # Calculate the performance increase ratio (0: as good as default, 1: as good as theoretical best)
        def ratio(score, default_score, theoretical_score):
            default_score = np.expand_dims(default_score, axis=0)
            theoretical_score = np.expand_dims(theoretical_score, axis=0)
            return (score - default_score) / (theoretical_score - default_score)

        ratio_array = ratio(
            scores_array[:, :, decor_id],
            scores_array[1, :, decor_id],
            scores_array[0, :, decor_id],
        )

        sns.heatmap(
            ratio_array,
            annot=True,
            fmt=decimals,
            xticklabels=["RS3L(0->4)", "RS3L(4->0)"],
            yticklabels=method_types,
            cbar=True,
            vmin=0,
            vmax=1,
            cmap=cmap,
            # linecolor="white",
            # linewidths=0.1,
            # annot_kws={"color": "black", "size": 11},
        )

        if not Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").exists():
            Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").mkdir(parents=True)

        decor_suffix = "_decor" if decor else ""
        plt.savefig(
            f"{cfg.exp_path}/{cfg.network_type}/scores_array/{metric}_heatmap_ratio_M_{cfg.method_file_name}{decor_suffix}.png"
        )
        plt.close()

        return

    plot_heatmap(decor=False)
    plot_heatmap(decor=True)

    return


def save_reference_scores(cfg, metric):
    """Save the scores for the reference model for each dataset in a numpy file"""

    _, evaluated_on_dataset_types, _, _, evaluated_on_tagger_list = load_configs(cfg)

    # Initialize numpy array to store scores
    scores_array = np.zeros(
        (len(evaluated_on_dataset_types), len(evaluated_on_dataset_types), 2)
    )

    for i in range(len(evaluated_on_dataset_types)):
        for j in range(len(evaluated_on_dataset_types)):
            labels, predictions = load_label_and_predictions(
                evaluated_on_tagger_list[i], evaluated_on_dataset_types[j], False
            )
            labels_decor, predictions_decor = load_label_and_predictions(
                evaluated_on_tagger_list[i], evaluated_on_dataset_types[j], True
            )
            scores_array[i, j, 0] = get_score_value(labels, predictions, metric=metric)
            scores_array[i, j, 1] = get_score_value(
                labels_decor, predictions_decor, metric=metric
            )

    # Save the scores to a numpy file
    if not Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").exists():
        Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").mkdir(parents=True)
    np.save(
        f"{cfg.exp_path}/{cfg.network_type}/scores_array/reference_{metric}.npy",
        scores_array,
    )

    def plot_heatmap(decor=False):
        cmap = sns.color_palette("mako", as_cmap=True)
        decimals = ".3f" if metric == "AUC" else ".1f"
        sns.set_theme(font_scale=2.0)
        plt.figure(figsize=(10, 10))

        decor_id = 1 if decor else 0
        sns.heatmap(
            scores_array[:, :, decor_id],
            annot=True,
            fmt=decimals,
            xticklabels=evaluated_on_dataset_types,
            yticklabels=evaluated_on_dataset_types,
            cbar=True,
            cmap=cmap,
        )

        plt.xlabel("Evaluated on dataset")
        plt.ylabel("Trained on dataset")

        if not Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").exists():
            Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").mkdir(parents=True)

        decor_suffix = "_decor" if decor else ""
        plt.savefig(
            f"{cfg.exp_path}/{cfg.network_type}/scores_array/reference_{metric}_heatmap{decor_suffix}.png"
        )
        plt.close()

        return

    plot_heatmap(decor=False)
    plot_heatmap(decor=True)

    return


def save_scores(cfg, metric):
    """Save all the non-decor and decor scores for each model for each dataset in a numpy file"""

    (
        trained_on_dataset_types,
        evaluated_on_dataset_types,
        method_types,
        trained_on_tagger_list,
        _,
    ) = load_configs(cfg)

    method_types.insert(0, "optimal")

    # Initialize numpy array to store scores
    scores_array = np.zeros(
        (
            len(trained_on_dataset_types),
            len(method_types),
            len(evaluated_on_dataset_types),
            2,
        )
    )

    def plot_heatmap(train_id=0, decor=False, plot_ratio=False):
        decor_id = 1 if decor else 0
        decimals = ".3f" if metric == "AUC" else ".1f"
        decimals = ".3f" if plot_ratio else decimals

        sns.set_theme(font_scale=2.0)
        plt.figure(figsize=(10, 10))

        scores_array_slice = scores_array[train_id, :, :, decor_id].copy()
        evaluated_on_dataset_types_slice = evaluated_on_dataset_types.copy()

        if plot_ratio:
            scores_array_slice = np.delete(scores_array_slice, train_id, axis=1)
            evaluated_on_dataset_types_slice.remove(trained_on_dataset_types[train_id])

            # Calculate the performance increase ratio (0: as good as default, 1: as good as theoretical best)
            def ratio(score, default_score, theoretical_score):
                default_score = np.expand_dims(default_score, axis=0)
                theoretical_score = np.expand_dims(theoretical_score, axis=0)
                return (score - default_score) / (theoretical_score - default_score)

            scores_array_slice = ratio(
                scores_array_slice[:, :],
                scores_array_slice[1, :],
                scores_array_slice[0, :],
            )

        cmap = sns.color_palette("mako", as_cmap=True)
        if plot_ratio:
            sns.heatmap(
                scores_array_slice,
                annot=True,
                fmt=decimals,
                xticklabels=evaluated_on_dataset_types_slice,
                yticklabels=method_types,
                cbar=True,
                vmin=0,
                vmax=1,
                cmap=cmap,
            )
        else:
            sns.heatmap(
                scores_array_slice,
                annot=True,
                fmt=decimals,
                xticklabels=evaluated_on_dataset_types_slice,
                yticklabels=method_types,
                cbar=True,
                cmap=cmap,
            )

        plt.xlabel("Evaluated on dataset")
        plt.ylabel("Method type")

        if not Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").exists():
            Path(f"{cfg.exp_path}/{cfg.network_type}/scores_array").mkdir(parents=True)

        decor_suffix = "_decor" if decor else ""
        ratio_suffix = "_ratio" if plot_ratio else ""
        plt.savefig(
            f"{cfg.exp_path}/{cfg.network_type}/scores_array/{metric}_heatmap_{trained_on_dataset_types[train_id]}{ratio_suffix}_M_{cfg.method_file_name}{decor_suffix}.png"
        )
        plt.close()

        return

    for i in range(len(trained_on_dataset_types)):
        for j in range(len(method_types)):
            # tagger = deepcopy(tagger_list[i])
            # if method_types[j] != "optimal":
            #     # Change "default" to "method_type" in the tagger dictionary
            #     tagger["name"] = tagger["name"].replace("default", method_types[j])
            #     tagger["path"] = tagger["path"].replace("default", method_types[j])

            for k in range(len(evaluated_on_dataset_types)):
                tagger = deepcopy(trained_on_tagger_list[i])
                if method_types[j] == "optimal":
                    tagger["name"] = tagger["name"].replace(
                        trained_on_dataset_types[i], evaluated_on_dataset_types[k]
                    )
                    tagger["path"] = tagger["path"].replace(
                        trained_on_dataset_types[i], evaluated_on_dataset_types[k]
                    )
                else:
                    tagger["name"] = tagger["name"].replace("default", method_types[j])
                    tagger["path"] = tagger["path"].replace("default", method_types[j])

                # Non-decor AUC
                labels, predictions = load_label_and_predictions(
                    tagger, evaluated_on_dataset_types[k], False
                )
                scores_array[i, j, k, 0] = get_score_value(
                    labels, predictions, metric=metric
                )
                # Decor AUC
                labels, predictions = load_label_and_predictions(
                    tagger, evaluated_on_dataset_types[k], True
                )
                scores_array[i, j, k, 1] = get_score_value(
                    labels, predictions, metric=metric
                )

        plot_heatmap(train_id=i, decor=False, plot_ratio=False)
        # plot_heatmap(train_id=i, decor=False, plot_ratio=True)
        plot_heatmap(train_id=i, decor=True, plot_ratio=False)
        # plot_heatmap(train_id=i, decor=True, plot_ratio=True)

    return


if __name__ == "__main__":
    main()
