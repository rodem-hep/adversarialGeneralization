# Franck Rothen plotting_FR.py
# Don't want to modify matthews code. I just place this here for now.

import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

import hydra
from omegaconf import DictConfig
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from src.datasets_utils import DatasetManager

log = logging.getLogger(__name__)

from pathlib import Path
from src.datamodules.dataset import JetData


def plot_score_distributions(
    output_dir: Path | str,
    tagger: dict,
    files_dict: dict,
    dataset_types: list,
    trained_on_dataset_type: str,
    dset: str,
    quantileslist: list = None,
) -> None:
    log.info("Plotting score distributions")

    x_bins = np.linspace(-10, 10, 100)
    x_decor_bins = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(2, len(dataset_types), figsize=(10 * len(dataset_types), 10))
    for i in range(len(dataset_types)):
        scores = []
        decor_scores = []
        # Get the list of files
        for j in range(len(files_dict[dataset_types[i]])):
            ax[0, i].set_title(f"{dataset_types[i]} score distribution")
            ax[1, i].set_title(f"{dataset_types[i]} decor score distribution")

            # Load the scores from the tagger
            tagger.score_name = "output"
            scores.append(
                DatasetManager().load_scores(
                    tagger, dataset_types[i], files_dict[dataset_types[i]][j], dset=dset
                )
            )
            score_hist, _ = np.histogram(scores[j], bins=x_bins)

            # Plot the score distributions
            ax[0, i].stairs(score_hist, x_bins, label=files_dict[dataset_types[i]][j])
            ax[0, i].set_xlabel("Score")

            # Load the decor scores from the tagger
            tagger.score_name = "output"
            decor_scores.append(
                DatasetManager().load_decor_scores(
                    tagger, dataset_types[i], files_dict[dataset_types[i]][j], dset=dset
                )
            )
            decor_hist, _ = np.histogram(decor_scores[j], bins=x_decor_bins)
            # Plot the decor score distributions
            ax[1, i].stairs(
                decor_hist, x_decor_bins, label=files_dict[dataset_types[i]][j]
            )
            ax[1, i].set_xlabel("Score")

        if quantileslist is not None:
            # Add quantiles
            text_heights = np.linspace(0.95, 0.75, len(quantileslist) + 1)[:-1]

            # Increase ylim
            ax[0, i].set_ylim(top=ax[0, i].get_ylim()[1] * 1.5)
            ax[1, i].set_ylim(top=ax[1, i].get_ylim()[1] * 1.5)

            for quantile, text_height in zip(quantileslist, text_heights):
                if quantile == 0:
                    continue

                quantile_value = np.quantile(np.concatenate(scores), quantile)
                ax[0, i].axvline(
                    quantile_value,
                    color="black",
                    linestyle="--",
                    ymax=text_height - 0.05,
                    alpha=0.3,
                )
                ax[0, i].text(
                    quantile_value,
                    ax[0, i].get_ylim()[1] * text_height,
                    f"Q:{quantile}",
                    color="black",
                    ha="center",
                    va="top",
                    clip_on=True,
                )

                quantile_value_decor = np.quantile(
                    np.concatenate(decor_scores), quantile
                )
                ax[1, i].axvline(
                    quantile_value_decor,
                    color="black",
                    linestyle="--",
                    ymax=text_height - 0.05,
                    alpha=0.3,
                )
                ax[1, i].text(
                    quantile_value_decor,
                    ax[1, i].get_ylim()[1] * text_height,
                    f"Q: {quantile}",
                    color="black",
                    ha="center",
                    va="top",
                    clip_on=True,
                )

        # Add legend
        for i in range(len(dataset_types)):
            ax[0, i].legend()
            ax[1, i].legend()

    log.info(
        f"Saving score distributions to {output_dir}/{trained_on_dataset_type}_score_distributions.png"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f"{output_dir}/{trained_on_dataset_type}_score_distributions.png")

    return


def plot_mass_distribution(cfg) -> None:
    log.info(f"Plotting mass distribution of dataset {cfg.dataset_type}")

    x_bins = np.linspace(0, 300, 100)

    # Load the masses from the original HDF files: high lvl = pt, eta, phi, mass
    print(" - Loading mass...")
    for i in range(len(cfg.files)):
        print(f"Loading {cfg.files[i]}")
        mass = DatasetManager().load_mass(
            dataset_type=cfg.dataset_type, file_name=cfg.files[i], dset=cfg.dset
        )
        mass_hist, _ = np.histogram(mass, bins=x_bins)
        plt.stairs(mass_hist, x_bins, label=cfg.files[i])

    plt.xlabel("mass [GeV]")
    plt.legend()
    log.info(
        f"Saving mass distributions to {cfg.output_dir}/{cfg.dataset_type}_mass_distributions.png"
    )
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    plt.savefig(f"{cfg.output_dir}/{cfg.dataset_type}_mass_distribution.png")
    plt.close()

    return


def plot_mass_distribution_comparison(cfg) -> None:
    log.info(f"Plotting mass distribution comparison")

    xbins = np.linspace(0, 300, 100)
    trained_on_dataset_type = ["RS3L0", "RS3L4"]
    color_map = {cfg.files[j]: color for j, color in enumerate(["blue", "orange"])}
    linestyle_map = {
        trained_on_dataset_type[i]: linestyle for i, linestyle in enumerate(["-", "--"])
    }

    for i in range(len(trained_on_dataset_type)):
        for j in range(len(cfg.files)):
            mass = DatasetManager().load_mass(
                dataset_type=trained_on_dataset_type[i],
                file_name=cfg.files[j],
                dset=cfg.dset,
            )
            mass_hist, _ = np.histogram(mass, bins=xbins)
            plt.stairs(
                mass_hist,
                xbins,
                label=f"{trained_on_dataset_type[i]} {cfg.files[j]}",
                color=color_map[cfg.files[j]],
                linestyle=linestyle_map[trained_on_dataset_type[i]],
            )

    plt.xlabel("mass [GeV]")
    plt.legend()
    log.info(
        f"Saving mass distributions to {cfg.output_dir}/mass_distributions_comparison_RS3L0_RS3L4.png"
    )
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    plt.savefig(f"{cfg.output_dir}/mass_distributions_comparison_RS3L0_RS3L4.png")
    plt.close()

    return


def plot_all_distribution(cfg) -> None:
    log.info(f"Plotting all distributions")

    config_path = "/home/users/r/rothenf3/workspace/Jettagging/jettagging/configs/"

    with open(f"{config_path}/data_confs/evaluated_on.yaml", "r") as file:
        data_config = yaml.safe_load(file)
    dataset_types = [dataset["dataset_type"] for dataset in data_config]

    n_samples = 100000

    cfg.datamodule.data_conf.datasets.c0 = "QCD"
    cfg.datamodule.data_conf.datasets.c1 = "Hbb"
    cfg.datamodule.data_conf.n_csts = 40
    cfg.datamodule.data_conf.n_jets = n_samples

    for dataset_type in dataset_types:
        cfg.datamodule.data_conf.dataset_type = dataset_type

        # Create the JetData
        log.info(
            f"Instantiating the JetData for {max_n_constituents} constituents for {dataset_type} dataset."
        )
        dataset = JetData(dset="train", **cfg.datamodule.data_conf)

    return


def plot_env_mass_distributions(
    env_train_set,
    n_bins,
    mus_list=None,
    sigmas_list=None,
    output_dir: Path
    | str = "/home/users/r/rothenf3/workspace/Jettagging/jettagging/plots",
) -> None:
    label2name = {
        0: "QCD",
        1: "TTbar",
    }  # TODO: Make this more generall in case of case where n_class > 2

    x_bins = np.linspace(0, 250, n_bins)
    fig, ax = plt.subplots(len(env_train_set), 1, figsize=(8, 6 * len(env_train_set)))

    for i in range(len(env_train_set)):
        log.info(f"Plotting mass distributions of environment {i}")

        masses = env_train_set[i].dataset.get_masses()[env_train_set[i].indices]
        labels = env_train_set[i].dataset.get_labels()[env_train_set[i].indices]

        for j in range(len(np.unique(labels))):
            mass_hist, _ = np.histogram(masses[labels == j], bins=x_bins)
            ax[i].stairs(mass_hist, x_bins, label=f"{label2name[j]}", linewidth=2)
            # plot scaled gaussian curve
            if mus_list is not None and sigmas_list is not None:
                if len(mus_list) > i:
                    ax[i].plot(
                        x_bins,
                        np.max(mass_hist)
                        * np.exp(
                            -0.5 * ((x_bins - mus_list[i][j]) / sigmas_list[i][j]) ** 2
                        ),
                        label=f"Gaussian {label2name[j]}",
                        linestyle="--",
                        linewidth=2,
                    )
                else:
                    ax[i].plot([0], [0])

            ax[i].set_xlabel("Mass [GeV]", fontsize=12)
            ax[i].set_ylabel("Counts", fontsize=12)
            ax[i].legend(fontsize=10)
            ax[i].tick_params(axis="both", which="major", labelsize=10)
            ax[i].grid(True, linestyle="--", alpha=0.5)

    # if file folder does not exist, create it
    if not os.path.exists(f"{output_dir}/env_mass_distributions"):
        os.makedirs(f"{output_dir}/env_mass_distributions")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/env_mass_distributions/environments_mass_distribution.png",
        dpi=300,
    )
    plt.close()

    return


# def plot_mass_distribution(
#     output_dir: Path | str,
#     taggers: list,
#     files: list,
#     dataset_type: str,
#     dset: str,
# ) -> None:

#     log.info("Plotting mass distributions")

#     x_bins = np.linspace(0, 250, 100)

#     fig, ax = plt.subplots(2, len(taggers), figsize=(10*len(taggers), 10))

#     for i in range(len(taggers)):
#         try:
#             # Get the list of files
#             for j in range(len(files)):
#                 ax[0,i].set_title(f"{tagger.label} mass distribution")
#                 ax[1,i].set_title(f"{tagger.label} decor mass distribution")

#                 # Load the masses from the tagger
#                 mass = DatasetManager().load_mass(dataset_type, files[j], dset=dset)
#                 mass_hist, _ = np.histogram(mass, bins=x_bins)

#                 # Plot the score distributions
#                 ax[0,i].stairs(mass_hist, x_bins, label=files[j])
#                 ax[0,i].set_xlabel("mass [GeV]")

#             # Get the decor masses from the tagger
#             # First, load the decor scores from the tagger
#             tagger.score_name = "ext_decor_output"
#             decor_scores = []
#             mass = []
#             file_len_id = [0]
#             for j in range(len(files)):
#                 decor_scores.append(DatasetManager().load_scores(tagger, dataset_type, files[j], dset=dset))
#                 mass.append(DatasetManager().load_mass(dataset_type, files[j], dset=dset))
#                 file_len_id.append(file_len_id[j]+len(decor_scores[j]))

#             decor_scores = np.concatenate(decor_scores)
#             mass = np.concatenate(mass)

#             sorted_indices = np.argsort(decor_scores)
#             decor_mass = mass[sorted_indices]

#             for j in range(len(files)):
#                 decor_mass_hist, _ = np.histogram(decor_mass[file_len_id[j]:file_len_id[j+1]], bins=x_bins)

#                 # Plot the decor score distributions
#                 ax[1,i].stairs(decor_mass_hist, x_bins, label=files[j])
#                 ax[1,i].set_xlabel("mass [GeV]")

#         except Exception as e:
#             print(f"Failed for {tagger.label}")
#             print(e)
#             print()

#     # Add legend
#     for i in range(len(taggers)):
#         ax[0,i].legend()
#         ax[1,i].legend()

#     log.info(f"Saving mass distributions to {output_dir}/mass_distributions.png")
#     plt.savefig(f'{output_dir}/mass_distributions.png')

#     return


# QUANTILES ONLY ON BACKGROUND
def plot_mass_distribution_quantiles(
    output_dir: Path | str,
    tagger: dict,
    files: list,
    trained_on_dataset_type: str,
    evaluated_on_dataset_type: str,
    dset: str,
    quantileslist: list,
) -> None:
    log.info("Plotting quantile mass distributions")

    x_bins = np.linspace(0, 250, 100)

    fig, ax = plt.subplots(len(quantileslist), 2, figsize=(10, 5 * len(quantileslist)))

    # Get the list of files
    for i in range(len(files)):
        # Load the scores from the tagger
        tagger.score_name = "output"
        scores = DatasetManager().load_scores(
            tagger, evaluated_on_dataset_type, files[i], dset=dset
        )
        # Load the masses from the tagger
        mass = DatasetManager().load_mass(
            evaluated_on_dataset_type, files[i], dset=dset
        )

        sorted_indices = np.argsort(scores)
        scores = scores[sorted_indices]
        mass = mass[sorted_indices]

        for j in range(len(quantileslist)):
            ax[j, 0].set_title(f"mass distribution for quantile {quantileslist[j]}")
            ax[j, 1].set_title(
                f"decor mass distribution for quantile {quantileslist[j]}"
            )

            q_mass = mass[int(len(mass) * quantileslist[j]) :]
            q_mass_hist, _ = np.histogram(q_mass, bins=x_bins)

            # Plot the score distributions
            ax[j, 0].stairs(q_mass_hist, x_bins, label=files[i])
            ax[j, 0].set_xlabel("mass [GeV]")

    # Get the decor masses from the tagger
    # First, load the decor scores from the tagger
    # tagger.score_name = "ext_decor_output"
    decor_scores = []
    decor_mass = []
    new_labels = []
    for i in range(len(files)):
        decor_scores.append(
            DatasetManager().load_decor_scores(
                tagger, evaluated_on_dataset_type, files[i], dset=dset
            )
        )
        decor_mass.append(
            DatasetManager().load_mass(evaluated_on_dataset_type, files[i], dset=dset)
        )
        new_labels.append(np.ones(len(decor_scores[i])) * i)

    for i in range(len(quantileslist)):
        # get the quantiles for the background
        q_value = np.quantile(decor_scores[0], quantileslist[i])

        # filter the decor scores and masses
        decor_scores_i = np.concatenate(decor_scores)
        decor_mass_i = np.concatenate(decor_mass)
        new_labels_i = np.concatenate(new_labels)

        decor_mass_i = decor_mass_i[decor_scores_i > q_value]
        new_labels_i = new_labels_i[decor_scores_i > q_value]
        decor_scores_i = decor_scores_i[decor_scores_i > q_value]

        for j in range(len(files)):
            decor_mass_i_j = decor_mass_i[new_labels_i == j]
            q_decor_mass_i_j = decor_mass_i_j[
                int(len(decor_mass_i_j) * quantileslist[i]) :
            ]
            q_decor_mass_hist_i_j, _ = np.histogram(q_decor_mass_i_j, bins=x_bins)

            # Plot the score distributions
            ax[i, 1].stairs(q_decor_mass_hist_i_j, x_bins, label=files[j])
            ax[i, 1].set_xlabel("mass [GeV]")

    # Add legend
    for i in range(len(quantileslist)):
        ax[i, 0].legend()
        ax[i, 1].legend()

    log.info(
        f"Saving quantile mass distributions of {tagger.name} to {output_dir}/qmass_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png"
    )
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    plt.savefig(
        f"{output_dir}/qmass_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png"
    )

    return


def plot_mass_morphing(
    output_dir: Path | str,
    tagger: dict,
    files: list,
    trained_on_dataset_type: str,
    evaluated_on_dataset_type: str,
    dset: str,
    quantileslist: list,
    decor: bool = False,
):
    log.info("Plotting quantile mass morphing")
    x_bins = np.linspace(0, 250, 100)

    # Load the scores from the tagger
    tagger.score_name = "output"

    def load_score_and_mass(tagger, dataset_type, file_name, dset, decor=False):
        if decor:
            scores = DatasetManager().load_decor_scores(
                tagger, dataset_type, file_name, dset=dset
            )
        else:
            scores = DatasetManager().load_scores(
                tagger, dataset_type, file_name, dset=dset
            )

        mass = DatasetManager().load_mass(dataset_type, file_name, dset=dset)

        return scores, mass

    scores, mass = load_score_and_mass(
        tagger, evaluated_on_dataset_type, files[0], dset, decor
    )

    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    mass = mass[sorted_indices]

    cmap = cm.get_cmap("winter")

    doNorm = False if decor else True

    for j in range(len(quantileslist)):
        q_mass = mass[int(len(mass) * quantileslist[j]) :]
        q_mass_hist, _ = np.histogram(q_mass, bins=x_bins)
        if doNorm:
            q_mass_hist = q_mass_hist / np.sum(q_mass_hist)

        # Get a color from the colormap
        color = cmap(j / len(quantileslist))

        # Plot the score distributions
        linestyle = "solid" if j == 0 else "dashed"
        plt.stairs(
            q_mass_hist,
            x_bins,
            label=f"QCD - Quantile: {quantileslist[j]}",
            linestyle=linestyle,
            color=color,
        )

    # Plot Higgs dist
    if decor == False:
        signal_mass = DatasetManager().load_mass(
            evaluated_on_dataset_type, files[1], dset=dset
        )

        signal_mass_hist, _ = np.histogram(signal_mass, bins=x_bins)
        if doNorm:
            signal_mass_hist = signal_mass_hist / np.sum(signal_mass_hist)

        plt.stairs(
            signal_mass_hist,
            x_bins,
            label=f"Hbb - Quantile: 0",
            linestyle="solid",
            color="green",
        )
    # Plot kwargs
    plt.xlabel("mass [GeV]")
    y_label = "Normalized counts" if doNorm else "Counts"
    plt.ylabel(y_label)
    plt.legend(loc="upper right" if decor else None)

    log.info("Saving mass morphing")

    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")
    decorSuffix = "_decor" if decor else ""
    plt.savefig(
        f"{output_dir}/qmass_morphing_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}{decorSuffix}.pdf"
    )
    plt.close()

    return


# QUANTILES ALL CLASSES SEPARATELY
# def plot_mass_distribution_quantiles(
#     output_dir: Path | str,
#     taggers: list,
#     files: list,
#     dataset_type: str,
#     dset: str,
#     quantileslist: list,
# ) -> None:

#     log.info("Plotting quantile mass distributions")

#     x_bins = np.linspace(0, 250, 100)

#     for tag in taggers:
#         try:
#             fig, ax = plt.subplots(len(quantileslist), 2 , figsize=(10, 5*len(quantileslist)))

#             # Get the list of files
#             for j in range(len(files)):
#                 # Load the scores from the tagger
#                 tag.score_name = "output"
#                 scores = DatasetManager().load_scores(tag, dataset_type, files[j], dset=dset)
#                 # Load the masses from the tagger
#                 mass = DatasetManager().load_mass(dataset_type, files[j], dset=dset)

#                 sorted_indices = np.argsort(scores)
#                 scores = scores[sorted_indices]
#                 mass = mass[sorted_indices]

#                 for i in range(len(quantileslist)):
#                     ax[i,0].set_title(f"mass distribution for quantile {quantileslist[i]}")
#                     ax[i,1].set_title(f"decor mass distribution for quantile {quantileslist[i]}")

#                     q_mass = mass[int(len(mass)*quantileslist[i]):]
#                     q_mass_hist, _ = np.histogram(q_mass, bins=x_bins)

#                     # Plot the score distributions
#                     ax[i,0].stairs(q_mass_hist, x_bins, label=files[j])
#                     ax[i,0].set_xlabel("mass [GeV]")


#             # Get the decor masses from the tagger
#             # First, load the decor scores from the tagger
#             tag.score_name = "ext_decor_output"
#             decor_scores = []
#             decor_mass = []
#             new_labels = []
#             for j in range(len(files)):
#                 decor_scores.append(DatasetManager().load_scores(tag, dataset_type, files[j], dset=dset))
#                 decor_mass.append(DatasetManager().load_mass(dataset_type, files[j], dset=dset))
#                 new_labels.append(np.ones(len(decor_scores[j]))*j)

#             new_labels = np.concatenate(new_labels)
#             decor_scores = np.concatenate(decor_scores)
#             decor_mass = np.concatenate(decor_mass)

#             sorted_indices = np.argsort(decor_scores)
#             decor_scores = decor_scores[sorted_indices]
#             decor_mass = decor_mass[sorted_indices]
#             #New labels stays the same

#             for i in range(len(quantileslist)):
#                 for j in range(len(files)):
#                     decor_mass_j = decor_mass[new_labels==j]
#                     q_decor_mass_j = decor_mass_j[int(len(decor_mass_j)*quantileslist[i]):]
#                     q_decor_mass_hist_j, _ = np.histogram(q_decor_mass_j, bins=x_bins)

#                     # Plot the score distributions
#                     ax[i,1].stairs(q_decor_mass_hist_j, x_bins, label=files[j])
#                     ax[i,1].set_xlabel("mass [GeV]")


#         except Exception as e:
#             print(f"Failed for {tag.label}")
#             print(e)
#             print()

#         # Add legend
#         for i in range(len(taggers)):
#             ax[i,0].legend()
#             ax[i,1].legend()

#         log.info(f"Saving quantile mass distributions of {tag.label} to {output_dir}/qmass_{tag.label}_distribution.png")
#         plt.savefig(f'{output_dir}/qmass_{tag.label}_distribution.png')

#     return

# QUANTILES ON BACKGROUND AND SIGNAL
# def plot_mass_distribution_quantiles(
#     output_dir: Path | str,
#     taggers: list,
#     files: list,
#     dataset_type: str,
#     dset: str,
#     quantileslist: list,
# ) -> None:

#     log.info("Plotting quantile mass distributions")

#     x_bins = np.linspace(0, 250, 100)

#     for tag in taggers:
#         if True:
#             fig, ax = plt.subplots(len(quantileslist), 2 , figsize=(10, 5*len(quantileslist)))

#             mass = []
#             scores = []
#             true_label = []
#             # Get the list of files
#             for j in range(len(files)):
#                 # Load the scores from the tagger
#                 tag.score_name = "output"
#                 scores.append(DatasetManager().load_scores(tag, dataset_type, files[j], dset=dset))
#                 # Load the masses from the tagger
#                 mass.append(DatasetManager().load_mass(dataset_type, files[j], dset=dset))

#                 # set true label
#                 true_label.append(np.ones(len(scores[j]))*j)

#             true_label = np.concatenate(true_label)
#             scores = np.concatenate(scores)
#             mass = np.concatenate(mass)

#             sorted_indices = np.argsort(scores)
#             true_label = true_label[sorted_indices]
#             scores = scores[sorted_indices]
#             mass = mass[sorted_indices]

#             for i in range(len(quantileslist)):
#                 ax[i,0].set_title(f"mass distribution for quantile {quantileslist[i]}")
#                 ax[i,1].set_title(f"decor mass distribution for quantile {quantileslist[i]}")

#                 q_true_label = true_label[int(len(true_label)*quantileslist[i]):]
#                 q_mass = mass[int(len(mass)*quantileslist[i]):]

#                 for j in range(len(files)):
#                     q_mass_hist, _ = np.histogram(q_mass[q_true_label==j], bins=x_bins)

#                     # Plot the score distributions
#                     ax[i,0].stairs(q_mass_hist, x_bins, label=files[j])
#                     ax[i,0].set_xlabel("mass [GeV]")


#             # Get the decor masses from the tagger
#             # First, load the decor scores from the tagger
#             tag.score_name = "ext_decor_output"
#             decor_scores = []
#             decor_mass = []
#             true_label = []
#             file_len_id = [0]
#             for j in range(len(files)):
#                 decor_scores.append(DatasetManager().load_scores(tag, dataset_type, files[j], dset=dset))
#                 decor_mass.append(DatasetManager().load_mass(dataset_type, files[j], dset=dset))
#                 file_len_id.append(file_len_id[j]+len(decor_scores[j]))

#                 # set true label
#                 true_label.append(np.ones(len(decor_scores[j]))*j)

#             true_label = np.concatenate(true_label)
#             decor_scores = np.concatenate(decor_scores)
#             decor_mass = np.concatenate(decor_mass)

#             sorted_indices = np.argsort(scores)
#             true_label = true_label[sorted_indices]
#             decor_scores = decor_scores[sorted_indices]
#             decor_mass = decor_mass[sorted_indices]

#             for i in range(len(quantileslist)):
#                 q_true_label = true_label[int(len(true_label)*quantileslist[i]):]
#                 q_decor_mass = decor_mass[int(len(decor_mass)*quantileslist[i]):]

#                 for j in range(len(files)):
#                     q_decor_mass_hist, _ = np.histogram(q_decor_mass[q_true_label==j], bins=x_bins)

#                     # Plot the score distributions
#                     ax[i,1].stairs(q_decor_mass_hist, x_bins, label=files[j])
#                     ax[i,1].set_xlabel("mass [GeV]")


#         # except Exception as e:
#         #     print(f"Failed for {tag.label}")
#         #     print(e)
#         #     print()

#         # Add legend
#         for i in range(len(taggers)):
#             ax[i,0].legend()
#             ax[i,1].legend()

#         log.info(f"Saving quantile mass distributions of {tag.label} to {output_dir}/qmass_{tag.label}_distribution.png")
#         plt.savefig(f'{output_dir}/qmass_{tag.label}_distribution.png')

#     return


def plot_OT(
    output_dir: Path | str,
    scores: np.ndarray,
    decor_scores: np.ndarray,
    OT_decor_scores: np.ndarray,
    tag: list,
    dataset_type: str,
) -> None:
    import os

    log.info("Plotting OT distribution comparison")

    x_bins = np.linspace(-10, 10, 100)

    scores_hist, _ = np.histogram(scores, bins=x_bins)  # , density=True)
    decor_scores_hist, _ = np.histogram(decor_scores, bins=x_bins)  # , density=True)
    OT_decor_scores_hist, _ = np.histogram(
        OT_decor_scores, bins=x_bins
    )  # , density=True)

    plt.stairs(scores_hist, x_bins, label="scores")
    plt.stairs(decor_scores_hist, x_bins, label="decor_scores")
    plt.stairs(OT_decor_scores_hist, x_bins, label="OT_decor_scores")
    plt.xlabel("Score")
    plt.legend()

    # if file folder does not exist, create it
    if not os.path.exists(f"{output_dir}/OT_score_dist_comparison"):
        os.makedirs(f"{output_dir}/OT_score_dist_comparison", exist_ok=True)

    plt.savefig(
        f"{output_dir}/OT_score_dist_comparison/{tag.name}_OT_score_distributions.png"
    )
    plt.close()


@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="plotting_FR.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.single_tagger:
        log.info("Plotting single tagger")
        if cfg.tagger_name is None:
            raise ValueError("You must specify a tagger name to plot a single tagger")
        if cfg.tagger_path is None:
            raise ValueError("You must specify a tagger path to plot a single tagger")

        cfg.taggers = [
            {"name": cfg.tagger_name, "path": cfg.tagger_path, "score_name": "output"}
        ][0]

    import yaml

    # Load dataset configuration
    with open(f"{cfg.config_path}/data_confs/evaluated_on.yaml", "r") as file:
        evaluated_on_data_config = yaml.safe_load(file)

    # Initialize the lists
    evaluated_on_dataset_types = []
    files_list_dict = {}

    # Populate the lists
    for dataset in evaluated_on_data_config:
        evaluated_on_dataset_types.append(dataset["dataset_type"])
        files_list_dict[dataset["dataset_type"]] = [
            dataset["datasets"]["c0"],
            dataset["datasets"]["c1"],
        ]

    if cfg.plot_score_distribution:
        plot_score_distributions(
            output_dir=cfg.output_dir,
            tagger=cfg.taggers,
            files_dict=files_list_dict,
            dataset_types=evaluated_on_dataset_types,
            trained_on_dataset_type=cfg.trained_on_dataset_type,
            dset=cfg.dset,
            quantileslist=cfg.quantileslist,
        )
    if cfg.plot_mass_distribution:
        plot_mass_distribution(cfg)
        plot_mass_distribution_comparison(cfg)

    if cfg.plot_mass_distribution_quantiles:
        for evaluated_on_dataset_type in evaluated_on_dataset_types:
            plot_mass_distribution_quantiles(
                output_dir=cfg.output_dir,
                tagger=cfg.taggers,
                files=files_list_dict[evaluated_on_dataset_type],
                trained_on_dataset_type=cfg.trained_on_dataset_type,
                evaluated_on_dataset_type=evaluated_on_dataset_type,
                dset=cfg.dset,
                quantileslist=cfg.quantileslist,
            )

    if cfg.plot_mass_morphing:
        for evaluated_on_dataset_type in evaluated_on_dataset_types:
            plot_mass_morphing(
                output_dir=cfg.output_dir,
                tagger=cfg.taggers,
                files=files_list_dict[evaluated_on_dataset_type],
                trained_on_dataset_type=cfg.trained_on_dataset_type,
                evaluated_on_dataset_type=evaluated_on_dataset_type,
                dset=cfg.dset,
                quantileslist=cfg.quantileslist,
                decor=False,
            )
            plot_mass_morphing(
                output_dir=cfg.output_dir,
                tagger=cfg.taggers,
                files=files_list_dict[evaluated_on_dataset_type],
                trained_on_dataset_type=cfg.trained_on_dataset_type,
                evaluated_on_dataset_type=evaluated_on_dataset_type,
                dset=cfg.dset,
                quantileslist=cfg.quantileslist,
                decor=True,
            )

    return


if __name__ == "__main__":
    main()
