"""Example plotting script."""

import pickle
from typing import Mapping

import pyrootutils
from tqdm import tqdm

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from scipy.special import ndtr
from scipy.stats import gaussian_kde

from src.utils import iteratively_build_bins
from src.datasets_utils import DatasetManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def empirical_cdf(a: np.ndarray) -> tuple:
    x, counts = np.unique(a, return_counts=True)
    cumsum = np.cumsum(counts).astype("float32")
    cumsum /= cumsum[-1]
    return x, cumsum


def fit_binned_splines(
    decor_vals: np.ndarray, decor_bins: np.ndarray, scores: np.ndarray, out_path: str
) -> tuple:
    """Fit binned splines to data.

    This function takes in an array of values to be decorrelated,
    an array of bin edges, an array of scores, and an output path.
    It then bins the values to be decorrelated and fits a cubic spline to
    the empirical cumulative distribution function (CDF)
    and a kernel density estimate (KDE) of the data in each bin.
    The fitted splines are saved as pickled objects at the specified output path.

    Parameters
    ----------
    decor_vals : np.ndarray
        Array of values to be decorrelated.
    decor_bins : np.ndarray
        Array of bin edges.
    scores : np.ndarray
        Array of scores.
    out_path : str
        Path to save the fitted splines.

    Returns
    -------
    tuple
        A tuple containing two lists of fitted splines.
        The first list contains the splines fitted to the empirical CDFs,
        and the second list contains the splines fitted to the KDEs.
    """

    # Empty arrays to save the splines
    exact_cdfs = []
    kde_cdfs = []

    # Bin the variable to be decorrelated
    decor_idx = np.digitize(decor_vals, decor_bins)

    # Manually loop over the bins, fitting the data inside with a interp1d fn
    for idx in tqdm(range(1, len(decor_bins) + 1), "Fitting splines"):
        scores_in_bin = scores[decor_idx == idx]

        # Fit a KDE on the data
        kde = gaussian_kde(
            scores_in_bin,
        )

        # Get the empirical cdf of the data
        x, cdf = empirical_cdf(scores_in_bin)

        # Get the kde
        x_space = np.linspace(scores_in_bin.min(), scores_in_bin.max(), 100)
        kde_cdf = [
            ndtr(np.ravel(item - kde.dataset) / kde.factor).mean() for item in x_space
        ]

        # Define the functions
        exact_fn = interp1d(
            x, cdf, kind="linear", fill_value=(0, 1), bounds_error=False
        )
        kde_fn = interp1d(
            x_space, kde_cdf, kind="linear", fill_value=(0, 1), bounds_error=False
        )

        # Add to the total
        exact_cdfs.append(exact_fn)
        kde_cdfs.append(kde_fn)

    # Save the splines as pickled objects
    out_file = Path(out_path, "binned_splines.obj")
    with open(out_file, "wb") as f:
        pickle.dump(
            {
                "exact_cdfs": exact_cdfs,
                "kde_cdfs": kde_cdfs,
                "decor_bins": decor_bins,
            },
            f,
        )

    return exact_cdfs, kde_cdfs


def apply_binned_splines(
    decor_vals: np.ndarray,
    scores: np.ndarray,
    decor_bins: np.ndarray,
    binned_fns: np.ndarray,
) -> np.ndarray:
    bin_idx = np.digitize(decor_vals, decor_bins)

    # Start with zeros for the new scores
    decor_scores = np.zeros_like(scores)

    # Manually iterate through each bin applying the appropriate function
    for idx in tqdm(range(1, len(decor_bins) + 1), "decorrelating"):
        bin_mask = bin_idx == idx
        bin_scores = scores[bin_mask]
        decor_scores[bin_mask] = binned_fns[idx - 1](bin_scores)

    return decor_scores


def decorrelate_and_save(
    mass_bins: np.ndarray,
    tag: Mapping,
    exact_fns: np.ndarray,
    kde_fns: np.ndarray,
    file_name: str,
    dataset_type: str,
    dset: str = "test",
) -> None:
    """Decorrelates the scores and save the score output file.

    This function loads the scores from the tagger and the masses from the datafile.
    It then applies binned splines to the scores and masses using the given mass bins
    and exact and kde functions.
    The resulting scores are then saved as new columns in the score output file,
    overwriting any previous entries.

    Parameters
    ----------
    data_dir : str
        The directory containing the data file.
    mass_bins : np.ndarray
        The mass bins to use when applying binned splines.
    tag : Mapping
        A mapping containing information about the tagger,
        including its path, name, and score_name.
    exact_fns : np.ndarray
        The exact functions to use when applying binned splines.
    kde_fns : np.ndarray
        The kde functions to use when applying binned splines.
    file_name : str
        The name of the input (and output) HDF file

    Returns
    -------
    None
    """

    # # Load the scores from the tagger
    # if dataset_type == "JetNet":

    #     from franckstools.franckstools.utils import optimal_transport_1D
    #     scores = DatasetManager().load_scores(tag, dataset_type, file_name, dset=dset)

    #     # We need to optimal transport the scores to the decor distribution

    #     # First load the decor scores
    #     decor_scores = DatasetManager().load_scores(tag, decor_type, decor_file, dset=decor_dset)

    #     # Now optimal transport the scores
    #     scores = optimal_transport_1D(scores, decor_scores, np.linspace(-50,50,10000))

    # elif dataset_type == "Rodem":
    #     scores = DatasetManager().load_scores(tag, dataset_type, file_name, dset=dset)

    # Load the scores from the tagger
    scores = DatasetManager().load_scores(tag, dataset_type, file_name, dset=dset)

    # Load the masses from the datafile
    mass = DatasetManager().load_mass(dataset_type, file_name, dset=dset)

    # Get the bin indexes of the new file
    ext_scores = apply_binned_splines(mass, scores, mass_bins, exact_fns)
    kde_scores = apply_binned_splines(mass, scores, mass_bins, kde_fns)

    # Save these scores as new columns in the score output file

    # Matthews solution is to append a new column to the file I want to create a new file
    # with h5py.File(Path(tag.path,tag.name,"outputs",DatasetManager().get_output_file_name(dataset_type,file_name,dset)), "a") as outfile:
    #     # Delete the previous entries
    #     try:
    #         del outfile[f"kde_decor_{tag.score_name}"]
    #     except KeyError:
    #         pass
    #     try:
    #         del outfile[f"ext_decor_{tag.score_name}"]
    #     except KeyError:
    #         pass

    #     # Save the new datasets
    #     outfile.create_dataset(f"ext_decor_{tag.score_name}", data=ext_scores)
    #     outfile.create_dataset(f"kde_decor_{tag.score_name}", data=kde_scores)

    # New solution is to create a new file

    log.info(tag.path)
    log.info(tag.name)
    log.info(
        DatasetManager()
        .get_output_file_name(dataset_type, file_name, dset)
        .replace(".h5", "_decor.h5")
    )
    with h5py.File(
        Path(
            tag.path,
            tag.name,
            "outputs",
            f"{DatasetManager().get_output_file_name(dataset_type,file_name,dset).replace('.h5', '')}_decor.h5",
        ),
        "w",
    ) as outfile:
        # Save the new datasets
        outfile.create_dataset(
            f"{tag.score_name}", data=ext_scores
        )  # WARNING: I have removed the ext_decor_ prefix for easier access
        outfile.create_dataset(f"kde_decor_{tag.score_name}", data=kde_scores)


@hydra.main(
    version_base=None,
    config_path=str(root / "configs"),
    config_name="binned_decorrelation.yaml",
)
def main(cfg: DictConfig) -> None:
    """Main script."""

    # If snakemake was used, we need to convert the additional parameters to overwritte decor_files
    if hasattr(cfg, "sm_decor_files_c0"):
        log.info("Overwritting decor_files with snakemake parameters")
        cfg.decor_files = [cfg.sm_decor_files_c0, cfg.sm_decor_files_c1]
        cfg.files = [cfg.sm_files_c0, cfg.sm_files_c1]

    if cfg.decorrelate_single_tagger:
        log.info("Decorrelate single tagger set to true")
        if cfg.tagger_name is None:
            raise ValueError("No tagger name provided")
        if cfg.tagger_path is None:
            raise ValueError("No tagger path provided")

        cfg.taggers = [
            {
                "name": cfg.tagger_name,
                "path": cfg.tagger_path,
                "score_name": cfg.tagger_score_name,
            }
        ]

    # Load the mass from the decor file
    log.info("Loading mass information for CDF calculations")

    decor_mass = np.concatenate(
        [
            DatasetManager().load_mass(cfg.decor_type, file, dset=cfg.decor_dset)
            for file in cfg.decor_files
        ]
    )
    true_label = np.concatenate(
        [
            np.ones(
                len(
                    DatasetManager().load_mass(
                        cfg.decor_type, file, dset=cfg.decor_dset
                    )
                )
            )
            * i
            for i, file in enumerate(cfg.decor_files)
        ]
    )

    # Iteratively build the mass bins to use
    mass_bins = iteratively_build_bins(
        decor_mass,
        min_bw=cfg.min_bw,
        min_per_bin=cfg.min_per_bin,
        max_value=cfg.max_value,
    )

    if cfg.dataset_type != cfg.decor_type:
        log.info(
            f"Detected different dataset_type: {cfg.dataset_type} than for the decor set {cfg.decor_type}"
        )
        log.info(
            f"Will proceed with optimaly transporting the decor dataset to the {cfg.dataset_type} distribution"
        )

    # Cycle through the taggers
    for tag in cfg.taggers:
        log.info(f"Using tagger {tag.name}")

        # Load the scores of the taggers (pass through sigmoid to bound it)
        decor_scores = np.concatenate(
            [
                DatasetManager().load_scores(
                    tag, cfg.decor_type, file, dset=cfg.decor_dset
                )
                for file in cfg.decor_files
            ]
        )

        if cfg.dataset_type != cfg.decor_type:
            # We need to optimal transport the decor scores to the other (e.g JetNet) distribution
            from franckstools.franckstools.utils import optimal_transport_1D

            log.info(
                f"Optimal transporting decor dataset to {cfg.dataset_type} distribution"
            )
            # First load the tagger scores
            tagger_scores = np.concatenate(
                [
                    DatasetManager().load_scores(
                        tag, cfg.dataset_type, file, dset=cfg.dset
                    )
                    for file in cfg.files
                ]
            )
            # Now optimal transport the scores
            OT_decor_scores = optimal_transport_1D(
                decor_scores, tagger_scores, np.linspace(-50, 50, 10000)
            )

            if cfg.plot_OT == True:
                from scripts.plotting_FR import plot_OT

                plot_OT(
                    cfg.output_dir,
                    tagger_scores,
                    decor_scores,
                    OT_decor_scores,
                    tag,
                    cfg.dataset_type,
                )

        else:
            OT_decor_scores = decor_scores

        # Check that we are lined up and good to go
        assert len(OT_decor_scores) == len(decor_mass)

        # Filter to background for fitting mass decorrelation splines
        decor_background_mass = decor_mass[true_label == 0]
        decor_background_OT_scores = OT_decor_scores[true_label == 0]

        # Fit and save the splines
        outpath = Path(tag.path, tag.name)
        exact_fns, kde_fns = fit_binned_splines(
            decor_background_mass, mass_bins, decor_background_OT_scores, outpath
        )

        # Cycle through the output files and fit new splines
        for file in cfg.files:
            log.info(f"- applying to {file}")
            try:
                decorrelate_and_save(
                    mass_bins,
                    tag,
                    exact_fns,
                    kde_fns,
                    file,
                    cfg.decor_type,
                    dset=cfg.dset,
                )
            except Exception as e:
                log.info(f"Failed to decorrelate {file}")
                log.info(e)


if __name__ == "__main__":
    main()
