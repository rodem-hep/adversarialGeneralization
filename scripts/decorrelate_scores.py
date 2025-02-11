"""Example plotting script."""

import logging
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def empirical_cdf(a):
    x, counts = np.unique(a, return_counts=True)
    cumsum = np.cumsum(counts).astype("float32")
    cumsum /= cumsum[-1]
    return x, cumsum


def get_binned_cdfs(
    data_file: Path, score_file: Path, score_name: str, mass_bins: np.ndarray
) -> list:
    """Returns a list of fitted cdf functions for each mass bin for
    decorrelation."""

    logging.info("Loading information for CDF calculations")
    with h5py.File(data_file, "r") as f:
        mass = f["objects/jets/jet1_obs"][:, 3]

    with h5py.File(score_file, "r") as f:
        scores = sigmoid(f[score_name][:].flatten())
    bin_idx = np.digitize(mass, mass_bins)

    # Cycle through each of the mass bins and fit a cdf function
    logging.info("Fitting CDF functions for each mass bin")
    cdf_functions = []
    for idx in range(1, len(mass_bins) + 1):
        scores_in_bin = scores[bin_idx == idx]

        # Get the empirical cdf of the data, make sure it starts at (0,0)
        x, cdf = empirical_cdf(scores_in_bin)
        cdf = np.insert(cdf, 0, 0)
        x = np.insert(x, 0, 0)
        cdf_functions.append(
            interp1d(x, cdf, kind="linear", fill_value=(0, 1), bounds_error=False)
        )

    return cdf_functions


def apply_cdfs_decor(
    data_file: Path,
    score_file: Path,
    score_name: str,
    mass_bins: np.ndarray,
    cdfs: list,
):
    """Returns decorrelated scores by using cdfs for the data."""

    logging.info("Loading information to decorrelate the scores")
    with h5py.File(data_file, "r") as f:
        mass = f["objects/jets/jet1_obs"][:, 3]

    with h5py.File(score_file, "r") as f:
        scores = sigmoid(f[score_name][:].flatten())
    bin_idx = np.digitize(mass, mass_bins)

    # Cycle through each of the mass bins and fit a cdf function
    logging.info("Applying decorrelations per mass bin")
    decor_scores = np.zeros_like(scores)
    for idx in range(1, len(mass_bins) + 1):
        decor_scores[bin_idx == idx] = cdfs[idx - 1](scores[bin_idx == idx])

    return decor_scores


def add_decor_score(
    data_dir: Path,
    training_file: str,
    procs: list,
    networks: list,
    mass_bins: np.ndarray,
):
    """Add the decorrelated scores as a new column in the HDF files."""
    for net, score_name in networks:
        cdfs = get_binned_cdfs(
            data_dir / training_file, net / training_file, score_name, mass_bins
        )
        for proc in procs:
            decor_scores = apply_cdfs_decor(
                data_dir / proc, net / proc, score_name, mass_bins, cdfs
            )

            # Save these scores into the same output hdf file
            file = h5py.File(net / proc, "a")
            try:
                del file[f"decor_{score_name}"]
            except KeyError:
                pass
            file.create_dataset(f"decor_{score_name}", data=decor_scores)
            file.close()


def main():
    """Main script."""
    data_dir = Path("/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data/")
    tagger_dir = Path("/srv/beegfs/scratch/groups/rodem/anomalous_jets/taggers/")
    taggers = [
        (tagger_dir / "supervised" / "dense_test", "output"),
        (tagger_dir / "supervised" / "transformer_test", "output"),
    ]
    training_file = "QCD_jj_pt_450_1200_decor_test.h5"
    datafiles = [
        "QCD_jj_pt_450_1200_test.h5",
        "WZ_allhad_pt_450_1200_test.h5",
        "ttbar_allhad_pt_450_1200_test.h5",
        "H2tbtb_1700_HC_250_test.h5",
    ]
    mass_bins = np.linspace(0, 400, 200)
    logging.basicConfig(level=logging.INFO)

    add_decor_score(data_dir, training_file, datafiles, taggers, mass_bins)


if __name__ == "__main__":
    main()
