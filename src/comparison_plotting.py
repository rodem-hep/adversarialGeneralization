"""A collection of plotting functions useful for jet utilities."""

from copy import deepcopy
from functools import partial
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

# import pyBumpHunter as BH
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import poisson
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.datasets_utils import DatasetManager

# Some defaults for my plots to make them look nicer
plt.rcParams["xaxis.labellocation"] = "right"
plt.rcParams["yaxis.labellocation"] = "top"
plt.rcParams["legend.edgecolor"] = "1"
plt.rcParams["legend.loc"] = "upper left"
plt.rcParams["legend.framealpha"] = 0.0
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["legend.fontsize"] = 11


def kl_div(hist_1: np.ndarray, hist_2: np.ndarray):
    """Calculate the KL divergence between two binned densities."""
    assert hist_1.shape == hist_2.shape
    h_1 = hist_1 / np.sum(hist_1)
    h_2 = hist_2 / np.sum(hist_2)
    h_1 = np.clip(h_1, a_min=1e-8, a_max=None)  # To make the logarithm safe!
    h_2 = np.clip(h_2, a_min=1e-8, a_max=None)
    return np.sum(h_1 * np.log(h_1 / h_2))


def js_div(hist_1: np.ndarray, hist_2: np.ndarray) -> float:
    """Calculate the Jensen-Shannon Divergence between two binned densities."""
    assert hist_1.shape == hist_2.shape
    M = 0.5 * (hist_1 + hist_2)
    return 0.5 * (kl_div(hist_1, M) + kl_div(hist_2, M))


def get_hunter_scheme(hunter) -> np.ndarray:
    if hunter.str_scale == "lin":
        sig_str = np.arange(
            hunter.str_min,
            hunter.str_min + hunter.str_step * len(hunter.sigma_ar),
            step=hunter.str_step,
        )
    else:
        sig_str = np.array(
            [
                i % 10 * 10 ** (hunter.str_min + i // 10)
                for i in range(len(hunter.sigma_ar) + len(hunter.sigma_ar) // 10 + 1)
                if i % 10 != 0
            ]
        )
    return sig_str


# def run_signal_injection(
#     output_dir: str,
#     data_dir: str,
#     taggers: List[dict],
#     background_name: str,
#     background_file: str,
#     signal_name: str,
#     signal_file: str,
#     reject_frac: float = 0.95,
#     bins=[0, 500],
# ) -> None:
#     taggers = deepcopy(taggers)  # To be safe with pops

#     # Skip this step if the background matches the signal
#     if background_file == signal_file:
#         return None

#     # Load the background and signal masses
#     with h5py.File(Path(data_dir, background_file), mode="r") as h:
#         background_masses = h["objects/jets/jet1_obs"][:, 3]
#     with h5py.File(Path(data_dir, signal_file), mode="r") as h:
#         signal_masses = h["objects/jets/jet1_obs"][:, 3]

#     # Create a BumpHunter1D class instance
#     hunter = BH.BumpHunter1D(
#         rang=bins,
#         width_min=2,
#         width_max=10,
#         width_step=2,
#         scan_step=1,
#         npe=100000,
#         nworker=1,
#         seed=42,
#         use_sideband=True,
#         str_min=-5,
#         sigma_limit=8,
#         str_scale="log",
#     )

#     # Run the bump hunter without any cuts
#     hunter.signal_inject(signal_masses, background_masses)
#     raw_str = get_hunter_scheme(hunter)
#     raw_sens = hunter.sigma_ar.copy()
#     min_l = min(len(raw_sens), len(raw_str))  # Sometimes the hunter lengths dont match
#     raw_sens = raw_sens[:min_l]
#     raw_str = raw_str[:min_l]

#     # Create the figure
#     fig = plt.figure(figsize=(5, 5))
#     plt.errorbar(
#         raw_str,
#         raw_sens[:, 0],
#         xerr=0,
#         yerr=[raw_sens[:, 1], raw_sens[:, 2]],
#         marker="x",
#         color="k",
#         uplims=raw_sens[:, 2] == 0,
#         label="raw data",
#     )

#     # Cycle through the taggers
#     for tagger in taggers:
#         # Pop off the required variables
#         path = tagger.pop("path")
#         score_name = tagger.pop("score_name")

#         # Load the tagger background and signal scores
#         with h5py.File(Path(path, background_file), mode="r") as h:
#             background_scores = h[score_name][:]
#         with h5py.File(Path(path, signal_file), mode="r") as h:
#             signal_scores = h[score_name][:]

#         # Check that they are the same legth as the mass data
#         assert len(signal_masses) == len(signal_scores)
#         assert len(background_masses) == len(background_scores)

#         # Apply the threshold and run the signal injection test again
#         threshold = np.quantile(background_scores, reject_frac)
#         passing_background_masses = background_masses[background_scores > threshold]
#         passing_signal_masses = signal_masses[signal_scores > threshold]
#         hunter.signal_inject(passing_signal_masses, passing_background_masses)
#         cut_str = get_hunter_scheme(hunter)
#         cut_sens = hunter.sigma_ar.copy()
#         min_l = min(len(cut_sens), len(cut_str))  # hunter lengths dont match
#         cut_sens = cut_sens[:min_l]
#         cut_str = cut_str[:min_l]

#         # Add the search to the plots
#         plt.errorbar(
#             cut_str,
#             cut_sens[:, 0],
#             xerr=0,
#             yerr=[cut_sens[:, 1], cut_sens[:, 2]],
#             uplims=cut_sens[:, 2] == 0,
#             **tagger,
#         )

#     # Fix up the plot and save
#     plt.legend()
#     plt.title(
#         f"Signal sensitivity for {signal_name} vs {background_name} with {reject_frac} BR cut"
#     )
#     plt.xlabel("signal strengths")
#     plt.ylabel("measured significance")
#     plt.xscale("log")
#     plt.tight_layout()
#     outpath = Path(output_dir) / (f"si_{background_name}_{signal_name}").replace(
#         " ", "_"
#     )
#     fig.savefig(outpath)
#     plt.close(fig)


# def run_pybumphunter(
#     output_dir: str,
#     data_dir: str,
#     tagger: dict,
#     background_name: str,
#     background_file: str,
#     signal_name: str,
#     signal_file: str,
#     snb_ratio: float = 2e-3,
#     reject_frac: float = 0.95,
#     bins=[0, 500],
# ) -> None:
#     """Runs BumpHunter1D to search for bumps in the mass distributions of jet
#     events, and saves a bump plot to a file.

#     Parameters
#     ----------
#     output_dir : str
#         Path to the directory where the output files will be saved.
#     data_dir : str
#         Path to the directory where the input data files are stored.
#     tagger : dict
#         A dictionary with the following keys: "path" (path to the directory where
#         the tagger scores file is stored) and "score_name" (name of the tagger score
#         variable in the HDF5 file).
#     background_name : str
#         A string identifier for the background data file.
#     background_file : str
#         A string filename for the background data file.
#     signal_name : str
#         A string identifier for the signal data file.
#     signal_file : str
#         A string filename for the signal data file.
#     snb_ratio : float, optional
#         The signal-to-noise ratio to use when selecting signal events. Default is 2e-3.
#     reject_frac : float, optional
#         The fraction of background events to reject when calculating the score threshold.
#         Default is 0.95.
#     bins : list, optional
#         A list of two integers specifying the range of the mass distribution to consider.
#         Default is [0, 500].

#     Returns
#     -------
#     None

#     Raises
#     ------
#     AssertionError
#         If the length of the background mass and score arrays is not the same.

#     Notes
#     -----
#     This function loads the background and signal data files and corresponding tagger
#     score files from the specified directories, and combines them into a single array
#     of mass and score values. It then uses BumpHunter1D to search for bumps in the mass
#     distribution of events passing a specified score threshold, and saves a bump plot
#     to a file.
#     """

#     # Load the background masses
#     with h5py.File(Path(data_dir, background_file), mode="r") as h:
#         background_masses = h["objects/jets/jet1_obs"][:, 3]

#     # Load the background tagger scores
#     with h5py.File(Path(tagger["path"], background_file), mode="r") as h:
#         background_scores = np.squeeze(h[tagger["score_name"]][:])

#     # Check that they are the same length
#     assert len(background_masses) == len(background_scores)

#     # Load the appropriate number of signal samples
#     n_signal = int(snb_ratio * len(background_masses))
#     with h5py.File(Path(data_dir, signal_file), mode="r") as h:
#         signal_masses = h["objects/jets/jet1_obs"][:n_signal, 3]
#     with h5py.File(Path(tagger["path"], signal_file), mode="r") as h:
#         signal_scores = np.squeeze(h[tagger["score_name"]][:n_signal])

#     # Mix the two samples together
#     combined_masses = np.hstack([background_masses, signal_masses])
#     combined_scores = np.hstack([background_scores, signal_scores])

#     # Calculate the score threshold for the specified rejection value
#     threshold = np.quantile(combined_scores, reject_frac)

#     # Get the distributions of masses passing the threshold
#     passing_masses = combined_masses[combined_scores > threshold]

#     # Create a BumpHunter1D class instance
#     hunter = BH.BumpHunter1D(
#         rang=bins,
#         width_min=2,
#         width_max=10,
#         width_step=2,
#         scan_step=1,
#         npe=10000,
#         nworker=1,
#         seed=42,
#         use_sideband=True,
#         # weights=np.full_like(combined_masses, 1-reject_frac)
#     )

#     # Call the bump_scan method
#     hunter.bump_scan(passing_masses, combined_masses)

#     # Get and save bump plot
#     outpath = Path(output_dir) / (
#         f"bh_{background_name}_{signal_name}_{tagger['label']}({tagger['score_name']})"
#     ).replace(" ", "_")
#     hunter.plot_bump(
#         passing_masses, combined_masses, filename=outpath.with_suffix(".png")
#     )


def dummy_bump_hunt(
    tagger: dict,
    path: str,
    data_dir: str,
    background: str,
    signal: str,
    reject_frac: float = 0.95,
    snb_ratio: float = 1e-3,
    fig_size: tuple = (10, 5),
    n_bootstraps: int = 50,
    bins: np.ndarray | partial = np.linspace(0, 500, 50),
) -> None:
    print(
        f"Performing a dummy bump hunt on {background} and {signal} using {tagger.name}"
    )

    # Load the background masses and tagger scores
    data_path = Path(data_dir, background + "_test.h5")
    with h5py.File(data_path, mode="r") as h:
        masses = h["hlvs"][:, 3]

    tag_path = Path(tagger.path, tagger.name, "outputs", background + "_test.h5")
    with h5py.File(tag_path, mode="r") as h:
        scores = h[tagger.score_name][:].flatten()

    # Check that they are the same length
    assert len(masses) == len(scores)

    # Split them into background and control masses (for the bootstrapping)
    n_back = len(masses) // 2
    cntrl_masses = masses[:n_back]
    cntrl_scores = scores[:n_back]
    back_masses = masses[-n_back:]
    back_scores = scores[-n_back:]

    # Calculate how many signal processes to load
    n_signal = int(snb_ratio * n_back)

    # Load the signal masses and tagger scores
    data_path = Path(data_dir, signal + "_test.h5")
    with h5py.File(data_path, mode="r") as h:
        signal_masses = h["hlvs"][:n_signal, 3]

    tag_path = Path(tagger.path, tagger.name, "outputs", signal + "_test.h5")
    with h5py.File(tag_path, mode="r") as h:
        signal_scores = h[tagger.score_name][:n_signal].flatten()

    # Replace some of the samples with the signal
    comb_masses = np.hstack([signal_masses, back_masses])
    comb_scores = np.hstack([signal_scores, back_scores])

    # Build the bins before the cut
    pre_bins = bins(cntrl_masses) if isinstance(bins, partial) else bins

    # Get the histograms of these values
    comb_masses = np.clip(comb_masses, pre_bins[0], pre_bins[-1])
    cntrl_masses = np.clip(cntrl_masses, pre_bins[0], pre_bins[-1])
    comb_hist, _ = np.histogram(comb_masses, pre_bins)
    cntrl_hist, _ = np.histogram(cntrl_masses, pre_bins)
    comb_err = np.sqrt(comb_hist)
    cntrl_err = np.sqrt(cntrl_hist)

    # Determine the p-value before applying the cut
    orig_pvals = []
    orig_pval_down = []
    for m, m_e, c, e in zip(cntrl_hist, cntrl_err, comb_hist, comb_err):
        orig_pvals.append(1 - poisson.cdf(c, m + m_e))
        orig_pval_down.append(1 - poisson.cdf(c - e, m + m_e))
    orig_pvals = np.array(orig_pvals)
    orig_pval_down = np.array(orig_pval_down)

    # Define the threshold on the control region
    threshold = np.quantile(cntrl_scores, reject_frac)

    # Apply the make a histogram for those passing the test
    passing_masses = comb_masses[comb_scores > threshold]
    post_bins = bins(passing_masses) if isinstance(bins, partial) else bins
    passing_masses = (np.clip(passing_masses, post_bins[0], post_bins[-1]),)
    pass_hist, _ = np.histogram(passing_masses, post_bins)
    pass_err = np.sqrt(pass_hist)

    # We need to calculate the test statistic distribution so we bootstrap
    bin_counts = []
    for i in tqdm(range(n_bootstraps), "boostrapping", leave=False):
        idxes = np.random.choice(n_back, size=n_back, replace=True)
        boot_masses = cntrl_masses[idxes]
        boot_scores = cntrl_scores[idxes]
        passing_masses = boot_masses[boot_scores > threshold]
        passing_masses = np.clip(passing_masses, post_bins[0], post_bins[-1])
        hist, _ = np.histogram(passing_masses, post_bins)
        bin_counts.append(hist)
    bin_counts = np.vstack(bin_counts)

    # Fit a poisson distribution to each bin which just requires the mean
    cntrl_pass = np.mean(bin_counts, axis=0)
    cntrl_pass = np.clip(cntrl_pass, 1, None)
    cntrl_pass_err = np.sqrt(cntrl_pass)

    # Calculate the p_value of the pass hist
    new_pvals = []
    new_pval_down = []
    for m, m_e, c, e in zip(cntrl_pass, cntrl_pass_err, pass_hist, pass_err):
        new_pvals.append(1 - poisson.cdf(c, m + m_e))
        new_pval_down.append(1 - poisson.cdf(c - e, m + m_e))
    new_pvals = np.array(new_pvals)
    new_pval_down = np.array(new_pval_down)

    # Get the histograms of the distributions for plotting
    sign_hist, _ = np.histogram(
        np.clip(signal_masses, pre_bins[0], pre_bins[-1]), pre_bins
    )

    # Need these for plotting
    pre_mid_bins = (pre_bins[1:] + pre_bins[:-1]) / 2
    pre_bin_widths = (pre_bins[1:] - pre_bins[:-1]) / 2
    post_mid_bins = (post_bins[1:] + post_bins[:-1]) / 2
    post_bin_widths = (post_bins[1:] - post_bins[:-1]) / 2

    # Plot the distributions together
    fig, axis = plt.subplots(
        2, 2, figsize=fig_size, gridspec_kw={"height_ratios": [3, 1]}
    )
    axis[0, 0].stairs(
        sign_hist / np.max(sign_hist) * np.max(comb_hist) / 5,
        pre_bins,
        fill=True,
        alpha=0.3,
        label=f"Signal ({signal})",
    )
    axis[0, 0].errorbar(
        pre_mid_bins,
        cntrl_hist,
        cntrl_err,
        pre_bin_widths,
        fmt="r",
        linestyle="none",
        label=f"Control ({background})",
    )
    axis[0, 0].errorbar(
        pre_mid_bins,
        comb_hist,
        comb_err,
        pre_bin_widths,
        fmt="g",
        linestyle="none",
        label=f"Doped Sample ({snb_ratio:.0e})",
    )
    axis[0, 1].errorbar(
        post_mid_bins,
        cntrl_pass,
        cntrl_pass_err,
        post_bin_widths,
        fmt="r",
        linestyle="none",
        label=f"Control Passing Cut ({reject_frac})",
    )
    axis[0, 1].errorbar(
        post_mid_bins,
        pass_hist,
        pass_err,
        post_bin_widths,
        fmt="g",
        linestyle="none",
        label="Sample Passing Cut",
    )

    # The p value plots
    axis[1, 0].plot(pre_bins, np.ones_like(pre_bins), "--k")
    axis[1, 0].stairs(orig_pvals, pre_bins, color="r")
    axis[1, 0].stairs(
        orig_pval_down, pre_bins, baseline=orig_pvals, color="r", alpha=0.2, fill=True
    )
    axis[1, 1].plot(post_bins, np.ones_like(post_bins), "--k")
    axis[1, 1].stairs(new_pvals, post_bins, color="r")
    axis[1, 1].stairs(
        new_pval_down, post_bins, baseline=new_pvals, color="r", alpha=0.2, fill=True
    )

    # Neaten up the plot
    for i in range(2):
        for j in range(2):
            ax = axis[i, j]
            if j == 0:
                ax.set_xlim(pre_bins[0], pre_bins[-1])
            if j == 1:
                ax.set_xlim(post_bins[0], post_bins[-1])

            if i == 0:
                ax.set_ylabel("Counts")
                ax.legend(loc=1, frameon=False)
                ax.set_xticks([])
                if j == 0:
                    ax.set_ylim([0, 1.3 * max(cntrl_hist)])
                if j == 1:
                    ax.set_ylim([0, 1.3 * max(pass_hist)])

            if i == 1:
                ax.set_yscale("log")
                ax.set_ylim(min(new_pvals) / 10, 5)
                ax.set_ylabel("local p-value")
                ax.set_xlabel("Mass [GeV]")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    outpath = Path(path) / (
        f"bump_{background}_{signal}_{tagger['label']}.png"
    ).replace(" ", "_").replace(":", "_")
    fig.savefig(outpath)
    plt.close(fig)


def hist_norm_get_err(hist, bins, do_hist=True) -> tuple:
    if do_hist:
        hist, _ = np.histogram(hist)
    divisor = np.array(np.diff(bins), float) / hist.sum()
    err = np.sqrt(hist) * divisor
    hist = hist * divisor
    return hist, err


def get_chisqaure_excess(
    data_orig: np.ndarray,
    data_cut: np.ndarray,
    control_region: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    # Get the histograms of the distributions
    orig_hist, _ = np.histogram(data_orig, bins)
    pass_hist, _ = np.histogram(data_cut, bins)

    # Get the error terms from statistics
    orig_err = np.sqrt(orig_hist)
    pass_err = np.sqrt(pass_hist)

    # Get the scale factor and its error from the control region
    comb_in_cntrl = np.sum(data_orig < control_region)
    pass_in_cntrl = np.sum(data_cut < control_region)
    scale_fact = comb_in_cntrl / pass_in_cntrl
    scale_fact_err = scale_fact * np.sqrt(1 / comb_in_cntrl + 1 / pass_in_cntrl)

    # Apply the scale factor, propagate the uncertainties
    pass_nrm = pass_hist * scale_fact
    pass_nrm_err = pass_nrm * np.sqrt(
        (pass_err / (1e-8 + pass_hist)) ** 2
        + (scale_fact_err / (1e-8 + scale_fact)) ** 2
    )

    # Get the chi squared excess per bin
    excess = (pass_nrm - orig_hist) / np.sqrt(pass_nrm_err**2 + orig_err**2)

    return excess, pass_nrm, pass_nrm_err


def save_roc_curves(
    path: Path | str,
    taggers: list,
    files: list,
    sort_by_auc: bool = True,
    divide_br: bool = True,
    only_to_first: bool = True,
    br_at_eff: list | None = None,
    xlim: tuple | list = (0, 1),
    ylim: tuple | list = (1, 3e4),
    do_log: bool = True,
    fig_size: tuple = (6, 6),
    dataset_type: str = "Rodem",
    is_decor: bool = False,
    return_AUC_and_id: bool = False,
) -> None:
    print(f"Creating ROC curves for using combinations of: {files}")
    taggers = deepcopy(taggers)  # To be safe with pops

    # Cycle through the taggers
    # for t in deepcopy(taggers):
    #     if t["score_name"] == "mass_diff":
    #         continue
    #     t["score_name"] = "ext_decor_" + t["score_name"]
    #     t["label"] = t["label"] + " (decor)"
    #     t["plot_kwargs"]["linestyle"] = "dashed"
    #     taggers.append(t)

    # Start by designing the legend
    label_pad_length = max(len(tag["label"]) for tag in taggers) + 1
    label_pad_length = max([label_pad_length, 7])  # Allow for the Model title
    legend_header = f"{'Model':{label_pad_length}} {'AUC':>6}"
    if br_at_eff is not None:
        legend_header += "    ".join([""] + ["BR@" + str(val)[1:] for val in br_at_eff])

    # Create the list of combinations for the roc curves
    if only_to_first:
        comb = [(files[0], p) for p in files[1:]]
    else:
        comb = list(combinations(files, 2))

    # Cycle through the combinations to plot
    for background, signal in comb:
        print(f" - {signal} vs {background}")

        # Plotting will be done later so it can be in order
        aucs = []
        tprs = []
        frrs = []
        labels = []
        plt_kwargs = []

        best_id = -1
        best_auc = 0

        for tag in taggers:
            try:
                tagger_path = Path(tag.path, tag.name, "outputs")
                tagger_score = tag.score_name

                # Load and combine the two datasets for the roc curves
                scores = []
                targets = []
                for i, file_name in enumerate([background, signal]):
                    file_path = tagger_path / file_name
                    with h5py.File(file_path, "r") as f:
                        scores.append(f[tagger_score][:])
                        targets += len(f[tagger_score]) * [i]
                scores = np.concatenate(scores).flatten()
                targets = np.array(targets)

                if hasattr(tag, "flip") and tag.flip:
                    scores *= -1

                # Calculate the inclusive ROC and the values for the ROC curve
                auc = roc_auc_score(targets, scores)

                if return_AUC_and_id:
                    if auc > best_auc:
                        best_auc = auc
                        best_id = tag.id

                fpr, tpr, _ = roc_curve(targets, scores)
                if divide_br:
                    frr = np.divide(1, np.clip(fpr, 1e-8, None))
                else:
                    frr = 1 - fpr

                # Create the legend entry
                label = f"{tag.label:{label_pad_length}} {auc:.4f}"
                if br_at_eff is not None:
                    fr_at_vals = [frr[np.abs(tpr - val).argmin()] for val in br_at_eff]
                    label += "".join([f"{fr:>9.1f}" for fr in fr_at_vals])

                # Add to the lists so they can be ordered later
                aucs.append(auc)
                tprs.append(tpr)
                frrs.append(frr)
                labels.append(label)
                plt_kwargs.append(tag.plot_kwargs)
            except Exception as e:
                print(f"Failed for {tag.label}")
                print(e)
                print()

        # Create the figure
        fig, axis = plt.subplots(1, 1, figsize=fig_size)

        # Add the title entry for the legend
        axis.plot([], [], color="w", label=legend_header)

        # Sort the plots by auc and iterate through them
        if sort_by_auc:
            idxes = np.argsort(-np.array(aucs))
        else:
            idxes = np.arange(len(aucs))
        for i in idxes:
            axis.plot(tprs[i], frrs[i], label=labels[i], **plt_kwargs[i])

        # Add a diagonal line for random guessing
        x_space = 1 - np.linspace(1e-8, 1, 1000)
        y_space = 1 - np.linspace(1, 1e-8, 1000)
        axis.plot(x_space, y_space, "--k")

        # The name of the background and signal are stripped from the file names
        b_name = background.split("_test")[0]
        s_name = signal.split("_test")[0]

        # Formatting and saving the inclusive roc curves
        if not divide_br:
            ylim = (0, 1)
        axis.set_ylim(ylim)
        axis.set_xlim(xlim)
        axis.set_xticks(np.linspace(0, 1, 11))
        if divide_br:
            axis.set_ylabel(f"({b_name}) " + r"Backround Rejection $1/\epsilon_B$")
        else:
            axis.set_ylabel(f"({b_name}) " + r"Backround Rejection $1-\epsilon_B$")
        axis.set_xlabel(f"({s_name}) " + r"Signal Efficiency $\epsilon_S$")
        axis.grid(which="both", alpha=0.5)

        if divide_br:
            axis.legend(
                prop={"family": "monospace"},
                loc="upper right",
                framealpha=1,
                facecolor="w",
                edgecolor="w",
            )
        else:
            axis.legend(
                prop={"family": "monospace"},
                loc="lower left",
                framealpha=1,
                facecolor="w",
                edgecolor="w",
            )
        if do_log:
            axis.set_yscale("log")
        fig.tight_layout()

        # Create the output directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)

        divide_br_suffix = "_rej" if divide_br else ""
        decor_suffix = "_decor" if is_decor else ""

        outpath = (
            Path(path)
            / f"ROC_{dataset_type}_signal_vs_background{divide_br_suffix}{decor_suffix}.png"
        )

        fig.savefig(outpath)
        plt.close()

        if return_AUC_and_id:
            return best_auc, best_id


def plot_score_distributions(
    output_path: str,
    network: dict,
    processes: dict,
    bins: np.ndarray,
    xlim: list = None,
    ylim: list = None,
    do_log: bool = True,
    score_scalling_fn: callable = None,
    fig_size: tuple = (4, 4),
    fig_format: str = "png",
) -> None:
    """Plot the histograms a single network score for each process.

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A dictionary describing the network with three required keys
            path: The location of the network's exported directory.
            label: The label for the x_axis.
            score_name: Name of the column in the HDF files to use as tagging score.
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        processes:
            A dict of processes to plot. Keys are used for labels, items for filenames.
        bins: The bins to use for the score histograms

    kwargs:
        xlim: The x limits of the plot
        ylim: The y limits of the plot
        do_log: If the y axis is logarithmic
        score_scalling_fn: Function to apply to the scores, useful for bounding
        fig_size: The size of the output figure.
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
    """
    print(
        f"Creating score distribution plots for {network['label']} on  {list(processes.keys())}"
    )

    # Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create the figure
    fig, axis = plt.subplots(1, 1, figsize=fig_size)

    # For each of the processes
    for proc_name, file_name in processes.items():
        # Load the scores from the distributions
        with h5py.File(Path(network["path"]) / file_name, "r") as f:
            scores = f[network["score_name"]][:].flatten()

            if score_scalling_fn is not None:
                scores = score_scalling_fn(scores)

        # Create the histogram
        hist, edges = np.histogram(
            np.clip(scores, bins[0], bins[-1]), bins, density=True
        )

        # Plot as a step function
        axis.step(edges, [0] + hist.tolist(), label=proc_name)

    # Formatting axes
    if do_log:
        axis.set_yscale("log")
        axis.set_ylim(bottom=1e-1)
    else:
        axis.set_ylim(bottom=0)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_ylabel("Entries")
    axis.set_xlabel(f"{network['label']}({network['score_name']})")
    axis.legend()
    fig.tight_layout()

    # Saving image
    outpath = Path(output_path) / (
        f"score_dist_{network['label']}({network['score_name']})"
    ).replace(" ", "_")
    fig.savefig(outpath.with_suffix("." + fig_format))
    plt.close(fig)


def plot_sculpted_distributions(
    output_path: str,
    network: dict,
    processes: dict,
    data_dir: str,
    bins: np.ndarray,
    br_values: np.ndarray,
    var: str = "mass",
    ylim: tuple = None,
    ratio_ylim: tuple = (0, 1),
    do_log: bool = True,
    do_norm: bool = True,
    redo_quantiles: bool = False,
    fig_size: tuple = (10, 4),
    fig_format: str = "png",
) -> np.ndarray:
    """Plot the histograms of the jet masses for different cuts.

    The quantiles are taken using the background process which is the first process
    in the passed dictionary, unless redo_quantiles is True. Here the quantiles
    are recalculated per sample.

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A dictionary describing the network with three required keys
            path: The location of the network's exported directory.
            label: The label for the x_axis.
            score_name: Name of the column in the HDF files to use as tagging score.
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        processes:
            A dict of processes to plot. Keys are used for label, items for filenames.
            Background process must be first!
        data_dir: The original directory of the data to pull the masses
        bins: The bins to use for mass histograms
        br_values: A list of background rejection values, each will be a hist
    kwargs:
        var: The jet variable to plot, must be either: pt, eta, phi, mass
        ylim: The y limits of the plots
        ratio_ylim: The y limits of the ratio plots
        do_log: If the y axis is logarithmic
        do_norm: If the histograms should be normalised
        redo_quantiles: Calculate the quantiles per sample or just background
        fig_size: The size of the output figure.
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
    """
    print(
        f"Creating {var} after cuts on {network['label']} for {list(processes.keys())}"
    )

    # Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create the figure
    if do_norm:
        fig, axes = plt.subplots(1, len(processes), figsize=fig_size)
        axes = np.array([axes])
    else:
        fig, axes = plt.subplots(
            2, len(processes), figsize=fig_size, gridspec_kw={"height_ratios": [3, 1]}
        )
    if len(processes) == 1:
        axes = axes.reshape(2 - do_norm, -1)

    # For each process in the dictionary
    thresholds = None
    var_idx = {"pt": 0, "eta": 1, "phi": 2, "mass": 3}[var]
    for i, (proc_name, proc_file) in enumerate(processes.items()):
        print(f" - loading {proc_name}")

        # Load the jet var from the original HDF files: high lvl = pt, eta, phi, mass
        with h5py.File(Path(data_dir, proc_file), mode="r") as h:
            var_data = h["objects/jets/jet1_obs"][:, var_idx]

        # Load the scores
        with h5py.File(Path(network["path"]) / proc_file, "r") as f:
            scores = f[network["score_name"]][:].flatten()

        # Check that they are of compatible length
        if len(var_data) != len(scores):
            raise ValueError(
                f"The scores in {proc_file} for {network['label']}",
                "do not match original data length",
            )

        # Get the quantiles using the first process (background)
        if thresholds is None or redo_quantiles:
            thresholds = np.quantile(scores, br_values)

        # Make a plot of the original function without any cuts
        orig_hist, edges = np.histogram(
            np.clip(var_data, bins[0], bins[-1]), bins, density=do_norm
        )
        axes[0, i].step(edges, [0] + orig_hist.tolist(), "k", label="Original")

        # Add a dashed line at 1 on the ratio plot
        if not do_norm:
            axes[1, i].step([edges[0], edges[-1]], [1, 1], "--k")

        # For each of the thresholds
        for thresh, br_val in zip(thresholds, br_values):
            # Trim the data
            sel_data = var_data[scores >= thresh]

            # Only histogram if there is more than one sample
            # (sometimes nothing passes threshold)
            if len(sel_data) > 1:
                # Create the histogram in the original plot
                hist, edges = np.histogram(
                    np.clip(sel_data, bins[0], bins[-1]), bins, density=do_norm
                )
                axes[0, i].step(edges, [0] + hist.tolist(), label=f"{br_val:.2f}")

                # Create the ratio plot
                if not do_norm:
                    ratio = hist / (orig_hist + 1e-8)
                    axes[1, i].step(edges, [ratio[0]] + ratio.tolist())

            else:
                axes.plot(edges, [0] * len(edges), label=f"{br_val:.2f}")

        # Formatting the histogram axis
        if do_log:
            axes[0, i].set_yscale("log")
        else:
            axes[0, i].set_ylim(bottom=0)
        axes[0, i].set_ylim(ylim)
        axes[0, i].set_xlim([edges[0], edges[-1]])
        axes[0, i].set_title(proc_name)
        axes[0, i].set_xlabel(var)

        # Formatting the ratio plot
        if not do_norm:
            axes[0, i].set_xlabel("")
            axes[0, i].xaxis.set_ticklabels([])
            axes[1, i].set_ylim(ratio_ylim)
            axes[1, i].set_xlim((edges[0], edges[-1]))
            axes[1, i].set_xlabel(var)

        # Remove y axis ticklabels from middle plots
        if i > 0:
            axes[0, i].yaxis.set_ticklabels([])
            if not do_norm:
                axes[1, i].yaxis.set_ticklabels([])

    # Adjust and save the plot
    axes[0, 0].set_ylabel("a.u.")
    if not do_norm:
        axes[1, 0].set_ylabel("Ratio to Original")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    outpath = Path(
        output_path
    ) / f"{var}_dist_{network['label']}({network['score_name']})".replace(
        " ", "_"
    ).replace(":", "_")
    fig.savefig(outpath.with_suffix("." + fig_format))
    plt.close(fig)

    return np.array(thresholds)


def plot_mass_sculpting(
    path: str,
    taggers: list,
    dataset_type: str,
    file_name: str,
    data_dir: str,
    br_values: np.ndarray,
    bins: np.ndarray | int = 32,
    fig_size: tuple = (5, 5),
    ylim: tuple | None = None,
    xlim: tuple | None = None,
    do_log: bool = False,
    cfg: dict | None = None,
) -> None:
    print(f"Creating mass sculpting plots for {file_name}")
    if cfg is None:
        print("CFG is None")
        taggers = deepcopy(taggers)  # To be safe with pops
    else:
        # Custom tagger list
        print("CFG is NOT None")
        taggers = []
        taggers.append(
            {
                "label": "default",
                "path": f"{cfg.exp_path}/taggers/supervised_{cfg.dataset_type}_default",
                "name": f"dense_{cfg.dataset_type}_default",
                "score_name": "output",
                "plot_kwargs": {"color": "black", "linestyle": "dashed"},
                "decor": False,
            }
        )

        for i in range(len(cfg.method_types)):
            taggers.append(
                {
                    "label": f"{cfg.method_types[i]}",
                    "path": f"{cfg.exp_path}/taggers/supervised_{cfg.dataset_type}_{cfg.method_types[i]}",
                    "name": f"dense_{cfg.dataset_type}_{cfg.method_types[i]}",
                    "score_name": "output",
                    "plot_kwargs": {"color": cfg.color_list[i], "linestyle": "solid"},
                    "decor": cfg.decor,
                }
            )

    # Create the figure
    fig, axis = plt.subplots(1, 1, figsize=fig_size)

    # Load the masses from the original HDF files: high lvl = pt, eta, phi, mass
    print(" - Loading mass...")
    mass = DatasetManager().load_mass(dataset_type=dataset_type, file_name=file_name)

    # Get the original histogram of the masses
    orig_hist = np.histogram(mass, bins, density=True)[0]

    # Cycle through the networks
    max_y = 0
    for tag in taggers:
        # Load the scores
        tag_path = Path(
            tag["path"],
            tag["name"],
            "outputs",
            DatasetManager().get_output_file_name(
                dataset_type=dataset_type,
                file_name=file_name,
                decor=tag.get("decor", False),
            ),
        )
        with h5py.File(tag_path, "r") as f:
            scores = f[tag["score_name"]][:].flatten()

        # Check that they are of compatible length
        assert len(scores) == len(mass)

        # Get the thresholds for the score quantiles
        thresholds = np.quantile(scores, br_values)

        # For each of these thresholds calculate the JD-divergence to the original
        jsd_vals = []
        for thresh in thresholds:
            sel_mass = mass[scores >= thresh]
            hist = np.histogram(sel_mass, bins, density=True)[0]
            jsd_vals.append(js_div(orig_hist, hist))

        # Plot the data for the model
        label = tag["label"] + " (decor)" if tag.get("decor", False) else tag["label"]
        axis.plot(br_values, jsd_vals, label=label, **tag["plot_kwargs"])
        axis.scatter(br_values, jsd_vals, color=tag["plot_kwargs"]["color"], marker="o")
        max_y = max(jsd_vals) if max(jsd_vals) > max_y else max_y

    # Formatting and saving image
    if do_log:
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0)

    if cfg is not None:
        if cfg.decor:
            axis.set_ylim([0, 10 * max_y])
    axis.set_ylim(ylim)
    axis.set_xlim(xlim)
    axis.set_ylabel("Mass Sculpting (JSD)", fontsize=14)
    axis.set_xlabel("Background Rejection", fontsize=14)
    axis.legend(
        prop={"family": "monospace", "size": 14},
        framealpha=0.5,
        facecolor="w",
        edgecolor="w",
    )
    axis.grid(which="both", alpha=0.5)
    fig.tight_layout()

    # Create the output directory if it doesn't exist
    Path(path).mkdir(parents=True, exist_ok=True)

    outpath = Path(path) / f"mass_sculpting_{dataset_type}.pdf"
    fig.savefig(outpath)
    plt.close(fig)


def plot_mass_score_correlation(
    output_path: str,
    network: dict,
    proc_name: str,
    proc_file: str,
    data_dir: str,
    mass_bins: np.ndarray,
    score_bins: np.ndarray,
    do_log: bool = False,
    score_scalling_fn: callable = None,
    cmap: str = "coolwarm",
    fig_format: str = "png",
    fig_size: tuple = (4, 4),
) -> None:
    """Plot and save a 2D heatmap showing the tagger's score verses the jet
    mass.

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A dictionary with three required keys
            path: The location of the network's exported directory.
            label: The label for the x_axis.
            score_name: Name of the column in the HDF files use as disciminant scores
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        proc_name: Name of the process to plot, used for saving filename
        proc_file: Filename of the process to load data in network dir and for mass
        data_dir: The original directory of the data to pull the masses
        mass_bins: The bins to use for mass in the 2d histogram
        score_bins: The bins to use for the scores in the 2d histogram

    kwargs:
        do_log: Use the log of the 2d histogram heights for the colours
        score_scalling_fn: Function to apply to the scores, useful for bounding
        cmap: The cmap to use in the heatmap
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
        fig_size: The size of the output figure.
    """
    print(f"Creating mass-score heatmap for {network['label']} on {proc_name}")

    # Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Load the masses from the original HDF files: high lvl = pt, eta, phi, mass
    print(" - Loading mass...")
    with h5py.File(Path(data_dir, proc_file), mode="r") as h:
        mass = h["objects/jets/jet1_obs"][:, 3]

    # Load the scores
    with h5py.File(Path(network["path"]) / proc_file, "r") as f:
        scores = f[network["score_name"]][:].flatten()

    # Check that they are of compatible length
    if len(mass) != len(scores):
        raise ValueError(
            f"The scores in {proc_file} for {network['label']}",
            "do not match original data length",
        )

    # Apply the scaling function to the scores
    if score_scalling_fn is not None:
        scores = score_scalling_fn(scores)

    # Create the figure
    fig, axis = plt.subplots(1, 1, figsize=fig_size)

    # Create the histogram and plot the heatmap
    hist, xedges, yedges = np.histogram2d(mass, scores, bins=[mass_bins, score_bins])
    hist = hist / hist.sum(axis=-1, keepdims=True)
    imshow = axis.imshow(
        np.log(hist.T) if do_log else hist.T,
        origin="lower",
        cmap=cmap,
        extent=[min(xedges), max(xedges), min(yedges), max(yedges)],
        aspect="auto",
    )

    # Include a colour bar
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(imshow, cax=cax, orientation="vertical")

    # Formatting and saving image
    axis.set_ylabel(f"{network['label']}({network['score_name']})")
    axis.set_xlabel("Mass [GeV]")
    fig.tight_layout()
    outpath = (
        Path(output_path)
        / f"mass_score_heatmap_{proc_name}_{network['label']}({network['score_name']})".replace(
            " ", "_"
        )
    )
    fig.savefig(outpath.with_suffix("." + fig_format))
    plt.close(fig)
