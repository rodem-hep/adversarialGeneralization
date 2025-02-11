import numpy as np


def iteratively_build_bins(
    values: np.ndarray,
    min_bw: int = 0,
    min_per_bin: int | float = 0.01,
    min_value: int | float = 0,
    max_value: int | float = float("inf"),
    fixed_bw_interval: bool = False,
) -> np.ndarray:
    """Iteratively build bins for the given values.

    This function iteratively builds bins for the given values by sorting them
    and adding bin edges based on the minimum bin width, minimum number of values
    per bin, and maximum value.

    Parameters
    ----------
    values : np.ndarray
        Array of values to build bins for.
    min_bw : int, optional
        Minimum bin width. Default is 0.
    min_per_bin : int, float
        Minimum number of values per bin. Default is 100.
    max_value : int | float, optional
        Maximum value to include in the bins. Default is `float("inf")`.
    fixed_bw_interval : bool
        If the bin widths have to be fixed intergers of the min_bw. Default is False.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """

    if min_per_bin < 1:
        min_per_bin = int(len(values) * min_per_bin)

    sorted_mass = np.sort(values)
    bin_edges = [min_value]
    bin_count = 0
    for s in sorted_mass:
        bin_count += 1

        # Check to see if we should be putting this event in the next bin instead
        if bin_count > min_per_bin and s - bin_edges[-1] > min_bw:
            bin_edges.append(s)
            bin_count = 1

        # Exit early if we are beyond the limits of the histogram
        if s > max_value:
            # Add the upper limit of the histogram
            bin_edges.append(min(s, max_value))
            break

    return np.array(bin_edges)
