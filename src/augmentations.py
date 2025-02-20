import random
from typing import Optional, Tuple

import numpy as np

from mattstools.mattstools.numpy_utils import min_loc
from mattstools.mattstools.utils import signed_angle_diff
from src.jet_utils import (
    apply_boost,
    get_eng_from_ptetaphiM,
    ptetaphi_to_pxpypz,
    pxpypz_to_ptetaphi,
)

EPS = 1e-8


def crop_constituents(
    csts: np.ndarray, mask: np.ndarray, amount: int = 10, min_allowed: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the last 'amount' of constituents in the jet.

    Will not drop constituents passed the minimum allowed level
    """

    # First check how many we can drop
    n_csts = np.sum(mask)
    allowed_to_drop = min(n_csts - min_allowed, amount)

    # Generate randomly the number of nodes to kill if allowed to drop
    if allowed_to_drop > 0:
        drop_num = np.random.randint(allowed_to_drop + 1)

        # Drop them from the mask and the node features
        mask[n_csts - drop_num : n_csts] = False
        csts[n_csts - drop_num : n_csts] = 0

    return csts, mask


def smear_constituents(
    csts: np.ndarray, mask: np.ndarray, strength: float = 0.1, pt_min: float = 0.5
) -> np.ndarray:
    """Add noise to the constituents eta and phi to simulates soft emmisions.

    Noise is gaussian with mean = 0 and deviation = strength/pT
    The default strength for the blurr is set to 100 MeV
    The default minimum smearing value is 500 MeV
    - https://arxiv.org/pdf/2108.04253.pdf
    """

    pt = np.clip(csts[mask, 0:1], a_min=pt_min, a_max=None)
    smear = np.random.randn(len(pt), 2) * strength / pt
    csts[mask, 1:] += smear

    return csts


def cambridge_aachen(
    csts: np.ndarray, mask: np.ndarray, min_number: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the CA clustering algorithm over the jet until a minimum crieria is
    reached.

    This process takes some time.
    """

    # Constituents start off as massless, but by this process they gain mass
    mass = np.zeros((len(csts), 1))
    csts = np.concatenate([csts, mass], axis=-1)

    while True:
        # Test the exit condition
        if mask.sum() <= min_number:
            break

        # Break up the constituent coordinates
        pt = csts[..., 0]
        eta = csts[..., 1]
        phi = csts[..., 2]
        mass = csts[..., 3]

        # Get the seperation between the nodes
        del_eta = np.expand_dims(eta, -1) - np.expand_dims(eta, -2)
        del_phi = signed_angle_diff(np.expand_dims(phi, -1), np.expand_dims(phi, -2))
        delR_matrix = np.sqrt(del_eta**2 + del_phi**2)

        # Make sure that the fake nodes are considered too far away
        pad_mask = np.expand_dims(mask, -1) * np.expand_dims(mask, -2)
        delR_matrix[~pad_mask] = np.inf

        # Make the diagonal large, can't merge a node with itself
        np.fill_diagonal(delR_matrix, np.inf)

        # Check to see if there are any viable merge candidates
        if np.isinf(np.min(delR_matrix)):
            break

        # Select the two closest constituents
        i, j = min_loc(delR_matrix)
        sel_csts = csts[[i, j]]
        sel_pt = sel_csts[..., 0]
        sel_eta = sel_csts[..., 1]
        sel_phi = sel_csts[..., 2]
        sel_mass = sel_csts[..., 3]

        # Convert to cartesian 4 vector
        px = sel_pt * np.cos(sel_phi)
        py = sel_pt * np.sin(sel_phi)
        pz = sel_pt * np.sinh(sel_eta)
        e = np.sqrt(np.clip(px**2 + py**2 + pz**2 + sel_mass**2, EPS, None))

        # Get the cartesian 4 vector for the combined system of the two particles
        comb_px = px.sum()
        comb_py = py.sum()
        comb_pz = pz.sum()
        comb_p2 = comb_px**2 + comb_py**2 + comb_pz**2

        # Replace element i (pt, eta, phi, mass refer back to original csts ndarray)
        pt[i] = np.sqrt(comb_px**2 + comb_py**2)
        eta[i] = np.arctanh(
            np.clip(comb_pz / (np.sqrt(comb_p2) + EPS), -1 + EPS, 1 - EPS)
        )
        phi[i] = np.arctan2(comb_py, comb_px)
        mass[i] = np.sqrt(np.clip(e.sum() ** 2 - comb_p2, EPS, None))

        # Remove element j
        csts[j] = 0
        mask[j] = False

    # Sort the constituents to take advantage of the smaller padding
    csts = csts[csts[:, 0].argsort()[::-1]][:min_number]
    mask = mask[mask.argsort()][::-1][:min_number]

    return csts, mask


def collinear_splitting(
    csts: np.ndarray, mask: np.ndarray, max_splits: int = 20, min_pt_spit: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Split some of the constituents into a pair ontop of each other.

    In current form this function results in a array which is NOT pt ordered
    Will not split to exceed the padding limit

    kwargs:
        max_splits: Max number of splits allowed
        min_pt_spit: Not allowed to split a particle with less than this pt
    """

    # See how many constituents can be split
    n_csts = np.sum(mask)
    n_splittable = np.sum(csts[:, 0] > min_pt_spit)
    n_to_split = min([max_splits, len(csts) - n_csts, n_splittable])
    n_to_split = np.random.randint(n_to_split + 1)

    # If splitting will take place
    if n_to_split > 0:
        # Randomly choose how many to split and select the idxes of them from the jet
        idx_to_split = np.random.choice(n_splittable, n_to_split, replace=False)
        new_idxes = np.arange(n_to_split) + n_csts

        # Generate the splitting momentum fractions from uniform [0.25, 0.75]
        frc_of_splits = np.random.rand(n_to_split) / 2 + 0.25

        # Add new particles on the end of the array with the same values
        csts[new_idxes] = csts[idx_to_split].copy()
        csts[new_idxes, 0] *= frc_of_splits  # Reduce the pt

        # Subtract the pt fraction from the original locations
        csts[idx_to_split, 0] *= 1 - frc_of_splits

    # Update the mask to reflect the new additions
    mask = csts[:, 0] > 0

    return csts, mask


def rotatate_constituents(
    csts: np.ndarray,
    high: np.ndarray,
    mask: np.ndarray,
    angle: Optional[float] = np.ndarray,
) -> np.ndarray:
    """Rotate all constituents about the jet axis.

    If angle is None it will rotate by a random amount
    """

    # Define the rotation matrix
    angle = angle or np.random.rand() * 2 * np.pi
    c = np.cos(angle)
    s = np.sin(angle)
    rot_matrix = np.array([[c, -s], [s, c]])

    # Apply to variables wrt the jet axis
    del_csts = np.array(
        (csts[mask, 1] - high[1], signed_angle_diff(csts[mask, 2], high[2]))
    )
    del_csts = rot_matrix.dot(del_csts).T
    del_csts += high[1:3]

    # Modify the constituent with the new variables
    csts[mask, 1:] = del_csts

    return csts


def merge_constituents(
    csts: np.ndarray,
    mask: np.ndarray,
    max_del_r: float = 0.05,
    strength: float = 0.2,
    min_allowed: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly merge soft constituents if they are close enough together.

    Similar to the smearing, the merge conditions is based on the pt and a strength
    parameter.

    kwargs:
        max_del_r: Maximum seperation to consider merging
        strength: The strength parameter for weighting the pt, set to 100 MeV
        min_allowed: Will not do any merging that may reduce the constituent count past
    """

    # Count the number of constituents and how many we can drop
    num_csts = mask.sum()
    allowed_to_drop = num_csts - min_allowed
    if allowed_to_drop <= 0:
        return csts, mask

    # Break up the constituents into their respective coordinates
    pt = csts[..., 0]
    eta = csts[..., 1]
    phi = csts[..., 2]

    # Get the max seperation allowed by the sum pt (maker lower triangular)
    sum_pt = np.expand_dims(pt, -1) + np.expand_dims(pt, -2)
    pt_max_del_R = strength / (sum_pt + EPS)
    pt_max_del_R = np.clip(pt_max_del_R, 0, max_del_r)
    pt_max_del_R = np.tril(pt_max_del_R)
    np.fill_diagonal(pt_max_del_R, 0)

    # Get the actual del R seperation matrix
    del_eta = np.expand_dims(eta, -1) - np.expand_dims(eta, -2)
    del_phi = signed_angle_diff(np.expand_dims(phi, -1), np.expand_dims(phi, -2))
    delR_matrix = np.sqrt(del_eta**2 + del_phi**2)

    # Make sure that the fake nodes are considered too far away
    pad_mask = np.expand_dims(mask, -1) * np.expand_dims(mask, -2)
    delR_matrix[~pad_mask] = np.inf

    # Look for where the delR matrix is smaller than the max (lower right)
    pos_merges = delR_matrix < pt_max_del_R

    # Randomly kill off 50% of the merge candidates
    pos_merges[pos_merges != 0] = np.random.random(pos_merges.sum()) > 0.5
    pos_merges = np.argwhere(pos_merges)
    np.random.shuffle(pos_merges)

    # Loop through all candidates and merge them using the 4 momenta
    previous = []
    i_idx = []  # Idxes for the results of the combine
    j_idx = []  # Idxes for the nodes to delete in the combine
    for i, j in pos_merges:
        if i not in previous and j not in previous:
            previous += [i, j]
            i_idx.append(i)
            j_idx.append(j)
            allowed_to_drop -= 1
            if allowed_to_drop <= 0:
                break

    # Combined Momentum
    px = pt[i_idx] * np.cos(phi[i_idx]) + pt[j_idx] * np.cos(phi[j_idx])
    py = pt[i_idx] * np.sin(phi[i_idx]) + pt[j_idx] * np.sin(phi[j_idx])
    pz = pt[i_idx] * np.sinh(eta[i_idx]) + pt[j_idx] * np.sinh(eta[j_idx])

    # Replace element i
    mtm = np.sqrt(px**2 + py**2 + pz**2)
    pt[i_idx] = np.sqrt(px**2 + py**2)
    eta[i_idx] = np.arctanh(np.clip(pz / (mtm + EPS), -1 + EPS, 1 - EPS))
    phi[i_idx] = np.arctan2(py, px)

    # Kill element j
    csts[j_idx] = 0
    mask[j_idx] = False

    # Sort the constituents with respect to pt again (might not be needed)
    # csts = csts[csts[:, 0].argsort()[::-1]]
    # mask = mask[mask.argsort()][::-1]

    return csts, mask


def random_boost(
    cnsts: np.ndarray, jets: np.ndarray, max_boost: float = 0.002
) -> Tuple[np.ndarray, np.ndarray]:
    """Boost a jet and its constituents along the jet axis using a random boost
    vector.

    args:
        cnsts: The kinematics of the jet constituents (pt, eta, phi)
        jets: The kinematics of the jet (pt, eta, phi, mass)
        mask: Boolean array showing padded level of the constituents
    kwargs:
        max_boost: The maximum amount to boost the jet in GeV
    """

    # We need the cartesian representations of the constituent momenta
    cst_px, cst_py, cst_pz = ptetaphi_to_pxpypz(cnsts)
    cst_e = cnsts[..., 0] * np.cosh(cnsts[..., 1])  # From pt and eta
    cnsts = np.stack([cst_e, cst_px, cst_py, cst_pz], axis=1)

    # The boosting direction comes from the jet direction
    boost = ptetaphi_to_pxpypz(jets)
    boost = boost / np.linalg.norm(boost) * np.random.uniform(0, max_boost)

    # Apply the boost to the constituents
    cnsts = apply_boost(cnsts, -boost)
    cnsts = np.stack(pxpypz_to_ptetaphi(cnsts[..., 1:]), axis=1).astype("f")

    # Apply the boost the jet too
    jet_m = jets[-1]
    jet_eng = get_eng_from_ptetaphiM(jets)
    boosted_jets = np.hstack([jet_eng, *ptetaphi_to_pxpypz(jets)])
    boosted_jets = np.expand_dims(boosted_jets, 0)
    boosted_jets = apply_boost(boosted_jets, -boost)
    boosted_jets = pxpypz_to_ptetaphi(np.squeeze(boosted_jets)[1:])
    boosted_jets = np.append(boosted_jets, jet_m)

    return cnsts.astype("f"), boosted_jets.astype("f")


def apply_augmentations(
    csts: np.ndarray,
    high: np.ndarray,
    mask: np.ndarray,
    augmentation_list: list,
    augmentation_prob: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a sequence of jet augmentations based on a list of strings.

    args:
        csts: The constituent pt, eta, phi of the jet
        high: The jet pt, eta, phi, mass
        mask: A bool array showing the real vs padded nodes
        augmentation_list: A list of augmentations to apply in order to the jets
    """

    # Iterate through each of the augmentations and apply them in turn
    for aug in augmentation_list:
        # Check using the augmentation prob (rotation always happens!)
        if (
            aug == "rotate"
            or augmentation_prob == 1.0
            or augmentation_prob < random.random()
        ):
            # Rotation is AWLAYS done as it does not change the physics of the jet at all
            if aug == "rotate":
                csts = rotatate_constituents(csts, high, mask)
            elif aug == "smear":
                csts = smear_constituents(csts, mask)
            elif "crop" in aug:
                csts, mask = crop_constituents(csts, mask, int(aug.split("-")[1]))
            elif "merge" in aug:
                csts, mask = merge_constituents(csts, mask, float(aug.split("-")[1]))
            elif "split" in aug:
                csts, mask = collinear_splitting(csts, mask, int(aug.split("-")[1]))
            elif "boost" in aug:
                csts, high = random_boost(
                    csts, high, max_boost=float(aug.split("-")[1])
                )
            elif "CA" in aug:
                csts, mask = cambridge_aachen(csts, mask, int(aug.split("-")[1]))

            # Raise error for unknown augmentation
            else:
                raise ValueError(f"Unrecognised augmentation: {aug}")

    return csts, high, mask
