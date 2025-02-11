from typing import Tuple

import numpy as np


def subtract_hardest_constituent_phi(
    phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the phi of the hardest constituent from all the
    constituents."""
    phi -= phi[0]


def subtract_pt_centroid_naive(
    pT: np.ndarray, eta: np.ndarray, phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the pT weighted centroid from all the constituents."""
    # calculate the pT centroid
    phi_centroid = np.sum(pT * phi) / np.sum(pT)
    eta_centroid = np.sum(pT * eta) / np.sum(pT)
    # subtract the coordinates of pT centroid
    phi -= phi_centroid
    eta -= eta_centroid


def corerect_phi_cyclic(phi):
    """Correct the phi values to be in the range [-pi, pi]."""
    phi[phi < -np.pi] += 2 * np.pi
    phi[phi > np.pi] -= 2 * np.pi


def naive_const_preprocessing(
    pT: np.ndarray, eta: np.ndarray, phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the constituents by subtracting the hardest constituent phi,
    the pT weighted centroid and rotating the image so that the major principal
    axis is aligned with the x-axis."""
    subtract_hardest_constituent_phi(phi)
    corerect_phi_cyclic(phi)
    subtract_pt_centroid_naive(pT, eta, phi)
    corerect_phi_cyclic(phi)

    # calculate the moment of inertia tensor
    I_xx = np.sum(pT * (phi) ** 2)
    I_yy = np.sum(pT * eta**2)
    I_xy = np.sum(pT * eta * phi)
    moment = np.array([[I_xx, I_xy], [I_xy, I_yy]])

    # calculate the major principal axis
    w, v = np.linalg.eig(moment)
    if w[0] > w[1]:
        major = 0
    else:
        major = 1

    # turn the immage
    alpha = -np.arctan2(v[1, major], v[0, major])
    phi_new = phi * np.cos(alpha) - eta * np.sin(alpha)
    eta_new = phi * np.sin(alpha) + eta * np.cos(alpha)
    phi = phi_new
    eta = eta_new

    # flip the image according to the largest constituent
    q1 = sum(pT[(phi > 0) * (eta > 0)])
    q2 = sum(pT[(phi <= 0) * (eta > 0)])
    q3 = sum(pT[(phi <= 0) * (eta <= 0)])
    q4 = sum(pT[(phi > 0) * (eta <= 0)])
    indx = np.argmax([q1, q2, q3, q4])
    if indx == 1:
        phi *= -1
    elif indx == 2:
        phi *= -1
        eta *= -1
    elif indx == 3:
        eta *= -1

    return pT, eta, phi


def jetimage(
    csts: np.ndarray,
    bins,
    phi_bounds,
    eta_bounds,
    do_naive_const_preprocessing=True,
    image_transform=None,
) -> np.ndarray:
    """Create a jet image from the constituents."""
    pT_ = csts[..., 0]
    eta_ = csts[..., 1]
    phi_ = csts[..., 2]
    # Jet images ignore mass

    # if the constituents are not boosted/rotated/flipped properly
    # TODO: create all three options separately if some is missing
    if do_naive_const_preprocessing:
        pT, eta, phi = naive_const_preprocessing(pT_.copy(), eta_.copy(), phi_.copy())
    else:
        pT, eta, phi = pT_, eta_, phi_

    image = np.histogram2d(phi, eta, bins, [phi_bounds, eta_bounds], weights=pT)[0]

    # DELETE THIS TO NOT SLOW THE TRAINING
    # if np.sum(image) == 0:
    #     print("WTF")
    #     pT, eta, phi = naive_const_preprocessing(pT_, eta_, phi_)
    #     raise NameError("Image is 0")
    # if np.sum([image > 0]) > 200:
    #     raise NameError("Too many non-zero pixels")

    return np.expand_dims(image.astype(np.float32), axis=0)
