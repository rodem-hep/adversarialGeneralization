import torch as T

from mattstools.mattstools.torch_utils import undo_log_squash


def torch_locals_to_jet_pt_mass(
    nodes: T.Tensor, mask: T.BoolTensor, clamp_etaphi: float = 0
) -> T.Tensor:
    """Calculate the overall jet kinematics using only the local info:

    - del_eta
    - del_phi
    - log_squash_pt
    """

    # Calculate the constituent pt, eta and phi
    eta = nodes[..., 0]
    phi = nodes[..., 1]
    pt = undo_log_squash(nodes[..., 2])

    if clamp_etaphi:
        eta = T.clamp(eta, -clamp_etaphi, clamp_etaphi)
        phi = T.clamp(phi, -clamp_etaphi, clamp_etaphi)

    # Calculate the total jet values (always include the mask when summing!)
    jet_px = (pt * T.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * T.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * T.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * T.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = T.clamp(jet_px**2 + jet_py**2, 1e-8, None)
    jet_m = T.clamp(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 1e-8, None)

    jet_pt = T.sqrt(jet_pt)
    jet_m = T.sqrt(jet_m)

    return T.vstack([jet_pt, jet_m]).T
