import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np

from src.datamodules.jetutils import cstpxpypz_to_jet, pxpypz_to_ptetaphi

file_dir = "/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/20220330_udemontreal"
file_name = "H_HpHm_generation_merged_with_masses_20_40_60_80.h5"
out_path = (
    "/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data/montreal_mix_train.h5"
)

# Load the data and reshape
file_path = Path(file_dir, file_name)
with h5py.File(file_path, mode="r") as file:
    cst_data = file["constituents"][:].astype(np.float32)
cst_data = cst_data.reshape(len(cst_data), 100, 4)

# Splitting in this way does not result in any memory copy
cst_e = cst_data[..., 0:1]
cst_px = cst_data[..., 1:2]
cst_py = cst_data[..., 2:3]
cst_pz = cst_data[..., 3:4]

# Calculate the overall jet kinematics from the constituents
jet_px, jet_py, jet_pz, jet_m, _ = cstpxpypz_to_jet(cst_px, cst_py, cst_pz, cst_e)

# Convert both sets of values to spherical
cst_pt, cst_eta, cst_phi = pxpypz_to_ptetaphi(cst_px, cst_py, cst_pz)
jet_pt, jet_eta, jet_phi = pxpypz_to_ptetaphi(jet_px, jet_py, jet_pz)

# Combine the information and return
cst_data = np.concatenate([cst_pt, cst_eta, cst_phi], axis=-1)
jet_data = np.vstack([jet_pt, jet_eta, jet_phi, jet_m]).T

# Save the data
with h5py.File(out_path, mode="w") as file:
    file.create_dataset("objects/jets/jet1_obs", data=jet_data)
    file.create_dataset("objects/jets/jet1_cnsts", data=cst_data)
