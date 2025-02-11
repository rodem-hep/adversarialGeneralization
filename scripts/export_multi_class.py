"""Create a single dataset using the output of a multiclass classifier."""

from pathlib import Path

import h5py
import numpy as np

# Strings to find files
model_dir = """/srv/beegfs/scratch/groups/rodem/anomalous_jets/
taggers/supervised/transformer_multiclass/"""
score_name = "output"
data_dir = "/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data/"
data_files = [
    "QCD_jj_pt_450_1200_decor_test.h5",
    # "ttbar_allhad_pt_450_1200_test.h5",
    # "WZ_allhad_pt_450_1200_test.h5",
]
n_per_file = 2_000_000


all_masses = []
all_labels = []
all_scores = []
for i, data_file in enumerate(data_files):
    # Load the background masses
    with h5py.File(Path(data_dir, data_file), mode="r") as h:
        masses = h["objects/jets/jet1_obs"][:n_per_file, 3]

    # Load the background tagger scores
    with h5py.File(Path(model_dir, data_file), mode="r") as h:
        scores = np.squeeze(h[score_name][:n_per_file])

    # Check that they are the same length
    assert len(masses) == len(scores)

    # Add to the running lists
    all_masses.append(masses)
    all_scores.append(scores)
    all_labels.append(np.full_like(masses, i))

# Combine the data
all_data = np.concatenate(
    [
        np.hstack(all_masses)[:, None],
        np.hstack(all_labels)[:, None],
        np.vstack(all_scores),
    ],
    axis=-1,
)

# Save to a new hdf file
with h5py.File(Path(model_dir, "qcd_scores.h5"), mode="w") as file:
    file.create_dataset("data", data=all_data)
