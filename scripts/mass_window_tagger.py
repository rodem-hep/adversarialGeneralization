"""Creates a mass window tagger around a specified mass."""

from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    ####################
    anomaly_mass = 173
    output_dir = "/srv/beegfs/scratch/groups/rodem/anomalous_jets/taggers"
    data_dir = "/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data"

    data_files = [
        "QCD_jj_pt_450_1200_test.h5",
        "H2tbtb_1700_HC_250_test.h5",
        "ttbar_allhad_pt_450_1200_test.h5",
        "WZ_allhad_pt_450_1200_test.h5",
    ]
    ####################

    # Create the directory of the tagger
    out_path = Path(output_dir, f"mass_tagger_{anomaly_mass}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Cycle through each of the test files
    for file in data_files:
        print(file)

        # Load the masses of the test file
        data_file = Path(data_dir, file)
        with h5py.File(data_file, mode="r") as h:
            masses = h["objects/jets/jet1_obs"][:, 3]

        # Get the absolute differences to the specified mass, make neg as its anom score
        abs_diff = -np.abs(anomaly_mass - masses)

        # Get the absolute differences to the mass in
        out_file = Path(out_path, file)
        with h5py.File(out_file, mode="w") as h:
            h.create_dataset("mass_diff", data=abs_diff)


if __name__ == "__main__":
    main()
