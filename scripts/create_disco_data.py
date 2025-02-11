"""Create the dataset to fit the splines for decorrelation."""

from pathlib import Path

import h5py


def main() -> None:
    # Define the background datafile
    data_dir = Path("/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data/")
    data_file = Path("montreal_mix_train.h5")
    n_events = 50

    # Create the new file
    new_file = data_dir / (data_file.stem + "_test.h5")
    with h5py.File(new_file, "w") as nf:
        with h5py.File(data_dir / data_file, "r") as f:
            nf.create_group("objects/jets/")
            nf.create_dataset(
                "objects/jets/jet1_cnsts", data=f["objects/jets/jet1_cnsts"][-n_events:]
            )
            nf.create_dataset(
                "objects/jets/jet1_obs", data=f["objects/jets/jet1_obs"][-n_events:]
            )


if __name__ == "__main__":
    main()
