# Franck Rothen - Datasets Utils
# Tools that are useful for doing stuff across different datasets (JetNet, Rodem, TopTag)
import os
import numpy as np
import h5py
from pathlib import Path
import pandas as pd


class DatasetManager:
    def __init__(self) -> None:
        # Initialize paths
        Rodem_data_dir = "/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data"
        Toptag_data_dir = "/srv/beegfs/scratch/groups/rodem/datasets/TopTagging"
        JetNet_data_dir = "/srv/beegfs/scratch/groups/rodem/datasets/jetnet"

        RS3L_data_dir = "/srv/beegfs/scratch/groups/rodem/datasets/RS3L"

        keys = ["JetNet", "TopTag", "Rodem", "RS3L"]
        values = [JetNet_data_dir, Toptag_data_dir, Rodem_data_dir, RS3L_data_dir]

        self.data_dir_dict = dict(zip(keys, values))

        keys = [
            "QCD",
            "ttbar",
            "q",
            "t",
            "QCD_jj_pt_450_1200",
            "ttbar_allhad_pt_450_1200",
            "Hbb",
        ]
        values = [
            "background",
            "signal",
            "background",
            "signal",
            "background",
            "signal",
            "signal",
        ]
        self.signal_background_dir = dict(zip(keys, values))

        return

    def add_data_path(self, path: str, dataset_type: str) -> None:
        self.data_dir_dict[dataset_type] = path
        return

    def get_data_path(self, dataset_type: str) -> str:
        return self.data_dir_dict[dataset_type]

    def load_scores(
        self, tagger: dict, dataset_type: str, file_name: str, dset: str = "test"
    ) -> np.ndarray:
        score_file = Path(
            tagger.path,
            tagger.name,
            "outputs",
            self.get_output_file_name(dataset_type, file_name, dset),
        )

        with h5py.File(score_file, "r") as f:
            scores = f[tagger.score_name][:].flatten()

        return scores

    def load_decor_scores(
        self, tagger: dict, dataset_type: str, file_name: str, dset: str = "test"
    ) -> np.ndarray:
        score_file = Path(
            tagger.path,
            tagger.name,
            "outputs",
            self.get_output_file_name(dataset_type, file_name, dset).replace(
                ".h5", "_decor.h5"
            ),
        )

        with h5py.File(score_file, "r") as f:
            scores = f[tagger.score_name][:].flatten()

        return scores

    def load_mass(
        self, dataset_type: str, file_name: str, dset: str = "test"
    ) -> np.ndarray:
        if dataset_type == "RS3L0":
            # WARNING HARDCODED!!! (TODO)
            n_jets = 500000 if dset == "train" else 100000
            mass = self.load_RS3L_mass(
                dset=dset, process=file_name, n_jets=n_jets, augmentation_id=0
            )

        elif dataset_type == "RS3L2":
            # WARNING HARDCODED!!! (TODO)
            n_jets = 500000 if dset == "train" else 100000
            mass = self.load_RS3L_mass(
                dset=dset, process=file_name, n_jets=n_jets, augmentation_id=2
            )

        elif dataset_type == "RS3L3":
            # WARNING HARDCODED!!! (TODO)
            n_jets = 500000 if dset == "train" else 100000
            mass = self.load_RS3L_mass(
                dset=dset, process=file_name, n_jets=n_jets, augmentation_id=3
            )

        elif dataset_type == "RS3L4":
            # WARNING HARDCODED!!! (TODO)
            n_jets = 500000 if dset == "train" else 100000
            mass = self.load_RS3L_mass(
                dset=dset, process=file_name, n_jets=n_jets, augmentation_id=4
            )

        elif dataset_type == "JetNet":
            data_dir = self.get_data_path(dataset_type)
            from jetnet.datasets import JetNet

            _, mass = JetNet.getData(
                jet_type=file_name,
                data_dir=data_dir,
                particle_features=[],
                jet_features=["mass"],
                split=dset,
                num_particles=30,
            )
            mass = mass.flatten()

        elif dataset_type == "TopTag":

            def cstpxpypz_to_jet(
                cst_px: np.ndarray,
                cst_py: np.ndarray,
                cst_pz: np.ndarray,
                cst_e: np.ndarray = None,
            ) -> tuple:
                """Calculate high level jet variables using only the constituents.

                Args:
                    cst_px: The constituent px
                    cst_py: The constituent py
                    cst_pz: The constituent pz
                    cst_e: The constituent E to calculate total jet energy
                        If none then cst are assumed to be massless and energy = momentum
                """

                # Calculate the total jet momenta
                jet_px = np.squeeze(cst_px).sum(axis=-1)
                jet_py = np.squeeze(cst_py).sum(axis=-1)
                jet_pz = np.squeeze(cst_pz).sum(axis=-1)

                # Calculate the total jet energy
                if cst_e is None:
                    cst_e = np.sqrt(cst_px**2 + cst_py**2 + cst_pz**2)
                jet_e = np.squeeze(cst_e).sum(axis=-1)

                # Calculate the total jet mass
                jet_m = np.sqrt(
                    np.maximum(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0)
                )

                return jet_px, jet_py, jet_pz, jet_m, jet_e

            procs = []
            # for _, prc in datasets.items():
            #     procs.append(prc)
            #     if prc not in ["QCD", "ttbar"]:
            #         raise ValueError(f"Unknown process for toptag jets: {prc}")
            procs = [file_name]

            path = self.get_data_path("TopTag")
            # Load the relevant file using pandas
            cst_data = pd.read_hdf(Path(path, dset + ".h5"), "table", stop=None)

            # Pull out the class labels
            labels = cst_data.is_signal_new.to_numpy()

            # Trim the file based on the requested process then the number of jets
            selection = (("QCD" in procs) & (labels == 0)) | (
                ("ttbar" in procs) & (labels == 1)
            )

            cst_data = cst_data[selection][:-1]

            # Select the constituent columns columns
            col_names = ["E", "PX", "PY", "PZ"]
            cst_cols = [f"{var}_{i}" for i in range(200) for var in col_names]
            cst_data = np.reshape(
                cst_data[cst_cols].to_numpy().astype(np.float32), (-1, 200, 4)
            )

            min_n_csts = 1  # Minimum number of constituents in each jet #TODO: WARNING! THIS IS HARDCODED (should extract from pc_data.yaml or somewhere in the chain)
            # Filter out events with too few constituents
            if min_n_csts > 0:
                min_mask = np.sum(cst_data[..., 0] > 0, axis=-1) >= min_n_csts
                cst_data = cst_data[min_mask]

            # Splitting in this way does not result in any memory copy
            cst_e = cst_data[..., 0:1]
            cst_px = cst_data[..., 1:2]
            cst_py = cst_data[..., 2:3]
            cst_pz = cst_data[..., 3:4]

            # Calculate the overall jet kinematics from the constituents
            _, _, _, jet_m, _ = cstpxpypz_to_jet(cst_px, cst_py, cst_pz, cst_e)
            mass = jet_m

        elif dataset_type == "Rodem":
            data_file = Path(self.get_data_path(dataset_type), f"{file_name}_{dset}.h5")
            with h5py.File(data_file, "r") as f:
                mass = f["objects/jets/jet1_obs"][:, 3]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        return mass

    def get_output_file_name(
        self, dataset_type: str, file_name: str, dset: str = "test", decor: bool = False
    ) -> str:
        decor_suffix = "_decor" if decor else ""
        return "_".join(
            [
                dataset_type,
                self.signal_background_dir[file_name],
                f"{dset}{decor_suffix}.h5",
            ]
        )

    def load_RS3L_mass(
        self,
        dset: str = "test",
        process: str = "QCD",
        n_jets: int = 500000,
        augmentation_id: int = 0,
        jet_ids: list = [9, 10, 11, 12],  # pt, eta, phi, energy
    ) -> np.ndarray:
        """
        Load the mass of the jets
        """

        # Get the path to the dataset
        path = self.get_data_path("RS3L")

        # Load the data
        with h5py.File(Path(path, f"FR_RS3L_{process}_{dset}.h5"), "r") as f:
            jet_data = f["singletons"][:n_jets, augmentation_id, jet_ids]

        jet_pt = jet_data[..., 0:1]
        jet_eta = jet_data[..., 1:2]
        # jet_phi = jet_data[..., 2:3]
        jet_energy = jet_data[..., 3:4]

        mass = np.sqrt(jet_energy**2 - (jet_pt * np.cosh(jet_eta)) ** 2)

        # Change dtype to float32
        mass = mass.astype(np.float32)
        mass = mass.T[0]
        return mass
