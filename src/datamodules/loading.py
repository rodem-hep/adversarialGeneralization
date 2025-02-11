from abc import abstractmethod
from copy import deepcopy
from itertools import cycle, islice
from pathlib import Path

from typing import Mapping

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from src.datasets_utils import DatasetManager

from jutils.data_loading import read_all_data

from jetnet.datasets import JetNet

import h5py

from src.jet_utils import cstptetaphi_to_jet, cstpxpypz_to_jet, pxpypz_to_ptetaphi

def process_csts(cst_data, n_csts=-1):
    """Given cst_data of shape [n_samples, n_csts, 4] Where the last axis has
    e, px, py, pz, convert to pt, eta, phi and calculate the jet level
    information."""
    # Splitting in this way does not result in any memory copy
    cst_e = cst_data[..., 0:1]
    cst_px = cst_data[..., 1:2]
    cst_py = cst_data[..., 2:3]
    cst_pz = cst_data[..., 3:4]

    # Calculate the overall jet kinematics from the constituents
    jet_px, jet_py, jet_pz, jet_m, _ = cstpxpypz_to_jet(cst_px, cst_py, cst_pz, cst_e)

    # Limit constituent data to the number of requested nodes
    cst_px = cst_px[:, :n_csts]
    cst_py = cst_py[:, :n_csts]
    cst_pz = cst_pz[:, :n_csts]

    # Convert both sets of values to spherical
    cst_pt, cst_eta, cst_phi = pxpypz_to_ptetaphi(cst_px, cst_py, cst_pz)
    jet_pt, jet_eta, jet_phi = pxpypz_to_ptetaphi(jet_px, jet_py, jet_pz)

    # Combine the information and return
    cst_data = np.concatenate([cst_pt, cst_eta, cst_phi], axis=-1)
    jet_data = np.vstack([jet_pt, jet_eta, jet_phi, jet_m]).T
    return cst_data, jet_data


def load_data(
            dataset_type: str,
            dset: str,
            datasets: Mapping,
            n_jets: int = -1,
            n_csts: int = -1,
            min_n_csts: int = 0,
            incl_substruc: bool = False,
            leading: bool = False,
            recalculate_jet_from_pc: bool = False,
            rodem_predictions_path: str = None,
            score_name: str = "output",
        ) -> tuple:
            """
            Load data from the toptagging reference dataset or Rodem dataset and return them as numpy arrays.

            Parameters
            ----------
            dataset_type : str
                Either "TopTag" or "Rodem"
            dset : str
                Either train, test, or val
            datasets : Mapping
                dict containing which processes to load
            n_jets : int, optional
                The number of jets to load, split for each process (-1 = load all). Defaults to -1.
            n_csts : int, optional
                The number of constituents to load per jet (can be zero). Defaults to -1.
            min_n_csts : int, optional
                Minimum number of constituents per jet. Defaults to 0.
            incl_substruc : bool, optional
                Whether to include substructure variables. Defaults to False.
            leading : bool, optional
                Whether to only include leading jet. Defaults to False.
            recalculate_jet_from_pc : bool, optional
                Whether to recalculate jet from particle constituents. Defaults to False.

            Returns
            -------
            tuple
                High level jets variables [pt, eta, phi, M], constituent variables [pt, eta, phi] (empty if n_cnsts is 0), a boolean array showing the real vs padded elements of the cst data (empty if n_cnsts is 0) and jet class variables. The class label is based on the order of datasets list.

            Raises
            ------
            ValueError
                If an unknown process is encountered for TopTag jets.
            """        
            if(rodem_predictions_path == None):
                if dataset_type == "RS3L0":
                    return load_rs3l(dset, datasets, n_jets, n_csts, augmentation_id=0) # nominal scenario
                elif dataset_type == "RS3L2":
                    return load_rs3l(dset, datasets, n_jets, n_csts, augmentation_id=2) # changing the scale controlling the probability for final state radiation by 1/sqrt(2)
                elif dataset_type == "RS3L3":
                    return load_rs3l(dset, datasets, n_jets, n_csts, augmentation_id=3) # changing the scale controlling the probability for final state radiation by sqrt(2)
                elif dataset_type == "RS3L4":
                    return load_rs3l(dset, datasets, n_jets, n_csts, augmentation_id=4) # using Herwig7 as parton shower
                

                elif dataset_type == "Rodem":
                    return load_rodem(dset, datasets, n_jets, n_csts, min_n_csts, incl_substruc, leading, recalculate_jet_from_pc)

                elif dataset_type == "TopTag":
                    return load_toptag(dset, datasets, n_jets, n_csts, min_n_csts)
                
                elif dataset_type == "JetNet":
                    return load_jetnet(dset, datasets, n_jets, n_csts, min_n_csts)
                
                else:
                    raise ValueError(f'Unknown dataset type: {dataset_type}')
            else:
                if dataset_type == "JetNet":
                    return load_rodem_prediction_of_jetnet(rodem_predictions_path, dset, datasets, n_jets, n_csts, min_n_csts, score_name=score_name)
                
                else:
                    raise ValueError(f'Unknown dataset type: {dataset_type}')

def load_rs3l(
        dset: str,
        datasets: Mapping,
        n_jets: int = 300000,
        n_csts: int = 40,
        augmentation_id: int = 0,
        jet_ids: list = [9, 10, 11, 12], # pt, eta, phi, energy
        csts_ids: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #['pt','relpt','eta','phi','dr','e','rele','charge','pdgid','d0','dz']
) -> tuple:
    """
    Load jets and constituents from the RS3L virtual dataset and return
    them as numpy arrays defined by:

    - pt, eta, phi, mass (and extra vars for jets)

    jettype:
    1: q (light)
    2: c
    3: b
    4: H->bb
    5: g->qq
    6: g->cc
    7: g->bb
    8: g->gg
    9: W->two quarks
    10: Z->qq
    11: Z->bb

    Augmentations:
    0. nominal scenario: jet showered with Pythia8
    1. changing the numerical seed in Pythia8
    2. changing the scale controlling the probability for final state radiation by 1/sqrt(2)
    3. changing the scale controlling the probability for final state radiation by sqrt(2)
    4. using Herwig7 as parton shower

    """

    print(f"Loading {n_jets} n_jets with {n_csts} n_csts from the {dset} RS3L dataset")
    print(f" -- processes selected: {datasets}")
    

    # Get the path to the dataset
    path = DatasetManager().get_data_path("RS3L")

    jet_data = []
    csts_data = []
    mask = []
    labels = []


    for i in range(len(datasets)):
        
        print(f"Loading {datasets['c' + str(i)]} class")

        # Load the data
        with h5py.File(Path(path,f"FR_RS3L_{datasets['c' + str(i)]}_{dset}.h5"), "r") as f:
            # Get jet and constituents data
            if n_jets > len(f["singletons"]):
                print(f"Warning: The dataset contains only {len(f['singletons'])} jets. The requested number of jets {n_jets} was not reached.")

            jet_class_data = f["singletons"][:n_jets, augmentation_id, jet_ids]
            csts_class_data = f["jet_pflow_cands"][:n_jets, augmentation_id, csts_ids, :n_csts]

            # Reorder the shape of the constituents data
            csts_class_data = csts_class_data.transpose(0, 2, 1) # (n_jets, n_csts, n_features)

            # Create mask for the jet constituents (based on pT)
            mask_class_data = csts_class_data[..., 0] > 0

            # Append the data to the containers
            jet_data.append(jet_class_data)
            csts_data.append(csts_class_data)
            mask.append(mask_class_data)
            labels.append(np.ones(len(jet_class_data)) * i)

    # Concatenate the data
    jet_data = np.concatenate(jet_data)
    csts_data = np.concatenate(csts_data)
    mask = np.concatenate(mask)
    labels = np.concatenate(labels)

    # Change dtype to float32
    jet_data = jet_data.astype(np.float32)
    csts_data = csts_data.astype(np.float32)

    return jet_data, csts_data, mask, labels

def load_rs3l_old(
        dset: str,
        datasets: Mapping,
        n_jets: int = 300000,
        n_csts: int = 40,
        augmentation_id: int = 0,
        max_file_id: int = 6,
        jet_ids: list = [9, 10, 11, 12], # pt, eta, phi, energy
        csts_ids: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #['pt','relpt','eta','phi','dr','e','rele','charge','pdgid','d0','dz']
        signal_ids: list = [4], # H->bb
        background_ids: list = [1, 2, 3, 5, 6, 7, 8], # light, c, b, g->qq, g->cc, g->bb, g->gg
) -> tuple:
    """
    Load jets and constituents from the RS3L virtual dataset and return
    them as numpy arrays defined by:

    - pt, eta, phi, mass (and extra vars for jets)

    jettype:
    1: q (light)
    2: c
    3: b
    4: H->bb
    5: g->qq
    6: g->cc
    7: g->bb
    8: g->gg
    9: W->two quarks
    10: Z->qq
    11: Z->bb

    Augmentations:
    0. nominal scenario: jet showered with Pythia8
    1. changing the numerical seed in Pythia8
    2. changing the scale controlling the probability for final state radiation by 1/sqrt(2)
    3. changing the scale controlling the probability for final state radiation by sqrt(2)
    4. using Herwig7 as parton shower

    """

    id_dir = {"Hbb": [4], "QCD": [1, 2, 3, 5, 6, 7, 8]}

    print(f"Loading {n_jets} n_jets with {n_csts} n_csts from the {dset} RS3L dataset")
    print(f" -- processes selected: {datasets}")
    

    # Get the path to the dataset
    path = DatasetManager().get_data_path("RS3L")
    class_event_count = np.zeros(len(datasets), dtype=int)
    initialized = False

    file_id = 0 if dset == "train" else max_file_id + 1 # Temporary fix for the test set
    max_file_id = max_file_id if dset == "train" else max_file_id + 1 # Temporary fix for the test set

    if n_jets == -1: # Temporary fix for the test set
        n_jets = np.inf


    while (class_event_count < n_jets).any() and file_id <= max_file_id:
        # Load the data
        with h5py.File(Path(path,f"rs3l_{file_id}.h5"), "r") as f:
            # Get jet and constituents data
            jet_file_data = f["singletons"][:-1, augmentation_id, [0] + jet_ids]
            csts_file_data = f["jet_pflow_cands"][:-1, augmentation_id, csts_ids, :n_csts]

            # Remove unrequested jet types and assign labels
            file_labels = np.ones(jet_file_data.shape[0])
            file_labels = -file_labels.astype(int) # labels = -1 will be used to mask unwanted jets type

        for i in range(len(datasets)):
            file_labels = np.where(np.isin(jet_file_data[:, 0], id_dir[datasets[f'c{i}']]), i, file_labels) 

        jet_file_data = jet_file_data[file_labels != -1]
        csts_file_data = csts_file_data[file_labels != -1]
        file_labels = file_labels[file_labels != -1]

        # Remove jet type column
        jet_file_data = jet_file_data[:, 1:]   

        # Reorder the shape of the constituents data
        csts_file_data = csts_file_data.transpose(0, 2, 1) # (n_jets, n_csts, n_features)

        # Create mask for the jet constituents (based on pT)
        file_mask = csts_file_data[..., 0] > 0

        # Append the data to the containers
        for i in range(len(datasets)):
            file_event_count = min(len(file_labels[file_labels == i]), n_jets - class_event_count[i])

            if file_event_count <= 0:
                continue

            jet_data = jet_file_data[file_labels==i][:file_event_count] if initialized == False else np.concatenate([jet_data, jet_file_data[file_labels==i][:file_event_count]])
            csts_data = csts_file_data[file_labels==i][:file_event_count] if initialized == False else np.concatenate([csts_data, csts_file_data[file_labels==i][:file_event_count]])
            mask = file_mask[file_labels==i][:file_event_count] if initialized == False else np.concatenate([mask, file_mask[file_labels==i][:file_event_count]])
            labels = file_labels[file_labels==i][:file_event_count] if initialized == False else np.concatenate([labels, file_labels[file_labels==i][:file_event_count]])

            class_event_count[i] += file_event_count

            initialized = True

        file_id += 1

    if (class_event_count < n_jets).any():
            min_event_count = min(class_event_count)
            print(f"WARNING: Only {min_event_count} jets were found in the dataset with max_file_id = {max_file_id}. The requested number of jets was not reached.")
            print(f"Downsampling...")

            # Indices of each class labels
            class_indices = [np.where(labels == i)[0] for i in range(len(datasets))]
            # Take the first min_event_count indices
            class_indices = [class_indices[i][:min_event_count] for i in range(len(datasets))]
            # Concatenate the indices
            indices = np.concatenate(class_indices)

            jet_data = jet_data[indices]
            csts_data = csts_data[indices]
            mask = mask[indices]
            labels = labels[indices]


    
    # # Shuffle the data
    # shuffling = np.random.permutation(len(jet_data))
    # jet_data = jet_data[shuffling]
    # csts_data = csts_data[shuffling]
    # mask = mask[shuffling]
    # labels = labels[shuffling]

    # Change dtype to float32
    jet_data = jet_data.astype(np.float32)
    csts_data = csts_data.astype(np.float32)

    return jet_data, csts_data, mask, labels

def load_toptag(
    dset: str,
    datasets: Mapping,
    n_jets: int = -1,
    n_csts: int = -1,
    min_n_csts: int = 0,
) -> tuple:
    """Load jets and constituents from the toptagging reference dataset and
    return them as numpy arrays defined by:

    - pt, eta, phi, mass

    kwargs:
        dset: Either train, test, or val
        datasets: dict containing which processes to load
        n_jets: The number of jets to load, split for each process (-1 = load all)
        n_csts: The number of constituents to load per jet (can be zero)

    returns:
        jet_data: High level jets variables [pt, eta, phi, M, *substruc]
        cst_data: Constituent vairables [pt, eta, phi]
            empty if n_cnsts is 0
        mask: A boolean array showing the real vs padded elements of the cst data
            empty if n_cnsts is 0
        labels: Jet class variables
            Class label is based on order of datasets list
            min_n_csts: Minimum number of constituents per jet
    """

    procs = []
    for _, prc in datasets.items():
        procs.append(prc)
        if prc not in ["QCD", "ttbar"]:
            raise ValueError(f"Unknown process for toptag jets: {prc}")

    print(f"Loading {n_jets} jets from the {dset} set in the TopTagging ref. dataset")
    print(f" -- processes selected: {datasets}")

    # get the path to the data
    path = DatasetManager().get_data_path("TopTag")

    # Load the relevant file using pandas
    # n_jets = None # if n_jets != -1 else None #TODO: This was uncommented in matthew's code (WHY???)
    cst_data = pd.read_hdf(Path(path, dset + ".h5"), "table", stop=None) # I changed this to none to avoid having no data of a certain class (This is stupid) TODO

    # Pull out the class labels
    labels = cst_data.is_signal_new.to_numpy()

    # Trim the file based on the requested process then the number of jets
    selection = (("QCD" in procs) & (labels == 0)) | (
        ("ttbar" in procs) & (labels == 1)
    )
    cst_data = cst_data[selection][:n_jets]
    labels = labels[selection][:n_jets]

    # Select the constituent columns columns
    col_names = ["E", "PX", "PY", "PZ"]
    cst_cols = [f"{var}_{i}" for i in range(200) for var in col_names]
    cst_data = np.reshape(
        cst_data[cst_cols].to_numpy().astype(np.float32), (-1, 200, 4)
    )

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
    jet_px, jet_py, jet_pz, jet_m, _ = cstpxpypz_to_jet(cst_px, cst_py, cst_pz, cst_e)

    # Limit constituent data to the number of requested nodes
    cst_px = cst_px[:, :n_csts]
    cst_py = cst_py[:, :n_csts]
    cst_pz = cst_pz[:, :n_csts]

    # Convert both sets of values to spherical
    cst_pt, cst_eta, cst_phi = pxpypz_to_ptetaphi(cst_px, cst_py, cst_pz)
    jet_pt, jet_eta, jet_phi = pxpypz_to_ptetaphi(jet_px, jet_py, jet_pz)

    # Combine the information and return
    cst_data = np.concatenate([cst_pt, cst_eta, cst_phi], axis=-1)
    jet_data = np.vstack([jet_pt, jet_eta, jet_phi, jet_m]).T

    # Get the mask from the cst pt
    mask = np.squeeze(cst_pt > 0)

    return jet_data, cst_data, mask, labels


def load_rodem(
    dset: str,
    datasets: Mapping,
    n_jets: int = -1,
    n_csts: int = -1,
    min_n_csts: int = 0,
    incl_substruc: bool = False,
    leading: bool = True,
    recalculate_jet_from_pc: bool = False,
) -> tuple:
    """Load jets and constituents from the rodem's virtual dataset and return
    them as numpy arrays defined by:

    - pt, eta, phi, mass (and extra vars for jets)

    Args:
        dset: Either train or test, which type of data to pull from
        datasets: Config containing which files to load into each class
        n_jets: The number of jets to load, split for each process (-1 = load all)
        n_csts: The number of constituents to load per jet (can be zero)
        min_n_csts: Minimum number of constituents per jet
        incl_substruc: If the jet substructure vars should be in the high lvl array
        leading: If only to return the leading jet in each file
        recalculate_jet_from_pc: Recalculate the jet kinematics using the point clouds

    Returns:
        jet_data: High level jets variables [pt, eta, phi, M, *substruc]
        cst_data: Constituent vairables [pt, eta, phi]
            empty if n_cnsts is 0
        mask: A boolean array showing the real vs padded elements of the cst data
            empty if n_cnsts is 0
        labels: Jet class variables
            Class label is based on order of datasets list
    """
    print(f"Loading {n_jets} jets from the {dset} set in the RODEM dataset")
    print(f" -- processes selected: {datasets}")

    # Prevent the dictionary from being modified
    datasets = deepcopy(datasets)

    # Get the path to the data
    path = DatasetManager().get_data_path("Rodem")

    # Cycle through the files and add the path and dataset suffix
    for k, files in datasets.items():
        if isinstance(files, str):
            files = files.split(",")
            datasets[k] = files
        for i, file in enumerate(files):
            datasets[k][i] = str(Path(path, f"{file}_{dset}.h5"))

    # Read in data for each as numpy arrays
    data = read_all_data(
        config={
            "data_sets": datasets,
            "data_type": {
                "info": "all",
                "n_jets": n_jets,
                "n_cnsts": n_csts,
                "leading": leading,
                "incl_substruc": incl_substruc,
                "astype": "float32",
            },
        }
    )

    # Decode the results and put them into single tensors
    jet_data = np.concatenate([prcs[0] for _, prcs in data.items()])
    cst_data = np.concatenate([prcs[1][..., :3] for _, prcs in data.items()])  # No M

    # Labels are based on the ordering of the datasets dictionary
    labels = (
        np.concatenate(
            [len(cls[0]) * [cls_i] for cls_i, (_, cls) in enumerate(data.items())]
        )
        .flatten()
        .astype(np.int64)
    )

    # Filter out events with too few constituents
    if min_n_csts > 0:
        min_mask = np.sum(cst_data[..., 0] > 0, axis=-1) >= min_n_csts
        cst_data = cst_data[min_mask]
        jet_data = jet_data[min_mask]
        labels = labels[min_mask]

    # Get the mask based on pt
    mask = cst_data[..., 0] > 0

    # Clip the jet masses due to small error in RODEM data
    jet_data[..., 3:4] = np.clip(jet_data[..., 3:4], a_min=0, a_max=None)

    # If the high level jet variables should be recalculated from the constituents
    # This is to fix discrepancies with jet mass etc
    if recalculate_jet_from_pc:
        jet_data = cstptetaphi_to_jet(cst_data, mask, jet_data)

    print("loading complete")

    return jet_data, cst_data, mask, labels

def load_jetnet(
    dset: str,
    datasets: Mapping,
    n_jets: int = -1,
    n_csts: int = 30,
    min_n_csts: int = 0,
    recalculate_jet_from_pc: bool = False,
) -> tuple:

    # We want to be able to combine multiple sets into the same class
    # So we build this iteratively
    all_csts = []
    all_high = []
    all_mask = []
    all_labels = []

    # Get the path to the data
    path = DatasetManager().get_data_path("JetNet")

    for i, val in enumerate(datasets.values()):

        # Load the selected datasets for this class
        csts, high = JetNet.getData(
            jet_type = val.split(","),
            data_dir = path,
            particle_features = ["ptrel", "etarel", "phirel", "mask"],
            jet_features = ["pt", "eta", "mass", "num_particles"], #set phi to 0
            split = dset,
            num_particles = n_csts,
        )

        #Setting phi to zero as it is missing from the JetNat dataset
        high = np.c_[high[:,:2], np.zeros_like(high)[:,:1], high[:,2:]]

        # Convert from numpy 64, too large
        csts = csts.astype(np.float32)
        high = high.astype(np.float32)

        # Trim the data based on the requested number of jets
        if 0 < n_jets < len(csts):
            csts = csts[: n_jets]
            high = high[: n_jets]

        # We need the constituents in terms of pt, eta, phi, not relative vals
        csts[..., 0] *= high[..., 0:1] # ptrel is a fraction
        csts[..., 1] += high[..., 1:2]
        csts[..., 2] += high[..., 2:3] 

        # Split off the mask and number of particles
        csts, mask = np.split(csts, [-1], axis=2)
        high, num_csts = np.split(high, [-1], axis=1)
        mask = np.squeeze(mask.astype(int))
        num_csts = np.squeeze(num_csts)

        # Mask the data based on the number of csts
        if min_n_csts > 0:
            min_mask = num_csts >= min_n_csts
            csts = csts[min_mask]
            high = high[min_mask]

        # Get the label based on the order of the datasets dict
        labels = np.full((len(csts), ), i, dtype="int")

        # If the high level jet variables should be recalculated from the constituents
        # This is to fix discrepancies with jet mass etc
        if recalculate_jet_from_pc:
            high = cstptetaphi_to_jet(cst_data, mask, jet_data)

        # Add the current classes to the list
        all_csts.append(csts)
        all_high.append(high)
        all_mask.append(mask)
        all_labels.append(labels)

    # Combine the lists into single arrays
    all_csts = np.concatenate(all_csts, axis=0)
    all_high = np.concatenate(all_high, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Convert mask to boolean
    all_mask = all_mask.astype(bool)


    return all_high, all_csts, all_mask, all_labels

def load_jetclass(filepath, treename=None, n_csts: int = 64):
    """Load a file from the JetClass dataset in a way that is consistent with
    RODEM loading. Available features (accessed using branches):

    ['part_px', 'part_py', 'part_pz', 'part_energy', 'part_deta',
    'part_dphi', 'part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr',
    'part_charge', 'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon', 'label_QCD',
    'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql',
    'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl', 'jet_pt',
    'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles', 'jet_sdmass',
    'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4', 'aux_genpart_eta',
    'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt',
    'aux_truth_match']
    """

    branches = ["part_energy", "part_px", "part_py", "part_pz"]
    all_labels = [
        "label_QCD",
        "label_Tbl",
        "label_Tbqq",
        "label_Wqq",
        "label_Zqq",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
    ]

    with uproot.open(filepath) as f:
        if treename is None:
            treenames = {
                k.split(";")[0]
                for k, v in f.items()
                if getattr(v, "classname", "") == "TTree"
            }
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError(
                    "Need to specify `treename` as more than one trees are found in file %s: %s"
                    % (filepath, str(treenames))
                )
        tree = f[treename]
        outputs = tree.arrays(filter_name=branches, library="ak")
        labels = tree.arrays(filter_name=all_labels, library="pd")

    # awk_arr = ak.fill_none(ak.pad_none(outputs, 64, clip=True), 0)
    # Note, unless the clip value is set to be at or above the maximum number of nodes
    # awkward will set some masks all to zero
    awk_arr = ak.pad_none(outputs, n_csts, clip=True)
    part_data = np.stack(
        [ak.to_numpy(awk_arr[n]).astype("float32").data for n in branches], axis=1
    ).transpose(0, 2, 1)

    nan_mx = np.isnan(part_data)
    # TODO check that there aren't any that don't all have zeros?
    mask = ~np.any(nan_mx, axis=-1)
    part_data[nan_mx] = 0

    part_data, jet_data = process_csts(part_data, n_csts)
    mask = mask[:, :n_csts]

    # Just use the index as the label, not dealing with one hots here
    labels = labels[all_labels].to_numpy().astype(np.float32).argmax(axis=1)

    return jet_data, part_data, mask, labels


# In newer versions of itertools this is in the package.
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def load_files(file_paths, load_function, n_csts, shuffle=True):
    data_list = []
    for file in file_paths:
        data_list += [load_function(file, n_csts=n_csts)]
    data = [np.vstack([d[i] for d in data_list]) for i in range(len(data_list[0]))]
    # Reshape the labels
    data[-1] = data[-1].reshape(-1, 1)
    if shuffle:
        indx = np.random.permutation(len(data[0]))
        data = [d[indx] for d in data]
    return data


class IteratorBase:
    @abstractmethod
    def __next__(self):
        # return jet_data, part_data, mask, labels
        raise NotImplemented()

    @abstractmethod
    def get_nclasses(self):
        # Return number of classes in the dataset
        raise NotImplemented()


class JetClassIterator(IteratorBase):
    def __init__(
        self, dset: str, n_load: int, n_nodes: int = 64, processes: list = None
    ) -> None:
        """
        dset: string [train, test, val]
        n_load: Number of files to load. When using an iterator data is stored in many different files, the data in each file often isn't shuffled (containing samples all from one class for example). So load several files and shuffle the loaded samples.
        """
        self.n_load = n_load
        self.dset = dset
        # Get the path to the set of files to load
        # TODO unhardcode
        data_path = Path(
            "/srv/beegfs/scratch/groups/rodem/anomalous_jets/data/JetClass/Pythia/"
        )
        if dset == "train":
            direct = data_path / "train_100M"
            self.n_samples = 100_000_000
        elif dset == "test":
            direct = data_path / "test_20M"
            # TODO make sure this is always the same 2 million!
            self.n_samples = 2_000_000
        else:
            direct = data_path / "val_5M"
            self.n_samples = 1_000_000
        proc_dict = {
            "QCD": ["ZJets"],
            "WZ": ["ZTo", "WTo"],
            "ttbar": ["TTBar_", "TTBarLep"],
            "higgs": ["HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q"],
        }
        if isinstance(processes, str):
            processes = [processes]
        elif processes == None:
            processes = proc_dict.keys()
        self.file_list = []
        self.processes = processes
        for process in processes:
            proc = proc_dict[process]
            # Load the files
            for pr in proc:
                proc_files = np.array(list(direct.glob(f"{pr}*.root")))
                if dset == "test":
                    # Order the files by number
                    proc_files = proc_files[
                        np.argsort(
                            [int(file.stem.split("_")[-1]) for file in proc_files]
                        )
                    ]
                self.file_list += [proc_files]
        # Make an interleaved shuffled (glob grabs randomly) list of the files
        self.file_list = np.array(self.file_list).transpose().flatten().tolist()
        # Build an infinite iterator over the file list
        # TODO using cycle slightly minimising the amount of shuffling that is done, can you do better?
        self.file_iterator = batched(cycle(self.file_list), self.n_load)
        # TODO can this class be generalised?
        self.n_nodes = n_nodes
        self.load_func = load_jetclass
        # Set the index
        self.data_i = 0
        self.load_data()

    def load_data(self):
        # Reset the counting index
        self.data_i = 0
        files = next(self.file_iterator)
        # This returns: self.data = (jet_data, part_data, mask, labels)
        self.data = load_files(
            files, self.load_func, self.n_nodes, shuffle=self.dset != "test"
        )

        # TODO this is a hacky shit thing to do specific that doesn't even really work!
        if len(self.processes) == 1:
            process = list(self.processes)[0]
            if process == "QCD":
                label = np.zeros_like(self.data[-1])
            elif process == "ttbar":
                label = np.ones_like(self.data[-1])
            elif process == "WZ":
                label = 2 * np.ones_like(self.data[-1])
        elif len(self.processes) == 2:
            label = np.copy(self.data[-1])
            label[np.isin(label, [1, 2])] = 1
        elif len(self.processes) == 4:
            label = self.data[-1]
        else:
            label = np.copy(self.data[-1])
            label -= 1
        self.data[-1] = label

    def get_sample(self):
        sample = [d[self.data_i] for d in self.data]
        self.data_i += 1
        return sample

    def get_nclasses(self):
        # TODO this is also hacky and stupid
        return 2

    def __next__(self):
        try:
            data = self.get_sample()
        except IndexError:
            self.load_data()
            data = self.get_sample()
        return data



def load_rodem_prediction_of_jetnet(
    rodem_prediction_path: str,
    dset: str,
    datasets: Mapping,
    n_jets: int = -1,
    n_csts: int = 30,
    min_n_csts: int = 0,
    recalculate_jet_from_pc: bool = False,
    score_name: str = "output",
) -> tuple:
    print(f"Loading custom labels: {score_name} from {rodem_prediction_path}")
    
    import h5py
    all_high, all_csts, all_mask, all_labels = load_jetnet(dset, datasets, n_jets, n_csts, min_n_csts, recalculate_jet_from_pc) 
    rodem_predictions = []

    for i, val in enumerate(datasets.values()):
        dataset_name = f'JetNet_{val}_train.h5'
        file_path = f'{rodem_prediction_path}/{dataset_name}'
        
        with h5py.File(file_path, "r") as f:
            print(f"Reading {score_name} from {file_path}")
            rodem_predictions.append(np.array(f[score_name][:n_jets]))

    rodem_predictions = np.concatenate(rodem_predictions).reshape(-1)

    sorted_indices = np.argsort(rodem_predictions)

    # Sort all arrays based on the sorted_indices
    all_high_rp = all_high[sorted_indices]
    all_csts_rp = all_csts[sorted_indices]
    all_mask_rp = all_mask[sorted_indices]

    # all_labels_rp = np.zeros_like(all_labels)
    # all_labels_rp[len(all_labels[all_labels==1]):] = 1



    return all_high_rp, all_csts_rp, all_mask_rp, all_labels
