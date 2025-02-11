import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
log = logging.getLogger(__name__)

import numpy as np
import h5py
from src.datasets_utils import DatasetManager


@hydra.main(config_path=str(root / "franckstools/franckstools"), config_name="test.yaml", version_base=None)
def main(cfg: DictConfig):
    print("hello")
    path = "/home/users/r/rothenf3/scratch/data/generalization/rs3l_0.h5"

    _ = load_rs3l("RS3L")
    # with h5py.File(path, "r") as f:
    #     print(f.keys())
    #     f['singletons']
    #     f['jet_pflow_cands']

    #     # Count the number of events with jettype
    #     jettypeList = f['singletons'][:,0,0]
        
    #     # count occurrences
    #     counts = np.bincount(jettypeList.astype(int))

    #     print(counts)



def load_rs3l(
        dset: str,
        n_jets: int = 300000,
        n_csts: int = 40,
        augmentation_id: int = 0,
        max_file_id: int = 2,
        jet_ids: list = [9, 10, 11, 12], # pt, eta, phi, energy
        csts_ids: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #['pt','relpt','eta','phi','dr','e','rele','charge','pdgid','d0','dz']
        signal_ids: list = [4], # H->bb
        background_ids: list = [1, 2, 3, 5, 6, 7, 8], # light, c, b, g->qq, g->cc, g->bb, g->gg
) -> tuple:
    """
    - Franck Rothen, 2024
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


    print(f"Loading {n_jets} n_jets and {n_csts} n_csts from the {dset} RS3L dataset")
    # print(f" -- processes selected: {datasets}")

    # Get the path to the dataset
    path = DatasetManager().get_data_path("RS3L")

    if dset == "train":
        print("Loading training data")

        # File index
        file_id = 0

        # Containers
        initialized = False

        # Counters
        event_count = 0


        while (event_count < n_jets or event_count < n_jets) and file_id < max_file_id:
            # Load the data
            with h5py.File(path.append(f"rs3l_{file_id}.h5"), "r") as f:
                # Get jet and constituents data
                jet_file_data = f["singletons"][:-1, augmentation_id, [0] + jet_ids]
                csts_file_data = f["jet_pflow_cands"][:-1, augmentation_id, csts_ids, :n_csts]

                # Remove unrequested jet types and assign labels
                file_labels = np.ones(jet_file_data.shape[0])
                file_labels = -file_labels.astype(int) # labels = -1 will be used to mask unwanted jets type

                file_labels = np.where(np.isin(jet_file_data[:, 0], signal_ids), 1, file_labels) # signal
                file_labels = np.where(np.isin(jet_file_data[:, 0], background_ids), 0, file_labels) # background

                jet_file_data = jet_file_data[file_labels != -1]
                csts_file_data = csts_file_data[file_labels != -1]
                file_labels = file_labels[file_labels != -1]

                # Remove jet type column
                jet_file_data = jet_file_data[:, 1:]

                # Reorder the shape of the constituents data
                csts_file_data = csts_file_data.transpose(0, 2, 1) # (n_jets, n_csts, n_features)

                # Create mask for the jet constituents (based on pT)
                file_mask = csts_file_data[..., 0] > 0

                # Remove excess from dominant class
                if len(file_labels[file_labels == 0]) > len(file_labels[file_labels == 1]):
                    labels_max_id = 1
                else:
                    labels_max_id = 0
                
                file_event_count = min(len(file_labels[file_labels == labels_max_id]), n_jets - event_count)



                # Append the data to the containers
                jet_data = jet_file_data[file_labels==0][:file_event_count] if initialized == False else np.concatenate([jet_data, jet_file_data[file_labels==0][:file_event_count]])
                jet_data = np.concatenate([jet_data, jet_file_data[file_labels==1][:file_event_count]])

                csts_data = csts_file_data[file_labels==0][:file_event_count] if initialized == False else np.concatenate([csts_data, csts_file_data[file_labels==0][:file_event_count]])
                csts_data = np.concatenate([csts_data, csts_file_data[file_labels==1][:file_event_count]])

                mask = file_mask[file_labels==0][:file_event_count] if initialized == False else np.concatenate([mask, file_mask[file_labels==0][:file_event_count]])
                mask = np.concatenate([mask, file_mask[file_labels==1][:file_event_count]])

                labels = file_labels[file_labels==0][:file_event_count] if initialized == False else np.concatenate([labels, file_labels[file_labels==0][:file_event_count]])
                labels = np.concatenate([labels, file_labels[file_labels==1][:file_event_count]])

                initialized = True

                # Update the counters
                event_count += file_event_count
                file_id += 1
        
        if event_count < n_jets:
            print(f"WARNING: Only {event_count} jets were found in the dataset with max_file_id = {max_file_id}. The requested number of jets was not reached.")

        # Shuffle the data
        shuffling = np.random.permutation(2*event_count)
        jet_data = jet_data[shuffling]
        csts_data = csts_data[shuffling]
        mask = mask[shuffling]
        labels = labels[shuffling]
    
    elif dset == "test":
        print("Loading test data")

        # Load the data
        with h5py.File(path.append(f"rs3l_{max_file_id+1}.h5"), "r") as f:
            # Get jet and constituents data
            jet_file_data = f["singletons"][:-1, augmentation_id, [0] + jet_ids]
            csts_file_data = f["jet_pflow_cands"][:-1, augmentation_id, csts_ids, :n_csts]

            # Remove unrequested jet types and assign labels
            file_labels = np.ones(jet_file_data.shape[0])
            file_labels = -file_labels.astype(int) # labels = -1 will be used to mask unwanted jets type

            file_labels = np.where(np.isin(jet_file_data[:, 0], signal_ids), 1, file_labels) # signal
            file_labels = np.where(np.isin(jet_file_data[:, 0], background_ids), 0, file_labels) # background

            jet_file_data = jet_file_data[file_labels != -1]
            csts_file_data = csts_file_data[file_labels != -1]
            file_labels = file_labels[file_labels != -1]

            # Remove jet type column
            jet_file_data = jet_file_data[:, 1:]

            # Reorder the shape of the constituents data
            csts_file_data = csts_file_data.transpose(0, 2, 1) # (n_jets, n_csts, n_features)

            # Create mask for the jet constituents (based on pT)
            file_mask = csts_file_data[..., 0] > 0

            # Remove excess from dominant class
            if len(file_labels[file_labels == 0]) > len(file_labels[file_labels == 1]):
                event_count = len(file_labels[file_labels == 1])
            else:
                event_count = len(file_labels[file_labels == 0])

            jet_data = np.concatenate([jet_file_data[file_labels==0][:event_count], jet_file_data[file_labels==1][:event_count]])
            csts_data = np.concatenate([csts_file_data[file_labels==0][:event_count], csts_file_data[file_labels==1][:event_count]])
            mask = np.concatenate([file_mask[file_labels==0][:event_count], file_mask[file_labels==1][:event_count]])
            labels = np.concatenate([file_labels[file_labels==0][:event_count], file_labels[file_labels==1][:event_count]])
            


    return jet_data, csts_data, mask, labels

if __name__ == "__main__":
    main()