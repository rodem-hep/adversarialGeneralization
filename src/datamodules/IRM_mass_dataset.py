"""Pytorch Dataset definitions of various collections training samples."""

from copy import deepcopy
from pathlib import Path
from typing import List, Union

import logging

import numpy as np
from torch.utils.data import Dataset

from mattstools.mattstools.plotting import plot_multi_hists_2
from src.augmentations import apply_augmentations
from src.datamodules.loading import load_data
from src.jet_utils import boost_jet_mopt, build_jet_edges, graph_coordinates
from franckstools.franckstools.utils import gaussian_dist

log = logging.getLogger(__name__)


class JetData(Dataset):
    """A pytorch dataset object containing high and low level jet
    information."""

    def __init__(
        self,
        *,
        dset: str,
        path: Path,
        dataset_type: str,
        datasets: Union[dict, list],
        n_jets: int,
        n_csts: int,
        coordinates: dict,
        min_n_csts: int = 0,
        leading: bool = True,
        recalculate_jet_from_pc: bool = False,
        incl_substruc: bool = False,
        del_r_edges: float = 0,
        boost_mopt: float = 0,
        augmentation_list: Union[str, List[str]] = "none",
        augmentation_prob: float = 1.0,
        rodem_predictions_path: str = None,
        score_name: str = "output",
    ) -> None:
        """
        args:
            dset: Either train or test
            path: Path to find the jet datasets, must contain either rodem or toptag
            datasets: Which physics processes will be loaded with which labels
            n_jets: How many jets to load in the entire dataset
            n_csts: The number of constituents to load per jet
            coordinates: Dict of keys for which features to use in the graph
            min_n_csts: The minimum number of constituents in each jet
                - This filter is applied after data is loaded from file so it may
                result in less jets being returned than specified.
            leading: If the leading jet should be loaded, if False subleading is loaded
            recalculate_jet_from_pc: Redo jet eta, phi, pt, M using point cloud
            incl_substruc: If the substructure vars should be included (rodem only)
            del_r_edges: Build and attribute graph edges using the delta R of the nodes
            boost_mopt: Boost the jet along its axis until m/pt = X
            augmentation_list: List of order of augmentations to apply during get item
            augmentation_prob: Probability of each aug in list taking effect
        """

        # Check arguments
        if dataset_type == "TopTag":
            if incl_substruc:
                raise ValueError("Can't have substructure variables in toptag data")
            if not leading:
                raise ValueError("Toptag does not have subleading jet information")
        if boost_mopt and "boost" in augmentation_list:
            raise ValueError("Can not use boosting as preprocessing and augmentation!")
        if boost_mopt == -1:  # When boosting into the reference frame of the jet
            if any(tst in sr for tst in ["pt", "del"] for sr in coordinates["node"]):
                raise ValueError("Should only use xyz when boosting into jet frame!")

        # Check if the augmentation list is a string, and split using commas
        if isinstance(augmentation_list, str):
            if augmentation_list == "none":
                augmentation_list = []
            elif augmentation_list == "all":
                augmentation_list = [
                    "rotate",
                    "crop-10",
                    "merge-0.05",
                    "split-10",
                    "smear",
                    "boost-0.05",
                ]
            else:
                augmentation_list = [x for x in augmentation_list.split(",") if x]
        augmentation_list = augmentation_list.copy() or []

        # Class attributes
        self.path = path
        self.dset = dset
        self.n_nodes = n_csts
        self.datasets = datasets.copy()
        self.coordinates = deepcopy(coordinates)
        self.min_n_csts = min_n_csts
        self.leading = leading
        self.recalculate_jet_from_pc = recalculate_jet_from_pc
        self.incl_substruc = incl_substruc
        self.del_r_edges = del_r_edges
        self.boost_mopt = boost_mopt
        self.augmentation_list = augmentation_list
        self.augmentation_prob = augmentation_prob
        self.do_augment = bool(augmentation_list)
        
        # Load jets and constituents as pt, eta, phi, (M for jets)
        self.high_data, self.node_data, self.mask, self.labels = load_data(
            dataset_type,  
            dset,
            path,
            datasets,
            n_jets,
            n_csts,
            min_n_csts,
            incl_substruc,
            leading,
            recalculate_jet_from_pc,
            rodem_predictions_path,
            score_name=score_name,
        )

        self.n_classes = len(datasets)

        # Check for Nan's (happens sometimes...)
        if np.isnan(self.high_data).any():
            raise ValueError("Detected NaNs in the jet data!")
        if np.isnan(self.node_data).any():
            raise ValueError("Detected NaNs in the constituent data!")

    def plot(self, max_events: int = 10_000) -> None:
        """Plot the collection of inputs
        Args:
            max_events: Max number of events to plot
        """
        # Create empty lists to hold the data used to fit the scalers
        num_csts = []
        all_data = {"edge": [], "node": [], "high": []}

        # Cycle through the datasets, keeping only the data which is requested!
        take_every = int(len(self) / min(len(self), max_events))
        for i in np.arange(0, len(self), take_every):
            edges, nodes, high, adjmat, mask, label = self[i]
            num_csts.append(np.sum(mask))
            if np.any(edges):
                all_data["edge"].append(edges[adjmat])
            all_data["node"].append(nodes[mask])
            all_data["high"].append(high)

        # Create the plotting folder
        plot_path = Path("train_dist")
        plot_path.mkdir(parents=True, exist_ok=True)

        # Make a histogram of the multiplicities
        plot_multi_hists_2(
            data_list=np.expand_dims(np.array(num_csts), 1),
            data_labels="inclusive",
            col_labels="num_csts",
            path=plot_path / "num_csts",
            bins=np.arange(0, self.n_nodes + 2),
        )

        # For each data type: combine -> plot
        for key in all_data.keys():
            # combine if there is any data to combine
            try:
                all_data[key] = np.vstack(all_data[key])
            except (KeyError, ValueError):
                continue

            # plot
            if all_data[key].size:
                # The labels for the plots come from the coords, but if substruc we
                # need to manually add more
                coords = self.coordinates[key].copy()
                if self.incl_substruc and key == "high":
                    coords += ["tau1", "tau2", "tau3", "d12", "d23", "ECF2", "ECF3"]
                plot_multi_hists_2(
                    data_list=all_data[key],
                    data_labels=key,
                    col_labels=coords,
                    path=plot_path / key,
                )

    def __getitem__(self, idx: int) -> tuple:
        """Retrives a jet from the dataset and returns it as a graph object
        along with the class label.

        Args:
            idx: The index of the jet to pull from the dataset
        """

        # Load the particle constituent and high level jet information from the data
        nodes = self.node_data[idx].copy()
        mask = self.mask[idx].copy()
        high = self.high_data[idx].copy()
        label = self.labels[idx].copy()

        # Apply all augmentations
        if self.augmentation_list and self.do_augment:
            nodes, high, mask = apply_augmentations(
                nodes, high, mask, self.augmentation_list, self.augmentation_prob
            )

        # Build jet edges (will return empty if del_r is set to 0)
        # Edges are also compressed to save memory
        edges, adjmat = build_jet_edges(
            nodes, mask, self.coordinates["edge"], self.del_r_edges
        )

        # Apply boost pre-processing after the jet edges
        if self.boost_mopt != 0:
            nodes, high = boost_jet_mopt(nodes, high, self.boost_mopt)

        # Convert to the specified selection of local variables and extract edges
        nodes, high = graph_coordinates(nodes, high, mask, self.coordinates)

        return edges, nodes, high, adjmat, mask, label

    def __len__(self) -> int:
        return len(self.mask)
    
    # def transform_to_env(self, mass_bins, mus, sigmas, n_target_samples):
    #     # Create a new environment by sampling from the mass distribution
    #     new_env_ids = []

    #     # Calculate the bin indices for each mass
    #     bin_indices = np.digitize(self.masses, mass_bins) - 1

    #     # Keep track of the number of samples in each class
    #     n_class_samples = np.zeros(self.n_classes, dtype=int)

    #     # Iterate over each class
    #     for class_idx in range(self.n_classes):

    #         # Calculate the bin occupancy for the current class
    #         bin_occupancy = gaussian_dist(mass_bins, mus[class_idx], sigmas[class_idx])
    #         bin_occupancy = bin_occupancy / np.sum(bin_occupancy)  # normalize to 1

    #         # Find the indices of the data that fall into each bin for the current class
    #         class_indices = np.where(self.labels == class_idx)[0]
    #         bin_indices_class = bin_indices[class_indices]

    #         # Iterate over each bin
    #         for bin_idx in range(len(mass_bins) - 1):
    #             # Find the indices of the data that fall into the current bin
    #             bin_indices_bin = np.where(bin_indices_class == bin_idx)[0]

    #             # Check if the bin has enough samples
    #             if len(bin_indices_bin) <= n_total_samples * bin_occupancy[bin_idx]:
    #                 log.warning(f"Bin {bin_idx} has fewer samples than expected. Appending all samples to the environment.")
    #                 new_env_ids.extend(train_set.indices[class_indices[bin_indices_bin]])
    #                 n_class_samples[class_idx] += len(bin_indices_bin)
    #             else:
    #                 # Randomly select the samples to append to the environment
    #                 selected_indices = np.random.choice(bin_indices_bin, size=int(n_total_samples * bin_occupancy[bin_idx]), replace=False)
    #                 new_env_ids.extend(train_set.indices[class_indices[selected_indices]])
    #                 n_class_samples[class_idx] += len(selected_indices)

    #     # Create a new Subset object with the selected indices
    #     new_env_train_set = torch.utils.data.dataset.Subset(train_set.dataset, new_env_ids)

    #     # Log the number of samples in each class
    #     for class_idx in range(self.n_classes):
    #         log.info(f"Class {class_idx} total amount of samples: {n_class_samples[class_idx]}")
    #     return new_env_train_set

