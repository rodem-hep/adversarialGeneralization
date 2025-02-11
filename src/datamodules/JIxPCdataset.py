"""Pytorch Dataset definitions of various collections training samples."""

from copy import deepcopy
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from src.augmentations import apply_augmentations
from src.datamodules.jetimage import jetimage
from src.datamodules.JItransformations import image_transformations
from src.datamodules.loading import load_rodem, load_toptag
from src.jet_utils import boost_jet_mopt, build_jet_edges, graph_coordinates


class JIxPCdataset(Dataset):
    """A pytorch dataset object containing jet-image data."""

    def __init__(
        self,
        *,
        dset: str,
        path: Path,
        datasets: Union[dict, list],
        n_jets: int,
        n_csts: int,
        coordinates: dict,
        image_params: dict,
        min_n_csts: int = 0,
        leading: bool = True,
        recalculate_jet_from_pc: bool = False,
        incl_substruc: bool = False,
        boost_mopt: float = 0,
        augmentation_list: Union[str, List[str]] = "none",
        augmentation_list_image: Union[str, List[str]] = "none",
        augmentation_prob: float = 1.0,
        do_plots: bool = False,
        del_r_edges: float = 0,
    ) -> None:
        """
        Args:
            dset: Either train or test
            path: Path to find the jet datasets, must contain either rodem or toptag
            datasets: Which physics processes will be loaded with which labels
            n_jets: How many jets to load in the entire dataset
            n_csts: The number of constituents to load per jet
            min_n_csts: The minimum number of constituents in each jet
                - This filter is applied after data is loaded from file so it may
                result in less jets being returned than specified.
            coordinates: Dict of keys for which features to use in the graph
            leading: If the leading jet should be loaded, if False subleading is loaded
            recalculate_jet_from_pc: Redo jet eta, phi, pt, M using point cloud
            incl_substruc: If the substructure vars should be included (rodem only)
            boost_mopt: Boost the jet along its axis until m/pt = X
            augmentation_list: List of order of augmentations to apply during get item
            augmentation_list_image:  List of order of augmentations to apply during
            get item for JI data if augmentation_list_image is "same"
            then augmentation_list is used
            del_r_edges: Build and attribute graph edges using the delta R of the nodes
            augmentation_prob: Probability of each aug in list taking effect
            do_plots: If the dataset should plot the jet images
        """

        # Check arguments
        if "toptag" in path:
            if incl_substruc:
                raise ValueError("Can't have substructure variables in toptag data")
            if not leading:
                raise ValueError("Toptag does not have subleading jet information")
        if boost_mopt and "boost" in augmentation_list:
            raise ValueError("Can not use boosting as preprocessing and augmentation!")

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
        self.image_params = deepcopy(image_params)
        self.coordinates = coordinates.copy()
        self.del_r_edges = del_r_edges
        self.min_n_csts = min_n_csts
        self.leading = leading
        self.recalculate_jet_from_pc = recalculate_jet_from_pc
        self.incl_substruc = incl_substruc
        self.boost_mopt = boost_mopt
        self.augmentation_list = augmentation_list
        self.augmentation_list_image = augmentation_list_image
        self.augmentation_prob = augmentation_prob
        self.do_augment = bool(augmentation_list)
        self.do_plots = do_plots

        # Load jets and constituents as pt, eta, phi, (M for jets)
        if "rodem" in path:
            self.high_data, self.node_data, self.mask, self.labels = load_rodem(
                dset,
                path,
                datasets,
                n_jets,
                n_csts,
                min_n_csts,
                incl_substruc,
                leading,
                recalculate_jet_from_pc,
            )
        elif "toptag" in path:
            self.high_data, self.node_data, self.mask, self.labels = load_toptag(
                dset, path, datasets, n_jets, n_csts, min_n_csts
            )
        self.n_classes = len(datasets)

        # Check for Nan's (happens sometimes...)
        if np.isnan(self.high_data).any():
            raise ValueError("Detected NaNs in the jet data!")
        if np.isnan(self.node_data).any():
            raise ValueError("Detected NaNs in the constituent data!")

    def average_image(self, num=1000):
        """Get the average of the first num jets with a specific branch."""
        mean = np.zeros(self[0][0].shape)
        for i in range(num):
            mean += self[i][0]
        return mean / num, num

    def plot(self) -> None:
        """Plot the average, on linear and log scales of the first 1000 jets
        for each branch and also several examples for each branch."""
        if self.do_plots:
            aver, num = self.average_image()
            aver = aver[0]
            plt.figure()
            plt.title(f"Average of {num} jets")
            plt.imshow(aver, cmap="turbo")
            plt.colorbar()
            plt.savefig("average.pdf")

            plt.figure()
            plt.title(f"Average of {num} jets")
            plt.imshow(np.log(aver), cmap="turbo")
            plt.colorbar()
            plt.savefig("log_average.pdf")

            plt.figure(figsize=(20, 4))
            plt.title("Examples")
            for i in range(5):
                plt.subplot(1, 5, i + 1)
                plt.imshow(self[i][0][0], cmap="turbo")
                plt.colorbar()
            plt.savefig("examples.pdf")

    def __getitem__(self, idx: int) -> tuple:
        """Retrives a jet from the dataset and returns it as a doublet of
        jetimages (intended to use with different transforms/augmentations)

        Args:
            idx: The index of the jet to pull from the dataset
        """

        # Load the particle constituent and high level jet information from the data
        nodes = self.node_data[idx].copy()
        mask = self.mask[idx].copy()
        high = self.high_data[idx].copy()

        # ////////// Augmentations //////////
        # Apply all augmentations for the PC
        if self.augmentation_list and self.do_augment:
            nodes, high, mask = apply_augmentations(
                nodes, high, mask, self.augmentation_list, self.augmentation_prob
            )

        if self.augmentation_list_image == "same":
            nodes_im = nodes
            mask_im = mask
            high_im = high
        else:
            if self.augmentation_list_image and self.do_augment:
                nodes_im = self.node_data[idx].copy()
                mask_im = self.mask[idx].copy()
                high_im = self.high_data[idx].copy()
                nodes, high, mask = apply_augmentations(
                    nodes_im,
                    high_im,
                    mask_im,
                    self.augmentation_list_image,
                    self.augmentation_prob,
                )
        # ////////////////////////////////////////
        # ////////// Jet Image ///////////////////
        # produce a jet image
        image = jetimage(
            nodes_im,
            bins=self.image_params["bins"],
            phi_bounds=self.image_params["phi_bounds"],
            eta_bounds=self.image_params["eta_bounds"],
            do_naive_const_preprocessing=self.image_params[
                "do_naive_const_preprocessing"
            ],
        )

        # Apply all augmentations for the jet image
        if self.image_params["image_transform"] is not None:
            image = image_transformations(image, self.image_params["image_transform"])
        # ////////////////////////////////////////
        # ////////// Particle cloud //////////////

        # Build jet edges (will return empty if del_r is set to 0)
        # Edges are also compressed to save memory
        edges, adjmat = build_jet_edges(
            nodes, mask, self.coordinates["edge"], self.del_r_edges
        )

        # Apply boost pre-processing after the jet edges
        # TODO: This should be done before the jet images sometimes
        # so that the naive preprocessing is avoided to speed up things
        if self.boost_mopt != 0:
            nodes, high = boost_jet_mopt(nodes, high, self.boost_mopt)

        # Convert to the specified selection of local variables and extract edges
        nodes, high = graph_coordinates(nodes, high, mask, self.coordinates)

        return image, edges, nodes, high, adjmat, mask

    def __len__(self) -> int:
        return len(self.mask)
