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


class JIdoubledataset(Dataset):
    """A pytorch dataset object containing jet-image data."""

    def __init__(
        self,
        *,
        dset: str,
        path: Path,
        datasets: Union[dict, list],
        n_jets: int,
        n_csts: int,
        image_params: dict,
        min_n_csts: int = 0,
        leading: bool = True,
        recalculate_jet_from_pc: bool = False,
        incl_substruc: bool = False,
        boost_mopt: float = 0,
        augmentation_list: Union[str, List[str]] = "none",
        augmentation_prob: float = 1.0,
        do_plots: bool = False,
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
            leading: If the leading jet should be loaded, if False subleading is loaded
            recalculate_jet_from_pc: Redo jet eta, phi, pt, M using point cloud
            incl_substruc: If the substructure vars should be included (rodem only)
            boost_mopt: Boost the jet along its axis until m/pt = X
            augmentation_list: List of order of augmentations to apply during get item
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
        self.min_n_csts = min_n_csts
        self.leading = leading
        self.recalculate_jet_from_pc = recalculate_jet_from_pc
        self.incl_substruc = incl_substruc
        self.boost_mopt = boost_mopt
        self.augmentation_list = augmentation_list
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

    def average_image(self, num=1000, branch=0):
        """Get the average of the first num jets with a specific branch."""
        mean = np.zeros(self[0][0].shape)
        for i in range(num):
            mean += self[i][branch]
        return mean / num, num

    def plot(self) -> None:
        """Plot the average, on linear and log scales of the first 1000 jets
        for each branch and also several examples for each branch."""
        if self.do_plots:
            for branch in range(2):
                aver, num = self.average_image(branch=branch)
                aver = aver[0]
                plt.figure()
                plt.title(f"Average {branch} of {num} jets")
                plt.imshow(aver, cmap="turbo")
                plt.colorbar()
                plt.savefig(f"average{branch}.pdf")

                plt.figure()
                plt.title(f"Average {branch} of {num} jets")
                plt.imshow(np.log(aver), cmap="turbo")
                plt.colorbar()
                plt.savefig(f"log_average{branch}.pdf")

                plt.figure(figsize=(20, 4))
                plt.title(f"Examples {branch}")
                for i in range(5):
                    plt.subplot(1, 5, i + 1)
                    plt.imshow(self[i][branch][0], cmap="turbo")
                    plt.colorbar()
                plt.savefig(f"examples{branch}.pdf")

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

        # Apply all augmentations
        if self.augmentation_list and self.do_augment:
            nodes, high, mask = apply_augmentations(
                nodes, high, mask, self.augmentation_list, self.augmentation_prob
            )
        image = jetimage(
            nodes,
            bins=self.image_params["bins"],
            phi_bounds=self.image_params["phi_bounds"],
            eta_bounds=self.image_params["eta_bounds"],
            do_naive_const_preprocessing=self.image_params[
                "do_naive_const_preprocessing"
            ],
        )

        if self.image_params["image_transform1"] is not None:
            image1 = image_transformations(image, self.image_params["image_transform1"])
        else:
            image1 = image

        if self.image_params["image_transform2"] is not None:
            image2 = image_transformations(image, self.image_params["image_transform2"])
        else:
            image2 = image

        return image1, image2

    def __len__(self) -> int:
        return len(self.mask)
