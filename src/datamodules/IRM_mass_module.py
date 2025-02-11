import numpy as np
import logging
from copy import deepcopy
from typing import Mapping, Optional

import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mattstools.mattstools.torch_utils import train_valid_split
from src.datamodules.dataset import JetData
from scripts.plotting_FR import plot_env_mass_distributions
from franckstools.franckstools.utils import gaussian_dist

log = logging.getLogger(__name__)


class PointCloudDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        val_frac: float = 0.1,
        data_conf: Optional[Mapping] = None,
        loader_kwargs: Optional[Mapping] = None,
        predict_n_test: int = -1,
        export_train: bool = False,
        n_environments: int = 3,
        mus_list: list = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        sigmas_list: list = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        add_original_dist: bool = False,
        n_bins: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load a mini dataset to infer the dimensions
        mini_conf = deepcopy(self.hparams.data_conf)
        mini_conf["n_jets"] = 5
        self.mini_set = JetData(dset="test", **mini_conf)
        self.inpt_dim = self.get_dims()

        self.predict_n_test = predict_n_test

        # check if the length of the lists are the same
        assert len(mus_list) == n_environments or len(mus_list) == n_environments-1
        assert len(mus_list) == len(sigmas_list)

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets depending on the stage of
        training/eval."""

        if stage in ["fit", "validate"]:
            dataset = JetData(dset="train", **self.hparams.data_conf)
            dataset.plot()
            train_set, self.valid_set = train_valid_split(
                dataset, self.hparams.val_frac
            )

            log.info(f"Loaded: {len(train_set)} train, {len(self.valid_set)} valid")

            log.info(f"Creating {self.hparams.n_environments} sub environments for training")
            self.env_train_set = []
            mass_bins = np.linspace(15, 250, self.hparams.n_bins) #TODO: Find better way to define the mass bins
            for i in range(len(self.hparams.mus_list)):
                self.env_train_set.append(self.create_environment(train_set, mass_bins, self.hparams.mus_list[i], self.hparams.sigmas_list[i]))
                log.info(f"Loaded: {len(self.env_train_set[i])} train in environment {i}")

            if self.hparams.add_original_dist:
                self.env_train_set.append(train_set)
                log.info(f"Loaded: {len(train_set)} train in original environment")

            # Plot the mass distribution of the environment
            log.info(f"Plotting mass distribution of environments")
            plot_env_mass_distributions(self.env_train_set, self.hparams.n_bins, self.hparams.mus_list, self.hparams.sigmas_list)


        if stage in ["test", "predict"]:
            test_conf = deepcopy(self.hparams.data_conf)
            test_conf["n_jets"] = self.predict_n_test
            test_conf["min_n_csts"] = 0
            test_conf["leading"] = True
            if hasattr(self.hparams, 'export_train') and self.hparams['export_train'] == True:
                self.test_set = JetData(dset="train", **test_conf)
            else:
                self.test_set = JetData(dset="test", **test_conf)
            log.info(f"Loaded: {len(self.test_set)} test")

    def create_environment(self, train_set, mass_bins, mus, sigmas):
        # Create a new environment by sampling from the mass distribution
        new_env_ids = []

        # Define a total amount of samples
        n_total_samples = len(train_set) // (5 * self.hparams.n_environments)  # There is no logic behind this. We just to minimize the number of bins "underflowing". We are losing lots of data here TODO: find a better way to do this
        
        # Get the masses and labels of all events in the training set
        masses = train_set.dataset.get_masses()[train_set.indices]
        labels = train_set.dataset.get_labels()[train_set.indices]

        # Calculate the bin indices for each mass
        bin_indices = np.digitize(masses, mass_bins) - 1

        # Keep track of the number of samples in each class
        n_class_samples = np.zeros(self.n_classes, dtype=int)

        # Iterate over each class
        for class_idx in range(self.n_classes):

            # Calculate the bin occupancy for the current class
            bin_occupancy = gaussian_dist(mass_bins, mus[class_idx], sigmas[class_idx])
            bin_occupancy = bin_occupancy / np.sum(bin_occupancy)  # normalize to 1

            # Find the indices of the data that fall into each bin for the current class
            class_indices = np.where(labels == class_idx)[0]
            bin_indices_class = bin_indices[class_indices]

            # Iterate over each bin
            for bin_idx in range(len(mass_bins) - 1):
                # Find the indices of the data that fall into the current bin
                bin_indices_bin = np.where(bin_indices_class == bin_idx)[0]

                # Check if the bin has enough samples
                if len(bin_indices_bin) <= n_total_samples * bin_occupancy[bin_idx]:
                    log.warning(f"Bin {bin_idx} has fewer samples than expected. Appending all samples to the environment.")
                    new_env_ids.extend(train_set.indices[class_indices[bin_indices_bin]])
                    n_class_samples[class_idx] += len(bin_indices_bin)
                else:
                    # Randomly select the samples to append to the environment
                    selected_indices = np.random.choice(bin_indices_bin, size=int(n_total_samples * bin_occupancy[bin_idx]), replace=False)
                    new_env_ids.extend(train_set.indices[class_indices[selected_indices]])
                    n_class_samples[class_idx] += len(selected_indices)

        # Create a new Subset object with the selected indices
        new_env_train_set = torch.utils.data.dataset.Subset(train_set.dataset, new_env_ids)

        # Log the number of samples in each class
        for class_idx in range(self.n_classes):
            log.info(f"Class {class_idx} total amount of samples: {n_class_samples[class_idx]}")
        return new_env_train_set

    def train_dataloader(self) -> DataLoader:
        # Combine the dataloaders of the different environments
        listed_dataloaders = [DataLoader(self.env_train_set[i], **self.hparams.loader_kwargs, shuffle=True) for i in range(self.hparams.n_environments)]
        return listed_dataloaders

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_set, **self.hparams.loader_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        test_kwargs = deepcopy(self.hparams.loader_kwargs)
        test_kwargs["drop_last"] = False
        return DataLoader(self.test_set, **test_kwargs, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_dims(self) -> tuple:
        """Return the dimensions of the input dataset."""
        edges, nodes, high, adjmat, mask, label = self.mini_set[0]
        return edges.shape[-1], nodes.shape[-1], high.shape[0]

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the input dataset."""
        return self.mini_set[0][1].shape[-2]

    @property
    def n_classes(self) -> int:
        """Return the number of jet types/classes used in training."""
        return self.mini_set.n_classes
