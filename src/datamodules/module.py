import logging
from copy import deepcopy
from typing import Mapping, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mattstools.mattstools.torch_utils import train_valid_split
from src.datamodules.dataset import JetData

log = logging.getLogger(__name__)

import vector


class PointCloudDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        val_frac: float = 0.1,
        data_conf: Optional[Mapping] = None,
        loader_kwargs: Optional[Mapping] = None,
        predict_n_test: int = 200000,
        # predict_n_test: int = -1,
        export_train: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load a mini dataset to infer the dimensions
        mini_conf = deepcopy(self.hparams.data_conf)
        mini_conf["n_jets"] = 5
        self.mini_set = JetData(dset="test", **mini_conf)
        self.inpt_dim = self.get_dims()

        self.predict_n_test = predict_n_test

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets depending on the stage of
        training/eval."""

        if stage in ["fit", "validate"]:
            dataset = JetData(dset="train", **self.hparams.data_conf)
            dataset.plot()
            self.train_set, self.valid_set = train_valid_split(
                dataset, self.hparams.val_frac
            )
            log.info(
                f"Loaded: {len(self.train_set)} train, {len(self.valid_set)} valid"
            )

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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_kwargs, shuffle=True)

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
