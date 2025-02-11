import logging
from copy import deepcopy
from typing import Mapping, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mattstools.mattstools.torch_utils import train_valid_split
from src.datamodules.JIdataset import JIData

log = logging.getLogger(__name__)


class JIModulesBase(LightningDataModule):
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


class JIDataModule(JIModulesBase):
    def __init__(
        self,
        *,
        val_frac: float = 0.1,
        data_conf: Optional[Mapping] = None,
        loader_kwargs: Optional[Mapping] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load a mini dataset to infer the dimensions
        mini_conf = deepcopy(self.hparams.data_conf)
        mini_conf["n_jets"] = min(10, mini_conf["n_jets"])
        self.mini_set = JIData(dset="test", **mini_conf)
        self.dims = self.get_dims()

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""

        if stage in ["fit", "validate"]:
            dataset = JIData(dset="train", **self.hparams.data_conf)
            dataset.plot()
            self.train_set, self.valid_set = train_valid_split(
                dataset, self.hparams.val_frac
            )

            log.info(
                f"Jets Loaded: {len(self.train_set)} train, {len(self.valid_set)} valid"
            )

        if stage in ["test", "predict"]:
            test_conf = deepcopy(self.hparams.data_conf)
            test_conf["n_jets"] = -1
            self.test_set = JIData(dset="test", **test_conf)
            log.info(f"Jets Loaded: {len(self.test_set)} test")

    def get_dims(self) -> tuple:
        """Return the dimensions of the input dataset."""
        image, _ = self.mini_set[0]
        return image.shape

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the input dataset."""
        return None

    @property
    def n_classes(self) -> int:
        """Return the number of jet types/classes used in training."""
        return self.mini_set.n_classes
