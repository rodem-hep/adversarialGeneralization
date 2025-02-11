import logging
from copy import deepcopy
from typing import Mapping, Optional

from mattstools.mattstools.torch_utils import train_valid_split
from src.datamodules.JImodule import JIModulesBase
from src.datamodules.JIxPCdataset import JIxPCdataset

log = logging.getLogger(__name__)


class JIxPCmodule(JIModulesBase):
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
        self.mini_set = JIxPCdataset(dset="train", **mini_conf)
        self.dims = self.get_dims()

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""

        if stage in ["fit", "validate"]:
            dataset = JIxPCdataset(dset="train", **self.hparams.data_conf)
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
            self.test_set = JIxPCdataset(dset="test", **test_conf)

            log.info(f"Jets Loaded: {len(self.test_set)} test")

    def get_dims(self) -> tuple:
        """Return the dimensions of the input dataset."""
        image, edges, nodes, high, adjmat, mask = self.mini_set[0]
        return (
            image.shape,
            edges.shape[-1],
            nodes.shape[-1],
            high.shape[0],
            len(mask),
        )

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the input dataset."""
        return self.mini_set[0][1].shape[-2]

    @property
    def n_classes(self) -> int:
        """Return the number of jet types/classes used in training."""
        return self.mini_set.n_classes
