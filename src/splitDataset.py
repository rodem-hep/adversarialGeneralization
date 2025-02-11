import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)


import numpy as np
import h5py
from pathlib import Path


import hydra
from omegaconf import DictConfig

from datasets_utils import DatasetManager

import logging 
log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="splitDataset.yaml"
)
def main(cfg: DictConfig) -> None:
    id_dir = {"Hbb": [4], "QCD": [1, 2, 3, 5, 6, 7, 8]}
    path = DatasetManager().get_data_path("RS3L")
    ids = id_dir[cfg.class_name]
    max_file_id = 6

    # Adjust the starting file_id and max_file_id based on the dataset type
    if cfg.dset == "test":
        file_id = max_file_id + 1
        max_file_id = 10  # or whatever the maximum file_id for the test set is
    else:
        file_id = 0

    # Adjust the output file name based on the dataset type
    output_file_name = f"FR_RS3L_{cfg.class_name}_test.h5" if cfg.dset == "test" else f"FR_RS3L_{cfg.class_name}_train.h5"

    with h5py.File(Path(path, output_file_name), "a") as f:  # the file in append mode
        total_indices = 0
        while total_indices < cfg.target_amount and file_id < max_file_id:
            log.info(f"File ID: {file_id}")
            log.info(f"Total indices: {total_indices}")
            with h5py.File(Path(path,f"rs3l_{file_id}.h5"), "r") as original_file:
                jet_types = original_file["singletons"][:-1, 0, 0]
                indices = np.where(np.isin(jet_types, ids))[0]
                if total_indices + len(indices) > cfg.target_amount:
                    indices = indices[:cfg.target_amount - total_indices]
                total_indices += len(indices)
                for name, data in original_file.items():
                    if len(indices)>0:
                        if name in f:  # Check if the dataset already exists
                            # Append the new data
                            f[name][total_indices-len(indices):total_indices] = data[indices]
                        else:
                            # Create a new dataset, chunked
                            f.create_dataset(name, data=np.empty((cfg.target_amount,) + data.shape[1:]), chunks=True)
                            f[name][total_indices-len(indices):total_indices] = data[indices]
            file_id += 1
    return

if __name__ == "__main__":
    main()
