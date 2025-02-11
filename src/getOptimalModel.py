# getOptimalModel.py - Franck Rothen
# automatically compare results of snakemake sweeps, report the final best hyperparameter combination and produces various plots (ROC, ParallelPlot)

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import hydra
from omegaconf import DictConfig
import yaml
import os

from src.comparison_plotting import save_roc_curves
from src.datasets_utils import DatasetManager

@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="getOptimalConfigs.yaml"
)
def main(cfg: DictConfig) -> None:
    # Plot the ROC curve for each tagger for each dataset and get best_combination_id
    best_aucs = {}
    best_combination_ids = {}

    for data_conf in cfg.data_confs:
        cfg.roc_plots_config.dataset_type = data_conf.dataset_type
        cfg.roc_plots_config.files = [DatasetManager().get_output_file_name(data_conf.dataset_type, data_conf.datasets.c0), DatasetManager().get_output_file_name(data_conf.dataset_type, data_conf.datasets.c1)]
        best_aucs[data_conf.dataset_type], best_combination_ids[data_conf.dataset_type] = save_roc_curves(**cfg.roc_plots_config)


    # Export experiment file of the best hyperparameter combination
    best_combination_id = best_combination_ids[cfg.target_dataset_type]
    experiment_file_path = f"{cfg.config_path}/experiment/sweeps_FR/generated/{cfg.reference_experiment_name}_{best_combination_id}.yaml"
    destination_file_path = f"{cfg.output_dir}/optimal_hyperparameters.yaml"
    
    # Copy the best experiment file to the taggers folder
    print(f"Copying best experiment file to the taggers folder: {destination_file_path}")
    with open(experiment_file_path, 'rb') as source_file, open(destination_file_path, 'wb') as destination_file:
        destination_file.write(source_file.read())

if __name__ == "__main__":
    main()