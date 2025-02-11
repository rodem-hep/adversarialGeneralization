# generateSweepConfigs.py - Franck Rothen
# automatically generate sweep experiment configs for hydra based on snakemake and hydra hyperparameter list

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import hydra
from omegaconf import DictConfig
import yaml
import os
from ruamel.yaml import YAML



@hydra.main(
    version_base=None, config_path=str(root / "configs/generatedConfs"), config_name="generateSweepConfigs.yaml"
)
def main(cfg: DictConfig) -> None:
    # create a YAML object
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Load the reference experiment yaml
    with open(f"{cfg.config_path}/experiment/{cfg.reference_experiment_name}.yaml", 'r') as file:
        reference_experiment = yaml.load(file)

    # Load the hyperparameter combinations list
    with open(f"{cfg.config_path}/experiment/sweeps_FR/{cfg.method_type}.yaml", 'r') as file:
        hyperparameter_combinations = yaml.load(file)

    # For each hyperparameter combination, create a new experiment
    for i, hyperparameter_combination in enumerate(hyperparameter_combinations):
        # For each hyperparameter, update the experiment
        for key, value in hyperparameter_combination.items():
            reference_experiment[key] = value

        # Save the new experiment
        # create the output directory if it does not exist
        if not os.path.exists(f"{cfg.config_path}/experiment/sweeps_FR/generated"):
            os.makedirs(f"{cfg.config_path}/experiment/sweeps_FR/generated")
        with open(f"{cfg.config_path}/experiment/sweeps_FR/generated/{cfg.reference_experiment_name}_{i}.yaml", 'w') as file:
            yaml.dump(reference_experiment, file)

    # Generate tagger list
    # List of colors
    colors = ['green', 'red', 'purple', 'orange', 'pink', 'brown', 'blue', 'yellow', 'gray', 'black']
    tagger_list = []
    for i in range(len(hyperparameter_combinations)):
        color = colors.pop(0)
        tagger = [
            {
                "name": f"dense_{cfg.trained_on_dataset_type}_{cfg.method_type}_{i}",
                "path": f"{cfg.exp_path}/taggers_sweep/supervised_{cfg.trained_on_dataset_type}_{cfg.method_type}",
                "score_name": "output",
                "label": f"dense {cfg.method_type} trained on {cfg.trained_on_dataset_type} {i}",
                "plot_kwargs": {
                    "linestyle": "solid",
                    "color": color
                },
                "id": i,
            }
        ]
        tagger_list.extend(tagger)

    if not os.path.exists(f"{cfg.config_path}/taggers/generated"):
        os.makedirs(f"{cfg.config_path}/taggers/generated")
    with open(f"{cfg.config_path}/taggers/generated/taggers_sweep_{cfg.method_type}.yaml", 'w') as file:
        yaml.dump(tagger_list, file)
    
            
if __name__ == "__main__":
    main()