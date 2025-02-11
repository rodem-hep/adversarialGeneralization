# generateHydraTaggers.py - Franck Rothen
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import logging

log = logging.getLogger(__name__)
import yaml
import hydra
from omegaconf import DictConfig
import os
# import random


def generateReferenceTaggerList(cfg: DictConfig) -> None:
    """
    This function generates a new reference tagger list based on the reference tagger list provided in the config file.
    """

    assert cfg.trainedORevaluatedOn in ["trained_on", "evaluated_on"]
    exportDict = {
        "trained_on": "trainedOnTaggerList",
        "evaluated_on": "evaluatedOnTaggerList",
    }

    with open(
        f"{cfg.config_path}/data_confs/{cfg.trainedORevaluatedOn}.yaml", "r"
    ) as file:
        data_conf = yaml.safe_load(file)

    # Create a new reference tagger list
    references = []

    # For each dataset in the data configuration, add a new tagger with modified path and name
    for dataset in data_conf:
        new_tagger = [
            {
                "name": f"{cfg.network_type}_{dataset['dataset_type']}_default",
                "path": f"{cfg.exp_path}/taggers/supervised_{dataset['dataset_type']}_default",
                "score_name": "output",
                "label": f"{dataset['dataset_type']} (default)",
                "plot_kwargs": {"linestyle": "dashed", "color": f"{dataset['color']}"},
            }
        ]
        references.extend(new_tagger)

    # create the output directory if it does not exist
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # Save the new reference
    with open(
        f"{cfg.output_dir}/{cfg.experiment_name}_{cfg.network_type}_{exportDict[cfg.trainedORevaluatedOn]}.yaml",
        "w",
    ) as file:
        yaml.dump(references, file)

    return None


def generateTaggerListForMethod(cfg: DictConfig) -> None:
    """
    This function generates a new tagger list for a specific method based on the reference tagger list provided in the config file.
    """

    # Load the reference tagger
    with open(
        f"{cfg.output_dir}/{cfg.experiment_name}_{cfg.network_type}_trainedOnTaggerList.yaml",
        "r",
    ) as file:
        reference_taggers = yaml.safe_load(file)

    # Create a new reference tagger list
    new_reference = reference_taggers.copy()

    # For each entrie in the reference tagger list, add a new tagger with modified path and name
    if cfg.method_type != "default":
        for tag in reference_taggers:
            print(tag)
            new_tagger = [
                {
                    "name": f"{tag['name'].replace('default', cfg.method_type)}",
                    "path": f"{tag['path'].replace('default', cfg.method_type)}",
                    "score_name": "output",
                    "label": f"{tag['label'].replace('default', cfg.method_type)}",
                    "plot_kwargs": {
                        "linestyle": "solid",
                        "color": tag["plot_kwargs"]["color"],
                    },
                }
            ]
            new_reference.extend(new_tagger)

            # if tag['name'].replace("dense_", "").replace("_default","")==cfg.target_dataset_type:
            #     # Replace the tagger name with (reference)
            #     tag['label'] = f"{tag['label']} (reference)"

    # Save the new reference
    # create the output directory if it does not exist
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    with open(
        f"{cfg.output_dir}/{cfg.experiment_name}_{cfg.network_type}_{cfg.method_type}_taggerList.yaml",
        "w",
    ) as file:
        yaml.dump(new_reference, file)

    return None


@hydra.main(
    version_base=None,
    config_path=str(root / "configs/generatedConfs"),
    config_name="generateHydraTaggers.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.generateReferenceTaggerList:
        generateReferenceTaggerList(cfg)

    if cfg.generateTaggerListForMethod:
        generateTaggerListForMethod(cfg)

    return None


if __name__ == "__main__":
    main()


# # List of colors
# colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray']

# @hydra.main(
#     version_base=None, config_path=str(root / "configs/generatedConfs"), config_name="generateHydraTaggers.yaml"
# )
# def main(cfg: DictConfig) -> None:
#     # Load the reference tagger
#     with open(f"{cfg.config_path}/taggers/reference.yaml", 'r') as file:
#         new_reference = yaml.safe_load(file)

#     # Define the new entries
#     used_colors = [tagger['plot_kwargs']['color'] for tagger in new_reference if 'plot_kwargs' in tagger and 'color' in tagger['plot_kwargs']]

#     for tag in cfg.taggers:
#         color = random.choice([col for col in colors if col not in used_colors])
#         used_colors.append(color)
#         new_tagger = [
#             {
#                 "name": f"{tag.name}",
#                 "path": f"{tag.path}",
#                 "score_name": "output",
#                 "Label": f"{tag.label}",
#                 "plot_kwargs": {
#                     "linestyle": "solid",
#                     "color": color
#                 }
#             }
#         ]

#         new_reference.extend(new_tagger)


#     # Save the new reference
#     with open(f"{cfg.output_dir}/ROC_taggers.yaml", 'w') as file:
#         yaml.dump(new_reference, file)


# if __name__ == "__main__":
#     main()
