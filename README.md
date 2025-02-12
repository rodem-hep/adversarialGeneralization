<div align="center">

# Enhancing generalization in high energy physics using white-box adversarial attacks

[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB_0.14-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
![](./nuflows.png)
</div>

This repository contains the code necessary to reproduce the results described in the associated paper:

- Associated paper: <https://arxiv.org/abs/2411.09296>

## Setup

1) Setup the environment.
    - This project was tested with python 3.11.5
    - You can use the requirement.txt file to setup the appropriate python packages.
    - Alternatively, the docker build file can be used to create an image which can run the package.
2) Download the RS3L dataset.
    - The datafiles are too large and thus are not stored in this repository.
    - You can find them on Zenodo:
        - doi: 10.5281/zenodo.10633814

## Configuration

- This package uses Snakemake, Hydra and OmegaConf.

- The different Snakemake modules are found in the `snakemake/workflow` folder. They are all invoked by the main file `all.smk` where individual results can be selected.

- General training configuration is given by the parent config files `configs/train.yaml`, which composes all others, and specific training configuration are founds in the `configs/experiment/` folders.

- In this file you can specify the `network_name`, `seed`, etc.
- More specific settings are found in one of the other config folders:
  - `callbacks`:
    - Provides a collection of `lightning.pytorch.callbacks` to run during training.
  - `datamodule`:
    - Defines the data files used for training and testing.
    - Also which coordinates are used for the object kinematics.
  - `hydra`:
    - Configures the hydra package for running, does not need to be changed.
  - `loggers`:
    - By default we use `Weights and Biases` for logging.
      - You will need to make a free account here: `https://wandb.ai/` and put your username in the `entity` entry of this yaml file.
  - `model`:
    - Configures the model architecture and hyperparameters for training.
    - By default the transformer + normalising flow is used.
  - `paths`:
    - Define the paths to the data download here as well as the desired save directory for the models.
  - `trainer`:
    - Configuration for the PyTorch Lightning `Trainer` class.

## Running

- The main workflow can be launched using the `invoke experiment-run name --workflow all`, where `name` is used to separate different runs and can be chosen arbitrarily. Execution order of scripts can be retraced by following `inputs` and `outputs` section of the individual snakemake files.

- The most important scripts are given below:

1) `scripts/train.py`
    - Compiles the run config as described above and trains the model.
    - Will save checkpoints based on the `paths.output_dir` key.

2) `scripts/export.py`
    - Creates an output `.h5` file containing results for each event in the models test set.

3) `franckstools/sam.py`
    - Contains a wrapper including all the logic to train a pytorch lightning model using the different sharpness aware methods. (Weight-space)

4) `franckstools/adversarial_attack.py`
    - Contains a wrapper including all the logic to train a pytorch lightning model using the different adversarial methods. (Feature-space)
