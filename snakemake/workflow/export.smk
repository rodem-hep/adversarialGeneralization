configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

# use realpath to avoid symlinks
container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)
exp_id = f"{exp_group}/{exp_name}"

config_path = os.path.join(config["config_path"])

envvars:
    "WANDB_API_KEY",

# Import other snakemake files
include: "train.smk"

import yaml
# Load dataset configuration
with open(f"{config_path}/data_confs/evaluated_on.yaml", 'r') as file:
    evaluated_on_data_config = yaml.safe_load(file)
# Initialize the lists/dicts
evaluated_on_dataset_types = []

# Populate the lists/dicts
for dataset in evaluated_on_data_config:
    evaluated_on_dataset_types.append(dataset['dataset_type'])

def SWAG_method_convertor(wildcards):
    if wildcards.method_type == "SWA":
        return "SWAG"
    elif wildcards.method_type == "SWAcyclic":
        return "SWAGcyclic"
    else:
        return wildcards.method_type

rule export:
    output:
        # Exports predictions of the tagger on all datasets
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_signal_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_background_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],

    params:
        "scripts/export_FR.py",

        "export_single_tagger=True",
        lambda wc: f"tagger_name={wc.network_type}_{wc.trained_on_dataset_type}_{SWAG_method_convertor(wc)}",
        lambda wc: f"tagger_path={exp_path}/taggers/supervised_{wc.trained_on_dataset_type}_{SWAG_method_convertor(wc)}",
        
        # SWAG export flags
        lambda wc: "SWAG_export=True" if wc.method_type == "SWAG" or wc.method_type == "SWAGcyclic" else "SWAG_export=False",
        lambda wc: "SWA_export=True" if wc.method_type == "SWA" or wc.method_type == "SWAcyclic" else "SWA_export=False",
        lambda wc: "get_best=False" if wc.method_type in ["SWA", "SWAcyclic", "SWAG", "SWAGcyclic"] else "get_best=True", # SWAG should always get last.ckpt

    wildcard_constraints:
        # Specify a regular expression for dataset_types that excludes underscores
        trained_on_dataset_type = "[^_]+",
        predicted_on_dataset_type = "[^_]+",

    resources:
        mem_mb=lambda wildcards: 20000 if wildcards.method_type in ["SWA", "SWAcyclic", "SWAG", "SWAGcyclic"] else 8000,
        runtime=8*60,
        
    input:
        lambda wildcards: f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{SWAG_method_convertor(wildcards)}/{{network_type}}_{{trained_on_dataset_type}}_{SWAG_method_convertor(wildcards)}/checkpoints/last.ckpt",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_Hbb_test.h5",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_QCD_test.h5",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

