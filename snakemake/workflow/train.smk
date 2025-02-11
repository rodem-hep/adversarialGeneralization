configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

# Import other snakemake files
include: "prepareDataset.smk"

# use realpath to avoid symlinks
container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)
exp_id = f"{exp_group}/{exp_name}"

config_path = os.path.join(config["config_path"])

envvars:
    "WANDB_API_KEY",

# train_with_mass = config["train_with_mass"] #TODO like this
train_with_mass = True

import yaml
# Load dataset configuration
with open(f"{config_path}/data_confs/trained_on.yaml", 'r') as file:
    trained_on_data_config = yaml.safe_load(file)
with open(f"{config_path}/data_confs/evaluated_on.yaml", 'r') as file:
    evaluated_on_data_config = yaml.safe_load(file)

rule train:
    output:
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/checkpoints/last.ckpt",
    
    params:
        "scripts/train.py",
        
        "experiment=train_{network_type}_classifier_{method_type}",
        "datamodule.data_conf.dataset_type={trained_on_dataset_type}",
        # lambda wc: f"datamodule.data_conf.datasets.c0={signal_background_name_dict[wc.trained_on_dataset_type][0]}",
        # lambda wc: f"datamodule.data_conf.datasets.c1={signal_background_name_dict[wc.trained_on_dataset_type][1]}",
        "+append_log_mass=true" if train_with_mass else "+append_log_mass=false",
        "project_name=supervised_{trained_on_dataset_type}_{method_type}",
        "network_name={network_type}_{trained_on_dataset_type}_{method_type}",
        f"paths.output_dir={exp_path}/taggers",

        # Transfer path (for SWAG)
        lambda wc: f"transfer_ckpt_path={exp_path}/taggers/supervised_{wc.trained_on_dataset_type}_default/{wc.network_type}_{wc.trained_on_dataset_type}_default/checkpoints" if wc.method_type in ["SWAG","SWAGcyclic"] else "",

    resources:
        runtime=48*60,
        slurm_partition="shared-gpu,private-dpnc-gpu",
        slurm_extra="--gres=gpu:1",

    input: 
        f"{config_path}experiment/train_{{network_type}}_classifier_{{method_type}}.yaml",
        lambda wc: [f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_default/{{network_type}}_{{trained_on_dataset_type}}_default/checkpoints/last.ckpt"] if wc.method_type in ["SWAG","SWAGcyclic"] else [],
        
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_QCD_train.h5",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_Hbb_train.h5",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

