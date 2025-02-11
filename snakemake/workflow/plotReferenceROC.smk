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

import yaml
# Load dataset configuration
with open(f"{config_path}/data_confs/default.yaml", 'r') as file:
    data_config = yaml.safe_load(file)

# Initialize the lists/dicts
dataset_types = []
files_list_dict = {}

# Populate the lists/dicts
for dataset in data_config:
    dataset_types.append(dataset['dataset_type'])
    files_list_dict[dataset['dataset_type']] = [dataset['datasets']['c0'], dataset['datasets']['c1']]

# Print the lists
# print(f"Dataset Types: {dataset_types}")

rule all:
    input:
        expand(f"{exp_path}/plots/reference_comparison/ROC_{{target_dataset_type}}_ttbar_vs_QCD.png", target_dataset_type=dataset_types)

rule plotROC:
    output:
        plot=f"{exp_path}/plots/reference_comparison/ROC_{{target_dataset_type}}_ttbar_vs_QCD.png",

    params:
        "plotting/compare_taggers.py",
        f"taggers={config['experiment_name']}_reference",
        f"output_dir={exp_path}/plots/reference_comparison",
        lambda wc: f"+sm_files_c0={wc.target_dataset_type}_{files_list_dict[wc.target_dataset_type][0]}_test.h5",
        lambda wc: f"+sm_files_c1={wc.target_dataset_type}_{files_list_dict[wc.target_dataset_type][1]}_test.h5",
        "dataset_type={target_dataset_type}",
        "do_roc_plots=True",
        "do_sculpt_plots=True",        
        lambda wc: f"sculpt_plots_config.file_name={files_list_dict[wc.target_dataset_type][0]}",

    resources:
        runtime=10,
    # log: # I don't think this is necessary
        # f"{exp_path}/logs/plotReferenceRoc_{{target_dataset_type}}.log",
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
