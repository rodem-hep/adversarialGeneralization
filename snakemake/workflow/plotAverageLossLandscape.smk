
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
# Load the experiment configuration
with open(f"{config_path}/exp_confs/all.yaml", 'r') as file:
    exp_config = yaml.safe_load(file)

# Initialize the lists
method_types = [method['experiment_name'] for method in exp_config]

def SWAG_method_convertor(wildcards):
    if wildcards.method_type == "SWA":
        return "SWAG"
    elif wildcards.method_type == "SWAcyclic":
        return "SWAGcyclic"
    else:
        return wildcards.method_type


# rule all:
# 	input:
# 		# Average loss landscapes
# 		*[f"{exp_path}/plots/methods/{method_type}/avgLossLandscape_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png" for trained_on_dataset_type in dataset_types for evaluated_on_dataset_type in dataset_types for method_type in method_types],
#
rule plotAdversarialAvgLossLandscape:
    output:
        f"{exp_path}/{{network_type}}/plots/avgLossLandscape_TR_{{trained_on_dataset_type}}_EV_{{evaluated_on_dataset_type}}.pdf"

    params:
        "scripts/plt_avg_loss_landscape.py",
        "datamodule.data_conf.dataset_type={evaluated_on_dataset_type}",
        # lambda wc: f"datamodule.data_conf.datasets.c0={files_list_dict[wc.evaluated_on_dataset_type][0]}",
        # lambda wc: f"datamodule.data_conf.datasets.c1={files_list_dict[wc.evaluated_on_dataset_type][1]}",
        f"exp_path={exp_path}",
        "method_types=[SAM,SSAMD,FGSM,PGD]",
        "color_list=[blue,orange,red,green]",
        f"output_dir={exp_path}/{{network_type}}/plots/",
        "evaluated_on_dataset_type={evaluated_on_dataset_type}",
        "trained_on_dataset_type={trained_on_dataset_type}",
        "network_type={network_type}",

    wildcard_constraints:
        # Specify a regular expression for dataset_types that excludes underscores
        trained_on_dataset_type = "[^_]+",
        evaluated_on_dataset_type = "[^_]+",

    input:
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{evaluated_on_dataset_type}}_avgLossIncrease.pt" for method_type in ["default", "SAM", "SSAMD", "FGSM", "PGD"]],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{evaluated_on_dataset_type}}_distances.pt" for method_type in ["default", "SAM", "SSAMD", "FGSM", "PGD"]],

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule plotAvgLossLandscape:
	output:
		averageLossLandscape=f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/avgLossLandscape_TR_{{trained_on_dataset_type}}_EV_{{evaluated_on_dataset_type}}.pdf"

	params:
		"scripts/plt_avg_loss_landscape.py",

		"datamodule.data_conf.dataset_type={evaluated_on_dataset_type}",
		# lambda wc: f"datamodule.data_conf.datasets.c0={files_list_dict[wc.evaluated_on_dataset_type][0]}",
		# lambda wc: f"datamodule.data_conf.datasets.c1={files_list_dict[wc.evaluated_on_dataset_type][1]}",
		f"exp_path={exp_path}",
		"method_type={method_type}",
		f"output_dir={exp_path}/{{network_type}}/plots/methods/{{method_type}}",
		"evaluated_on_dataset_type={evaluated_on_dataset_type}",
		"trained_on_dataset_type={trained_on_dataset_type}",
		"network_type={network_type}",

	wildcard_constraints:
		# Specify a regular expression for dataset_types that excludes underscores
		trained_on_dataset_type = "[^_]+",
		evaluated_on_dataset_type = "[^_]+",

	input:
		f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_avgLossIncrease.pt",
		f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_distances.pt",
		f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_default/{{network_type}}_{{trained_on_dataset_type}}_default/outputs/{{evaluated_on_dataset_type}}_avgLossIncrease.pt",
		f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_default/{{network_type}}_{{trained_on_dataset_type}}_default/outputs/{{evaluated_on_dataset_type}}_distances.pt",
		

	wrapper:
		"https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
	

rule exportAvgLossLandscape:
	output:
		f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_avgLossIncrease.pt",
		f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_distances.pt",

	params:
		"scripts/export_avg_loss_landscape.py",
		
		"datamodule.data_conf.dataset_type={evaluated_on_dataset_type}",
		# lambda wc: f"datamodule.data_conf.datasets.c0={files_list_dict[wc.evaluated_on_dataset_type][0]}",
		# lambda wc: f"datamodule.data_conf.datasets.c1={files_list_dict[wc.evaluated_on_dataset_type][1]}",
		f"output_dir={exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs",
		
		lambda wc: f"tagger_name={wc.network_type}_{wc.trained_on_dataset_type}_{SWAG_method_convertor(wc)}",
        lambda wc: f"tagger_path={exp_path}/taggers/supervised_{wc.trained_on_dataset_type}_{SWAG_method_convertor(wc)}",
		
		"evaluated_on_dataset_type={evaluated_on_dataset_type}",
		"trained_on_dataset_type={trained_on_dataset_type}",

		# SWAG
		lambda wc: "SWAG_export=True" if wc.method_type in ["SWAG", "SWAGcyclic"] else "SWAG_export=False",
		lambda wc: "SWA_export=True" if wc.method_type in ["SWA", "SWAcyclic"] else "SWA_export=False",


	resources:
		runtime=6*60,
		mem_mb=14000, # Some out of memory crashes (TODO)

	wildcard_constraints:
		# Specify a regular expression for dataset_types that excludes underscores
		trained_on_dataset_type = "[^_]+",
		evaluated_on_dataset_type = "[^_]+",
	
	input:
		lambda wc: f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{SWAG_method_convertor(wc)}/{{network_type}}_{{trained_on_dataset_type}}_{SWAG_method_convertor(wc)}/checkpoints/last.ckpt",

	wrapper:
		"https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

		
