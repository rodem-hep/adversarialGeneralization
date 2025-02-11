
configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

# use realpath to avoid symlinks
container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)
exp_id = f"{exp_group}/{exp_name}"

config_path = os.path.join(config["config_path"])

# Import other snakemake files
include: "train.smk"


import yaml
# Load the experiment configuration
with open(f"{config_path}/exp_confs/all.yaml", 'r') as file:
    exp_config = yaml.safe_load(file)

# Initialize the lists
method_types = [method['experiment_name'] for method in exp_config]
with open(f"{config_path}/exp_confs/SWAG.yaml", 'r') as file:
    exp_config = yaml.safe_load(file)
method_types += [method['experiment_name'] for method in exp_config]

def SWAG_method_convertor(wildcards):
    if wildcards.method_type == "SWA":
        return "SWAG"
    elif wildcards.method_type == "SWAcyclic":
        return "SWAGcyclic"
    else:
        return wildcards.method_type

adversarial_method_types = ["default", "SAM", "SSAMD", "FGSM", "PGD"]

# rule all:
# 	input:
# 		# Gradient Ascent
# 		*[f"{exp_path}/{{network_type}}/plots/methods/{method_type}/gradientAscentWeight_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png" for trained_on_dataset_type in dataset_types for evaluated_on_dataset_type in dataset_types for method_type in method_types],
# 		*[f"{exp_path}/{{network_type}}/plots/methods/{method_type}/gradientAscentInput_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png" for trained_on_dataset_type in dataset_types for evaluated_on_dataset_type in dataset_types for method_type in method_types],

rule plotAdversarialGradientAscent:
    output:
        f"{exp_path}/{{network_type}}/plots/gradientAscentWeight_TR_{{trained_on_dataset_type}}_EV_{{evaluated_on_dataset_type}}.pdf",
        f"{exp_path}/{{network_type}}/plots/gradientAscentInput_TR_{{trained_on_dataset_type}}_EV_{{evaluated_on_dataset_type}}.pdf",

    params:
        "scripts/plt_gradient_ascent.py",
        "datamodule.data_conf.dataset_type={evaluated_on_dataset_type}",
        f"exp_path={exp_path}",
        "method_types=[SAM,SSAMD,FGSM,PGD]",
        "color_list=[blue,orange,red,green]",
        f"output_dir={exp_path}/{{network_type}}/plots",
        "evaluated_on_dataset_type={evaluated_on_dataset_type}",
        "trained_on_dataset_type={trained_on_dataset_type}",
        "network_type={network_type}",

    wildcard_constraints:
        # Specify a regular expression for dataset_types that excludes underscores
        trained_on_dataset_type = "[^_]+",
        evaluated_on_dataset_type = "[^_]+",
    resources:
        runtime=15,

    input:
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_weight_loss_tensor.pt" for method_type in adversarial_method_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_weight_std_tensor.pt" for method_type in adversarial_method_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_input_loss_tensor.pt" for method_type in adversarial_method_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_input_std_tensor.pt" for method_type in adversarial_method_types],

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"


rule plotGradientAscent:
    output: 
        f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/gradientAscentWeight_TR_{{trained_on_dataset_type}}_EV_{{evaluated_on_dataset_type}}.pdf",
        f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/gradientAscentInput_TR_{{trained_on_dataset_type}}_EV_{{evaluated_on_dataset_type}}.pdf",

    params:
        "scripts/plt_gradient_ascent.py",
        "datamodule.data_conf.dataset_type={evaluated_on_dataset_type}",
#        lambda wc: f"datamodule.data_conf.datasets.c0={files_list_dict[wc.evaluated_on_dataset_type][0]}",
#        lambda wc: f"datamodule.data_conf.datasets.c1={files_list_dict[wc.evaluated_on_dataset_type][1]}",
        f"exp_path={exp_path}",
        "method_types=[{method_type}]",
        "color_list=[blue]",
        f"output_dir={exp_path}/{{network_type}}/plots/methods/{{method_type}}",
        "evaluated_on_dataset_type={evaluated_on_dataset_type}",
        "trained_on_dataset_type={trained_on_dataset_type}",
        "network_type={network_type}",

    wildcard_constraints:
        # Specify a regular expression for dataset_types that excludes underscores
        trained_on_dataset_type = "[^_]+",
        evaluated_on_dataset_type = "[^_]+",
    resources:
        runtime=15,

    input:
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_weight_loss_tensor.pt", 
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_input_loss_tensor.pt", 
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_default/{{network_type}}_{{trained_on_dataset_type}}_default/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_weight_loss_tensor.pt", 
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_default/{{network_type}}_{{trained_on_dataset_type}}_default/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_input_loss_tensor.pt", 

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule exportGradientAscent:
    output: 
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_weight_loss_tensor.pt",
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_weight_std_tensor.pt",
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_input_loss_tensor.pt",
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_gradient_ascent_input_std_tensor.pt",
    
    params:
        "scripts/export_gradient_ascent.py",
        "datamodule.data_conf.dataset_type={evaluated_on_dataset_type}",
#        lambda wc: f"datamodule.data_conf.datasets.c0={files_list_dict[wc.evaluated_on_dataset_type][0]}",
#        lambda wc: f"datamodule.data_conf.datasets.c1={files_list_dict[wc.evaluated_on_dataset_type][1]}",
        f"output_dir={exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs",
        "method_type={method_type}",

        lambda wc: f"tagger_name={wc.network_type}_{wc.trained_on_dataset_type}_{SWAG_method_convertor(wc)}",
        lambda wc: f"tagger_path={exp_path}/taggers/supervised_{wc.trained_on_dataset_type}_{SWAG_method_convertor(wc)}",
        
        "evaluated_on_dataset_type={evaluated_on_dataset_type}",
        "trained_on_dataset_type={trained_on_dataset_type}",

        # SWAG
        lambda wc: "SWAG_export=True" if wc.method_type in ["SWAG", "SWAGcyclic"] else "SWAG_export=False",
        lambda wc: "SWA_export=True" if wc.method_type in ["SWA", "SWAcyclic"] else "SWA_export=False",

    wildcard_constraints:
        # Specify a regular expression for dataset_types that excludes underscores
        trained_on_dataset_type = "[^_]+",
        predicted_on_dataset_type = "[^_]+",
        
    resources:
        runtime=8*60,
        
    input:
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_Hbb_test.h5",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_QCD_test.h5",

        lambda wc: f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{SWAG_method_convertor(wc)}/{{network_type}}_{{trained_on_dataset_type}}_{SWAG_method_convertor(wc)}/checkpoints/last.ckpt",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
