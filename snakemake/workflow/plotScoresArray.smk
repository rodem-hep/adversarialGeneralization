configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)

config_path = os.path.join(config["config_path"])

# Import other snakemake files
include: "decorrelatePredictions.smk"
include: "generateTaggerList.smk"
include: "export.smk"

import yaml
# Load the experiment configuration
with open(f"{config_path}/exp_confs/all.yaml", 'r') as file:
    exp_config = yaml.safe_load(file)
with open(f"{config_path}/data_confs/evaluated_on.yaml", 'r') as file:
    evaluated_on_data_config = yaml.safe_load(file)
with open(f"{config_path}/data_confs/trained_on.yaml", 'r') as file:
    trained_on_data_config = yaml.safe_load(file)

# Initialize the lists
method_types = [method['experiment_name'] for method in exp_config if method['experiment_name'] != 'default']
evaluated_on_dataset_types = [dataset['dataset_type'] for dataset in evaluated_on_data_config]
trained_on_dataset_types = [dataset['dataset_type'] for dataset in trained_on_data_config]

score_metrics = ["AUC", "rejection"]

rule plotReferenceScores:
    output:
        *[f"{exp_path}/{{network_type}}/scores_array/reference_{score_metric}_heatmap.png" for score_metric in score_metrics],
        *[f"{exp_path}/{{network_type}}/scores_array/reference_{score_metric}_heatmap_decor.png" for score_metric in score_metrics],
        
    params:
        "scripts/plt_scores_array.py",
        
        f"config_path={config_path}",
        f"exp_path={exp_path}",
        f"exp_name={exp_name}",
        f"network_type={{network_type}}", 
        "plotReferenceScores=True",

    input:    
        f"{config_path}data_confs/evaluated_on.yaml",
        f"{config_path}taggers/generated/{exp_name}_{{network_type}}_evaluatedOnTaggerList.yaml",
        
        
        # Signal and background prediction files
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{evaluated_on_dataset_type}_signal_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{evaluated_on_dataset_type}_background_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
       

        # Signal and background prediction files decor
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{evaluated_on_dataset_type}_signal_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{evaluated_on_dataset_type}_background_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule plotScores:
    output:
        *[f"{exp_path}/{{network_type}}/scores_array/{score_metric}_heatmap_{trained_on_dataset_type}_M_{{method_file_name}}.png" for trained_on_dataset_type in trained_on_dataset_types for score_metric in score_metrics],
        *[f"{exp_path}/{{network_type}}/scores_array/{score_metric}_heatmap_{trained_on_dataset_type}_M_{{method_file_name}}_decor.png" for trained_on_dataset_type in trained_on_dataset_types for score_metric in score_metrics],
        
    params:
        "scripts/plt_scores_array.py",
        
        f"config_path={config_path}",
        f"exp_path={exp_path}",
        f"exp_name={exp_name}",
        f"network_type={{network_type}}",
        f"method_file_name={{method_file_name}}",
        
        "plotScores=True",
        "plotRatioHeatmap=True",
    
    input:
        f"{config_path}data_confs/evaluated_on.yaml",
        f"{config_path}data_confs/trained_on.yaml",
        f"{config_path}taggers/generated/{exp_name}_{{network_type}}_trainedOnTaggerList.yaml",

         # Signal and background prediction files
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{method_type}/{{network_type}}_{trained_on_dataset_type}_{method_type}/outputs/{evaluated_on_dataset_type}_signal_test.h5" for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for method_type in method_types],
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{method_type}/{{network_type}}_{trained_on_dataset_type}_{method_type}/outputs/{evaluated_on_dataset_type}_background_test.h5" for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for method_type in method_types],


        # Signal and background prediction files decor
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{method_type}/{{network_type}}_{trained_on_dataset_type}_{method_type}/outputs/{evaluated_on_dataset_type}_signal_test_decor.h5" for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for method_type in method_types],
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{method_type}/{{network_type}}_{trained_on_dataset_type}_{method_type}/outputs/{evaluated_on_dataset_type}_background_test_decor.h5" for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for method_type in method_types],
    
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
