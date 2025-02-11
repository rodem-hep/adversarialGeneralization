configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)

config_path = os.path.join(config["config_path"])

envvars:
    "WANDB_API_KEY",

# Import other snakemake files
include: "decorrelatePredictions.smk"
include: "generateTaggerList.smk"
include: "export.smk"

import yaml
# Load trained on dataset configuration
with open(f"{config_path}/data_confs/trained_on.yaml", 'r') as file:
    trained_on_data_config = yaml.safe_load(file)
# Initialize the lists/dicts
trained_on_dataset_types = []

# Populate the lists/dicts
for dataset in trained_on_data_config:
    trained_on_dataset_types.append(dataset['dataset_type'])

# Load evaluated on dataset configuration
with open(f"{config_path}/data_confs/evaluated_on.yaml", 'r') as file:
    evaluated_on_data_config = yaml.safe_load(file)
# Initialize the lists/dicts
evaluated_on_dataset_types = []

# Populate the lists/dicts
for dataset in evaluated_on_data_config:
    evaluated_on_dataset_types.append(dataset['dataset_type'])

rule plotReferenceROC:
    output:
        # signal vs background reference ROC plot
        nondecor = f"{exp_path}/{{network_type}}/plots/reference_comparison/ROC_{{evaluated_on_dataset_type}}_signal_vs_background.png",
        decor = f"{exp_path}/{{network_type}}/plots/reference_comparison/ROC_{{evaluated_on_dataset_type}}_signal_vs_background_decor.png",
        # mass sculpting plot (JSD)
        mass_correlation = f"{exp_path}/{{network_type}}/plots/reference_comparison/mass_sculpting_{{evaluated_on_dataset_type}}.pdf",

    params:
        "plotting/compare_taggers.py",

        f"taggers=generated/{config['experiment_name']}_{{network_type}}_evaluatedOnTaggerList.yaml",
        f"output_dir={exp_path}/{{network_type}}/plots/reference_comparison",
        f"+sm_files_c0={{evaluated_on_dataset_type}}_background_test.h5",
        f"+sm_files_c1={{evaluated_on_dataset_type}}_signal_test.h5",
        "dataset_type={evaluated_on_dataset_type}",
        "do_roc_plot=True",
        "do_decor_roc_plot=True",
        "do_mass_correlation_plot=True",
        lambda wc: f"sculpt_plots_config.file_name=QCD",

    resources:
        mem_mb=4000,
        time_min=10,
    
    input:
        # Tagger list
        f"{config_path}taggers/generated/{config['experiment_name']}_{{network_type}}_evaluatedOnTaggerList.yaml",
        
        # The reference (default method) taggers are trained for all datasets and evaluated on all datasets

        # Signal and background prediction files
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_signal_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_background_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        

        # Signal and background prediction files decor
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_signal_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{evaluated_on_dataset_type}_default/{{network_type}}_{evaluated_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_background_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],


    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"


rule plotROC:
    output:
        ROC = f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/ROC_{{evaluated_on_dataset_type}}_signal_vs_background.png",
        DecorROC = f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/ROC_{{evaluated_on_dataset_type}}_signal_vs_background_decor.png",
    params:
        "plotting/compare_taggers.py",

        f"taggers=generated/{config['experiment_name']}_{{network_type}}_{{method_type}}_taggerList", 
        f"output_dir={exp_path}/{{network_type}}/plots/methods/{{method_type}}",
        f"+sm_files_c0={{evaluated_on_dataset_type}}_background_test.h5",
        f"+sm_files_c1={{evaluated_on_dataset_type}}_signal_test.h5",
        "dataset_type={evaluated_on_dataset_type}",
        "do_roc_plot=True",
        "do_decor_roc_plot=True",
        "do_mass_correlation_plot=True",
        lambda wc: f"sculpt_plots_config.file_name=QCD",

    resources:
        mem_mb=4000,
        time=10,

    input:
        # Tagger list
        f"{config_path}taggers/generated/{config['experiment_name']}_{{network_type}}_{{method_type}}_taggerList.yaml",

        # Signal and background prediction files
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{{method_type}}/{{network_type}}_{trained_on_dataset_type}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_signal_test.h5" for trained_on_dataset_type in trained_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{{method_type}}/{{network_type}}_{trained_on_dataset_type}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_background_test.h5" for trained_on_dataset_type in trained_on_dataset_types],

        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_default/{{network_type}}_{trained_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_signal_test.h5" for trained_on_dataset_type in trained_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_default/{{network_type}}_{trained_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_background_test.h5" for trained_on_dataset_type in trained_on_dataset_types],

        # Signal and background prediction files decor
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{{method_type}}/{{network_type}}_{trained_on_dataset_type}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_signal_test_decor.h5" for trained_on_dataset_type in trained_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_{{method_type}}/{{network_type}}_{trained_on_dataset_type}_{{method_type}}/outputs/{{evaluated_on_dataset_type}}_background_test_decor.h5" for trained_on_dataset_type in trained_on_dataset_types],

        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_default/{{network_type}}_{trained_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_signal_test_decor.h5" for trained_on_dataset_type in trained_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{trained_on_dataset_type}_default/{{network_type}}_{trained_on_dataset_type}_default/outputs/{{evaluated_on_dataset_type}}_background_test_decor.h5" for trained_on_dataset_type in trained_on_dataset_types],

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
