
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
include: "export.smk"

rule decorrelatePredictions:
    output:
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{predicted_on_dataset_type}}_signal_test_decor.h5",
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{predicted_on_dataset_type}}_background_test_decor.h5", 
    
    params:
        "scripts/binned_decorrelation.py",
        
        "decorrelate_single_tagger=True",
        "tagger_name={network_type}_{trained_on_dataset_type}_{method_type}",
        f"tagger_path={exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}",
        "tagger_score_name=output",
        f"plot_OT=True",
        f"output_dir={exp_path}/plots/OT_score_dist_comparison", #Path to save the plot
        "decor_type={predicted_on_dataset_type}",
        "decor_dset=test",
        "dataset_type={predicted_on_dataset_type}",
        f"decor_files=null",
        lambda wc: f"+sm_decor_files_c0=QCD",
        lambda wc: f"+sm_decor_files_c1=Hbb",
        f"files=null",
        lambda wc: f"+sm_files_c0=QCD",
        lambda wc: f"+sm_files_c1=Hbb",

    input:
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{predicted_on_dataset_type}}_signal_test.h5",
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{{predicted_on_dataset_type}}_background_test.h5", # Part of the input

    wildcard_constraints:
        # Specify a regular expression for dataset_types that excludes underscores
        trained_on_dataset_type = "[^_]+",
        predicted_on_dataset_type = "[^_]+",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
    


