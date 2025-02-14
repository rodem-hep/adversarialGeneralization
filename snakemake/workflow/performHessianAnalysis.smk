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

rule exportHessianLargestEigenvalue:
    output:
        *[f"{exp_path}/{{network_type}}/hessianAnalysis/{{method_type}}/TR_{{trained_on_dataset_type}}/EV_{{evaluated_on_dataset_type}}/{dataset}_hessian_largest_eigenvalue_input.txt" for dataset in ["QCD", "Hbb"]],
        *[f"{exp_path}/{{network_type}}/hessianAnalysis/{{method_type}}/TR_{{trained_on_dataset_type}}/EV_{{evaluated_on_dataset_type}}/{dataset}_hessian_largest_eigenvalue_weight.txt" for dataset in ["QCD", "Hbb"]],

    params:
        "scripts/export_hessian.py",
        "datamodule.data_conf.dataset_type={evaluated_on_dataset_type}",
        f"output_dir={exp_path}/{{network_type}}/hessianAnalysis/{{method_type}}/TR_{{trained_on_dataset_type}}/EV_{{evaluated_on_dataset_type}}",
        f"tagger_name={{network_type}}_{{trained_on_dataset_type}}_{{method_type}}",
        f"tagger_path={exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}",
        "evaluated_on_dataset_type={evaluated_on_dataset_type}",
        "trained_on_dataset_type={trained_on_dataset_type}",
    resources:
        runtime = 2*60,

    input:
        f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/checkpoints/last.ckpt",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

