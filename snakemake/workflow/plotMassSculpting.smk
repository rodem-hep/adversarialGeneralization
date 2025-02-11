
configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

# use realpath to avoid symlinks
container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)
exp_id = f"{exp_group}/{exp_name}"

config_path = os.path.join(config["config_path"])

# Import other snakemake files
include: "export.smk"

rule plotAdversarialMassSculpting:
  output:
    f"{exp_path}/{{network_type}}/plots/adversarial/mass_sculpting_{{trained_on_dataset_type}}.pdf"

  params:
    "plotting/compare_taggers.py",
    "do_sculpt_plots=False",
    "do_sculpt_plots_methods_list=True",
    "do_roc_plot=False",
    "do_mass_correlation_plot=False",
    f"exp_path={exp_path}",
    "method_types=[SAM,SSAMD,FGSM,PGD]",
    "color_list=[blue,orange,red,green]",
    f"output_dir={exp_path}/{{network_type}}/plots/adversarial/",
    "dataset_type={trained_on_dataset_type}",

  
  wildcard_constraints:
    # Specify a regular expression for dataset_types that excludes underscores
    trained_on_dataset_type = "[^_]+",
    evaluated_on_dataset_type = "[^_]+",

  resources:
    runtime=15,

  input:
    *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{trained_on_dataset_type}}_signal_test.h5" for method_type in ["SAM", "SSAMD", "FGSM", "PGD"]],
    *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{trained_on_dataset_type}}_background_test.h5" for method_type in ["SAM", "SSAMD", "FGSM", "PGD"]],

  wrapper:
    "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule plotDecorMassSculpting:
  output:
    f"{exp_path}/{{network_type}}/plots/decor/mass_sculpting_{{trained_on_dataset_type}}.pdf"

  params:
    "plotting/compare_taggers.py",
    "do_sculpt_plots=False",
    "do_sculpt_plots_methods_list=True",
    "do_roc_plot=False",
    "do_mass_correlation_plot=False",
    "decor=True",
    f"exp_path={exp_path}",
    "method_types=[default]",
    "color_list=[black]",
    f"output_dir={exp_path}/{{network_type}}/plots/decor/",
    "dataset_type={trained_on_dataset_type}",

  
  wildcard_constraints:
    # Specify a regular expression for dataset_types that excludes underscores
    trained_on_dataset_type = "[^_]+",
    evaluated_on_dataset_type = "[^_]+",

  resources:
    runtime=15,

  input:
    *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{trained_on_dataset_type}}_signal_test.h5" for method_type in ["default"]],
    *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{method_type}/{{network_type}}_{{trained_on_dataset_type}}_{method_type}/outputs/{{trained_on_dataset_type}}_background_test.h5" for method_type in ["default"]],

  wrapper:
    "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
