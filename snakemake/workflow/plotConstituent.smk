
configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

# use realpath to avoid symlinks
container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)
exp_id = f"{exp_group}/{exp_name}"

config_path = os.path.join(config["config_path"])


rule plot_pT_Ratio:
    output:
        f"{exp_path}/plots/pT_ratio.png"

    params:
        "scripts/plotConstituents.py",
        f"+exp_path={exp_path}",
        f"+plotPTRatio=True",

    resources:
        runtime=30,
        mem_mb=14000,

    input: 
        f"{config_path}data_confs/evaluated_on.yaml",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_QCD_train.h5",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_Hbb_train.h5",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"    

rule plot_constituents_distribution:
    output:
        f"{exp_path}/plots/constituent_distribution.png"

    params:
        "scripts/plotConstituents.py",
        f"+exp_path={exp_path}",
        f"+plotConstituentDistribution=True",

    resources:
        runtime=30,
        mem_mb=14000,

    input: 
        f"{config_path}data_confs/evaluated_on.yaml",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_QCD_train.h5",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_Hbb_train.h5",
        

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"    