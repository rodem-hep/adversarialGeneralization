configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)

config_path = os.path.join(config["config_path"])



rule create_train_set:
    output: 
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_{class_name}_train.h5",
        
    params:
        "src/splitDataset.py",
        "class_name={class_name}",
        "target_amount=500000",
        "dset=train",

    resources:
        runtime=24*60,
        mem_mb=32000,

    input:
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/rs3l_0.h5",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule create_test_set:
    output: 
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_{class_name}_test.h5",
        
    params:
        "src/splitDataset.py",
        "class_name={class_name}",
        "target_amount=100000",
        "dset=test",

    input:
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/rs3l_0.h5",

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"