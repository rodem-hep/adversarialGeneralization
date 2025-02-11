configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)

config_path = os.path.join(config["config_path"])

envvars:
    "WANDB_API_KEY",


    
rule generateExperimentTaggerList:
    output:
        f"{config_path}taggers/generated/{{exp_name}}_{{network_type}}_{{method_type}}_taggerList.yaml"

    params:
        "src/generateHydraTaggers.py",

        f"generateTaggerListForMethod=True",
        f"method_type={{method_type}}",
        f"network_type={{network_type}}",
        f"config_path={config_path}",
        f"exp_path={exp_path}",
        f"output_dir={config_path}/taggers/generated",
        f"experiment_name={config['experiment_name']}",
    
    resources:
        mem_mb=1000,
        time_min=10,

    input:
        f"{config_path}taggers/generated/{config['experiment_name']}_{{network_type}}_trainedOnTaggerList.yaml"

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
        
rule generateTrainedOnTaggerList:
    output: 
        f"{config_path}taggers/generated/{{exp_name}}_{{network_type}}_trainedOnTaggerList.yaml"

    params:
        "src/generateHydraTaggers.py",

        f"generateReferenceTaggerList=True",
        f"trainedORevaluatedOn=trained_on",
        f"network_type={{network_type}}",
        f"config_path={config_path}",
        f"exp_path={exp_path}",
        f"output_dir={config_path}/taggers/generated",
        f"experiment_name={{exp_name}}",
    
    resources:
        mem_mb=1000,
        time_min=10,

    input: 
        f"{config_path}data_confs/trained_on.yaml"

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule generateEvaluatedOnTaggerList:
    output:
        f"{config_path}taggers/generated/{{exp_name}}_{{network_type}}_evaluatedOnTaggerList.yaml"

    params:
        "src/generateHydraTaggers.py",

        f"generateReferenceTaggerList=True",
        f"trainedORevaluatedOn=evaluated_on",
        f"network_type={{network_type}}",
        f"config_path={config_path}",
        f"exp_path={exp_path}",
        f"output_dir={config_path}/taggers/generated",
        f"experiment_name={{exp_name}}",
    
    resources:
        mem_mb=1000,
        time_min=10,

    input: 
        f"{config_path}data_confs/evaluated_on.yaml"

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

    