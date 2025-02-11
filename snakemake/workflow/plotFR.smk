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
include: "export.smk"

rule PlotMassDistribution:
    output:
        f"{exp_path}/plots/mass_distribution/{{dataset_type}}_mass_distribution.png"
    
    params:
        "scripts/plotting_FR.py",

        "plot_mass_distribution=True",
        f"output_dir={exp_path}/plots/mass_distribution",
        f"config_path={config_path}",
        "dataset_type={dataset_type}",

    resources:
        runtime = 15,
        mem_mb = 12000,

    input:
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_QCD_train.h5",
        "/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_Hbb_train.h5",
    
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"
    

rule PlotScoreDistribution:
    output:
        scoreDistribution=f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/{{trained_on_dataset_type}}_score_distributions.png",
    
    params:
        "scripts/plotting_FR.py",

        "plot_score_distribution=True",
        "single_tagger=True",
        "tagger_name={network_type}_{trained_on_dataset_type}_{method_type}",
        f"tagger_path={exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}",
        f'tagger_label="{{network_type}} {{method_type}} trained on {{trained_on_dataset_type}}"',
        f"output_dir={exp_path}/{{network_type}}/plots/methods/{{method_type}}",
        f"config_path={config_path}",
        "trained_on_dataset_type={trained_on_dataset_type}",
    
    resources: 
        time = 10,
        mem_mb = 4000,

    input:
        # Signal and background prediction files
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_signal_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_background_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],

        # Signal and background prediction files decor
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_signal_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_background_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        

    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule PlotQuantileMassDistribution:
    output:
        *[f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/qmass_distribution/qmass_TR_{{trained_on_dataset_type}}_EV_{evaluated_on_dataset_type}.png" for evaluated_on_dataset_type in evaluated_on_dataset_types]
    
    params:
        "scripts/plotting_FR.py",

        "plot_mass_distribution_quantiles=True",
        "single_tagger=True",
        "tagger_name={network_type}_{trained_on_dataset_type}_{method_type}",
        f"tagger_path={exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}",
        f'tagger_label="{{network_type}} {{method_type}} trained on {{trained_on_dataset_type}}"',
        f"output_dir={exp_path}/{{network_type}}/plots/methods/{{method_type}}/qmass_distribution",
        f"config_path={config_path}",
        "trained_on_dataset_type={trained_on_dataset_type}",
    
    resources: 
        time = 10,
        mem_mb = 4000,

    input:
        # Signal and background prediction files
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_signal_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_background_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],

        # Signal and background prediction files decor
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_signal_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_background_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

rule PlotMassMorphing:
    output:
        *[f"{exp_path}/{{network_type}}/plots/methods/{{method_type}}/qmass_distribution/qmass_morphing_TR_{{trained_on_dataset_type}}_EV_{evaluated_on_dataset_type}.pdf" for evaluated_on_dataset_type in evaluated_on_dataset_types]
    
    params:
        "scripts/plotting_FR.py",

        "plot_mass_morphing=True",
        "single_tagger=True",
        "tagger_name={network_type}_{trained_on_dataset_type}_{method_type}",
        f"tagger_path={exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}",
        f'tagger_label="{{network_type}} {{method_type}} trained on {{trained_on_dataset_type}}"',
        f"output_dir={exp_path}/{{network_type}}/plots/methods/{{method_type}}/qmass_distribution",
        f"config_path={config_path}",
        "trained_on_dataset_type={trained_on_dataset_type}",
    
    resources:
        time = 10,
        mem_mb = 4000,

    input:
        # Signal and background prediction files
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_signal_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_background_test.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],

        # Signal and background prediction files decor
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_signal_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/taggers/supervised_{{trained_on_dataset_type}}_{{method_type}}/{{network_type}}_{{trained_on_dataset_type}}_{{method_type}}/outputs/{evaluated_on_dataset_type}_background_test_decor.h5" for evaluated_on_dataset_type in evaluated_on_dataset_types],
        
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.3/"

