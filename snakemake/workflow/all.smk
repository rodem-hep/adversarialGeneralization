configfile: "snakemake/template/common/default.yaml"
configfile: "snakemake/template/common/private.yaml"

container: os.path.realpath(config["container_path"])

exp_group, exp_name = config["experiment_group"], config["experiment_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)

config_path = os.path.join(config["config_path"])

envvars:
    "WANDB_API_KEY",

# Import other snakemake files
include: "prepareDataset.smk"
include: "plotROC.smk"
include: "plotGradientAscent.smk"
include: "plotAverageLossLandscape.smk"
include: "plotFR.smk"
include: "plotConstituent.smk"
include: "plotScoresArray.smk"
include: "plotMassSculpting.smk"
include: "performHessianAnalysis.smk"

import yaml
# Load dataset configuration
with open(f"{config_path}/data_confs/evaluated_on.yaml", 'r') as file:
    evaluated_on_data_config = yaml.safe_load(file)
with open(f"{config_path}/data_confs/trained_on.yaml", 'r') as file:
    trained_on_data_config = yaml.safe_load(file)

# Initialize the lists/dicts
evaluated_on_dataset_types = []
trained_on_dataset_types = []

# Populate the lists/dicts
for dataset in evaluated_on_data_config:
    evaluated_on_dataset_types.append(dataset['dataset_type'])
for dataset in trained_on_data_config:
    trained_on_dataset_types.append(dataset['dataset_type'])

# Load the experiment configuration
with open(f"{config_path}/exp_confs/adversarial.yaml", 'r') as file:
    exp_config = yaml.safe_load(file)


# Initialize the lists
# method_types = [method['experiment_name'] for method in exp_config if method['experiment_name'] != 'default']
method_types = [method['experiment_name'] for method in exp_config]

network_types = ["dense"]#, "simpleTransformer"]
method_file_names = ["all", "adversarial"], #"SWAG"]


rule all:
    input:
        # Constituents distribution plots
        *[f"{exp_path}/plots/pT_ratio.png"],
        *[f"{exp_path}/plots/constituent_distribution.png"],

        # Mass distribution plots
        *[f"{exp_path}/plots/mass_distribution/{dataset_type}_mass_distribution.png" for dataset_type in evaluated_on_dataset_types],
        
        # Reference comparison ROC plots
        *[f"{exp_path}/{network_type}/plots/reference_comparison/ROC_{evaluated_on_dataset_type}_signal_vs_background.png" for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types],
        *[f"{exp_path}/{network_type}/plots/reference_comparison/ROC_{evaluated_on_dataset_type}_signal_vs_background_decor.png" for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types], # ROC plots *[f"{exp_path}/{network_type}/plots/methods/{method}/ROC_{evaluated_on_dataset_type}_signal_vs_background.png" for method in method_types for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types],
        *[f"{exp_path}/{network_type}/plots/methods/{method}/ROC_{evaluated_on_dataset_type}_signal_vs_background_decor.png" for method in method_types for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types],

        # Mass sculpting 
        *[f"{exp_path}/{network_type}/plots/adversarial/mass_sculpting_{trained_on_dataset_type}.pdf" for network_type in network_types for trained_on_dataset_type in trained_on_dataset_types],
        *[f"{exp_path}/{network_type}/plots/decor/mass_sculpting_{trained_on_dataset_type}.pdf" for network_type in network_types for trained_on_dataset_type in trained_on_dataset_types],

        # Quantile mass distribution plots
        *[f"{exp_path}/{network_type}/plots/methods/{method}/qmass_distribution/qmass_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png" for method in method_types for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types],
        # Mass morphing
        *[f"{exp_path}/{network_type}/plots/methods/{method}/qmass_distribution/qmass_morphing_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.pdf" for method in method_types for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types],


        # Gradient ascent plots
        *[f"{exp_path}/{network_type}/plots/gradientAscentWeight_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.pdf" for network_type in network_types for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types],
        *[f"{exp_path}/{network_type}/plots/gradientAscentInput_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.pdf" for network_type in network_types for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types],

        # Individual plots
        # *[f"{exp_path}/{network_type}/plots/methods/{method}/gradientAscentWeight_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.pdf" for method in method_types for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types ], 
        # *[f"{exp_path}/{network_type}/plots/methods/{method}/gradientAscentInput_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.pdf" for method in method_types for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types for network_type in network_types ],

        # Average loss landscape plots
        *[f"{exp_path}/{network_type}/plots/avgLossLandscape_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.pdf" for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types if evaluated_on_dataset_type==trained_on_dataset_type for network_type in network_types], # Only plot the diagonal for now

        # Hessian analysis
        *[f"{exp_path}/{network_type}/hessianAnalysis/TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}_hessian_eigenvalues.txt" for network_type in network_types for trained_on_dataset_type in trained_on_dataset_types for evaluated_on_dataset_type in evaluated_on_dataset_types],

        # # Reference scores summary table
        # *[f"{exp_path}/{network_type}/scores_array/reference_rejection_heatmap.png" for network_type in network_types],
        # *[f"{exp_path}/{network_type}/scores_array/reference_rejection_heatmap_decor.png" for network_type in network_types], 
        # # Scores summary table
        # *[f"{exp_path}/{network_type}/scores_array/rejection_heatmap_{trained_on_dataset_type}_M_{method_file_name}.png" for trained_on_dataset_type in trained_on_dataset_types for network_type in network_types for method_file_name in method_file_names],
        # *[f"{exp_path}/{network_type}/scores_array/rejection_heatmap_{trained_on_dataset_type}_M_{method_file_name}_decor.png" for trained_on_dataset_type in trained_on_dataset_types for network_type in network_types for method_file_name in method_file_names],

        # Split Datasets in Train and Test
        # *[f"/srv/beegfs/scratch/groups/rodem/datasets/RS3L/FR_RS3L_{class_name}_{dset}.h5" for class_name in ["QCD", "Hbb"] for dset in ["train", "test"]],
        
        # # Average loss landscape plots
        # # *[f"{exp_path}/{network_type}/plots/methods/{method}/avgLossLandscape_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png" for method in method_types for trained_on_dataset_type in dataset_types for evaluated_on_dataset_type in dataset_types if evaluated_on_dataset_type==trained_on_dataset_type for network_type in network_types ], # Only plot the diagonal for now

        # # Score distribution plots
        # *[f"{exp_path}/{network_type}/plots/methods/{method}/{trained_on_dataset_type}_score_distributions.png" for method in method_types for trained_on_dataset_type in dataset_types for network_type in network_types],

        # # Quantile mass distribution plots
        # *[f"{exp_path}/{network_type}/plots/methods/{method}/qmass_distribution/qmass_TR_{trained_on_dataset_type}_EV_{evaluated_on_dataset_type}.png" for method in method_types for trained_on_dataset_type in dataset_types for evaluated_on_dataset_type in dataset_types for network_type in network_types],
        

