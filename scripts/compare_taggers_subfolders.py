"""Compares a collection of taggers together on a standard set of plots."""

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import numpy as np

import src.comparison_plotting as cp

import os


def main() -> None:
    """Main script."""

    output_dir = root / "plots"
    data_dir = Path("/srv/beegfs/scratch/groups/rodem/anomalous_jets/virtual_data/")
    #tagger_dir = Path("/srv/beegfs/scratch/groups/rodem/anomalous_jets/taggers/")
    tagger_dir = Path("/home/users/r/rothenf3/workspace/Jettagging/jettagging/jobs/taggers")
    networks_dir = tagger_dir / "supervised_swag"
    
    # Define colors for the models in the rainbow (excluding blue and black)
    rainbow_colors = ["red", "orange", "yellow", "green", "purple", "pink", "brown", "cyan", "magenta"]


    taggers = [
        #Reference sets
        {
            "path": tagger_dir / "supervised" / "dense_test" / "outputs",
            "label": "dense trained on Rodem",
            "score_name": "output",
            "linestyle": "solid",
            "color": "black",
        },
        # {
        #      "path": tagger_dir / "supervised_toptag" / "dense_toptag_test" / "outputs",
        #      "label": "dense trained on TopTag",
        #      "score_name": "output",
        #      "linestyle": "solid",
        #      "color": "blue",
        # },
        {
            "path": tagger_dir / "supervised_jetnet" / "dense_jetnet_test" / "outputs",
            "label": "dense trained on Jetnet",
            "score_name": "output",
            "linestyle": "solid",
            "color": "blue",
        },
        # {
        #      "path": tagger_dir / "supervised_jetclass" / "dense_test_jetclass" / "outputs",
        #      "label": "dense trained on JetClass",
        #      "score_name": "output",
        #      "linestyle": "solid",
        #      "color": "blue",
        # },
    ]

    # Get a list of subdirectories in the supervised_swag_cyclic_path
    subfolders = [f.path for f in os.scandir(networks_dir) if f.is_dir()]
    # Append models to the taggers list
    for subfolder in subfolders:
        if '.hydra' in subfolder:
            continue
        label = subfolder[-6:]
        taggers.append({
            "path": f"{subfolder}/outputs",
            "label": label,
            "score_name": "output",
            "linestyle": "solid",
            "color": rainbow_colors[len(taggers) % len(rainbow_colors)],
        })


    processes = {
        # Rodem
        # "QCD": "Rodem_QCD_jj_pt_450_1200_test_SWA.h5",
        # "ttbar": "Rodem_ttbar_allhad_pt_450_1200_test_SWA.h5",
        
        # TopTag
        # "QCD": "TopTag_QCD_test.h5",
        # "ttbar": "TopTag_ttbar_test.h5"

        # JetNet
        "QCD": "JetNet_q_test_SWAG.h5",
        "ttbar": "JetNet_t_test_SWAG.h5"

        # JetClass
        #  "QCD": "JetClass_JetClass_QCD_test.h5",
        #  "ttbar": "JetClass_JetClass_ttbar_test.h5",

        #"WZ": "WZ_allhad_pt_450_1200_test.h5",
        #"H2": "H2tbtb_1700_HC_250_test.h5",
    }

    # Plot the inclusive roc curves for each combination of processes dict
    cp.plot_roc_curves(
        output_dir, taggers, processes, br_at_eff=[0.3, 0.7], ylim=(1, 1e5)
    )

    exit()

    # Plot the mass sculpting (Jensen-Shannon) as a function of the background rej
    cp.plot_mass_sculpting(
        output_dir,
        taggers,
        "QCD",
        "QCD_jj_pt_450_1200_test.h5",
        data_dir,
        bins=np.linspace(0, 300, 40),
        br_values=[0.5, 0.8, 0.9, 0.95, 0.99],
        xlim=[0.4, 1.1],
        do_log=True,
    )

    for sig_name in processes:
        pass
        # cp.run_signal_injection(
        #     output_dir,
        #     data_dir,
        #     taggers,
        #     background_name="QCD",
        #     background_file="QCD_jj_pt_450_1200_test.h5",
        #     signal_name=sig_name,
        #     signal_file=processes[sig_name],
        #     reject_frac=0.99,
        #     bins=[0, 400],
        # )

    for tagger in taggers:
        # pass

        # cp.plot_sculpted_distributions(
        #     output_dir,
        #     tagger,
        #     processes,
        #     data_dir,
        #     bins=np.linspace(0, 400, 25),
        #     br_values=[0.8, 0.9, 0.95, 0.99],
        #     do_norm=True,
        #     do_log=False,
        # )

        for sig_name in processes:
            # pass

            cp.run_pybumphunter(
                output_dir,
                data_dir,
                tagger,
                background_name="QCD",
                background_file="QCD_jj_pt_450_1200_test.h5",
                signal_name=sig_name,
                signal_file=processes[sig_name],
                reject_frac=0.99,
                snb_ratio=2e-3,
                bins=[0, 400],
            )

            cp.dummy_bump_hunt(
                output_dir,
                data_dir,
                tagger,
                background_name="QCD",
                background_file="QCD_jj_pt_450_1200_test.h5",
                signal_name=sig_name,
                signal_file=processes[sig_name],
                reject_frac=0.99,
                snb_ratio=2e-3,
                bins=np.linspace(0, 400, 40),
                control_region=50,
            )


if __name__ == "__main__":
    main()
