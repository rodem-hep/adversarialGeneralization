"""Compares a collection of taggers together on a standard set of plots."""

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.comparison_plotting import (
    dummy_bump_hunt,
    plot_mass_sculpting,
    save_roc_curves,
)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="plotting.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main script."""
    cfg = hydra.utils.instantiate(cfg)

    # Make the plotting folder
    Path(cfg.output_dir).mkdir(exist_ok=True, parents=True)

    # Plot the inclusive roc curves for each combination of processes dict
    if cfg.do_roc_plot:
        if (
            cfg.sm_files_c0 is not None and cfg.sm_files_c1 is not None
        ):  # SnakeMake Hacky Solution (TODO: Fix this in the future)
            print(
                f"Using sm_files_c0: {cfg.sm_files_c0} and sm_files_c1: {cfg.sm_files_c1}"
            )
            cfg.roc_plots_config.files = [cfg.sm_files_c0, cfg.sm_files_c1]

        cfg.roc_plots_config.is_decor = False
        save_roc_curves(**cfg.roc_plots_config)

        cfg.roc_plots_config.divide_br = True
        cfg.roc_plots_config.do_log = True
        cfg.roc_plots_config.ylim = [1, 1.0e4]
        save_roc_curves(**cfg.roc_plots_config)

    if cfg.do_decor_roc_plot:
        if cfg.sm_files_c0 is not None and cfg.sm_files_c1 is not None:
            cfg.sm_files_c0 = cfg.sm_files_c0.replace(".h5", "_decor.h5")
            cfg.sm_files_c1 = cfg.sm_files_c1.replace(".h5", "_decor.h5")

            print(
                f"Using sm_files_c0: {cfg.sm_files_c0} and sm_files_c1: {cfg.sm_files_c1}"
            )
            cfg.roc_plots_config.files = [cfg.sm_files_c0, cfg.sm_files_c1]

        cfg.roc_plots_config.is_decor = True
        save_roc_curves(**cfg.roc_plots_config)

        cfg.roc_plots_config.divide_br = False
        cfg.roc_plots_config.do_log = False
        cfg.roc_plots_config.ylim = [0, 1]
        save_roc_curves(**cfg.roc_plots_config)

    if cfg.do_sculpt_plots:
        plot_mass_sculpting(**cfg.sculpt_plots_config)

    if cfg.do_sculpt_plots_methods_list:
        plot_mass_sculpting(**cfg.sculpt_plots_config, cfg=cfg)

    # # Following plots are done per tagger
    # for tagger in cfg.taggers:
    #     if cfg.do_dummy_bump:
    #         dummy_bump_hunt(tagger, **cfg.dummy_bump_config)


if __name__ == "__main__":
    main()
