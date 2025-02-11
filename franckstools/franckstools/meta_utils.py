import os

def list_subfolders(path):
    try:
        subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        formatted_output = ' '.join(subfolders)
        print(f'({formatted_output})')
        print(f'Length: {len(subfolders)}')
    except Exception as e:
        print(f'Error: {e}')

def extend_tagger_list_sweep(cfg, networks_dir, yaml_file_out='sweep.yaml'):
    import os
    from omegaconf import OmegaConf

    # Load the yaml file
    cfg = OmegaConf.load(cfg)
    new_cfg = cfg.copy()
    
    # Define colors for the models in the rainbow (excluding blue and black)
    rainbow_colors = ["red", "orange", "yellow", "green", "purple", "pink", "brown", "cyan", "magenta"]
    
    # Get a list of subdirectories in the supervised_swag_cyclic_path
    subfolders = [f.path for f in os.scandir(networks_dir) if f.is_dir()]
    # Append models to the taggers list
    for subfolder in subfolders:
        if '.hydra' in subfolder:
            continue
        label = subfolder[-41:] #TODO: make this more robust
        new_cfg.append({
            "name": label,
            "path": networks_dir,
            "score_name": "output",
            "label": label,
            "plot_kwargs":
            {
                "linestyle": "solid",
                "color": rainbow_colors[len(new_cfg) % len(rainbow_colors)],
            },
        })

    # Save the new cfg as a yaml file
    with open(yaml_file_out, 'w') as f:
        OmegaConf.save(new_cfg, f.name)

    return new_cfg

def main():
    path = '/home/users/r/rothenf3/workspace/Jettagging/jettagging/jobs/taggers/supervised_SSAMD'
    # list_subfolders(path)
    cfg_path = "/home/users/r/rothenf3/workspace/Jettagging/jettagging/configs/taggers/SSAMDsweep.yaml"
    extend_tagger_list_sweep(cfg_path, path, yaml_file_out='/home/users/r/rothenf3/workspace/Jettagging/jettagging/configs/taggers/ExtSamSweep.yaml')

if __name__ == '__main__':
    main()