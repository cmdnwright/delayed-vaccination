import yaml

def load_config(path: str = "config.yaml") -> dict:
    '''loads a yaml config

    Parameters
    ----------
    path : str, optional
        the full config path relative to the root, by default "config.yaml"

    Returns
    -------
    dict
        the config
    '''
    with open(path, "r") as f:
        return yaml.safe_load(f)