# ==============================================================================
# Script:           config.py
# Purpose:          Utility functions for configuration and initialization
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             11/18/2025
# ==============================================================================

import yaml
from pathlib import Path

from methylvae.constants import CONFIG_DIR


# =====| File I/O Utilities |===================================================

def resolve_path(path_str, default_path, build_path = False):
    """
    Resolves and returns the path. If a relative path is provided through
    a file or directory name, it is automatically resolved relative to the 
    project root. Else, the absolute path is returned as provided.

    Parameters
    ----------
    path_str (str): path to a YAML configuration file
    default_path (str): default path (constant) to the project root
    build_path (boolean): appends the path to the default if toggled

    Returns
    -------
    path (Path): resolved Path object from pathlib
    """

    p = Path(path_str)

    if p.is_absolute():
        return p.resolve()
    elif build_path:
        return (default_path / path_str).resolve()
    else:
        return default_path
    

def load_config(path_str):
    """
    Loads and returns the configuration file provided by the path. If a relative
    path is provided (filename), automatically resolves it relative to the 
    project root.

    Parameters
    ----------
    path_str (str): path to a YAML configuration file

    Returns
    -------
    config (dict): dictionary of configuration values

    Raises
    ------
    FileNotFoundError: if the file does not exist at the specified path
    ValueError: if the YAML file cannot be parsed into a dictionary
    """

    # Resolve to the project's root if the provided path is not absolute
    path = resolve_path(path_str, CONFIG_DIR, build_path = True)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}.")

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file {path} did not return a dictionary."
        )

    return config

# [END]