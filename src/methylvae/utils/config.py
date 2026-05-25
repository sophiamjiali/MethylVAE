# ==============================================================================
# Script:           config.py
# Purpose:          Configuration loading, merging, and path resolution.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             11/18/2025
# ==============================================================================

import yaml
from pathlib import Path
from typing import Any

from methylvae.constants import CONFIG_DIR


# =====| Path Utilities |=======================================================

def resolve_path(path_str: str, default_path: Path, build_path: bool = False) -> Path:
    """
    Resolves a path string to an absolute Path.

    If path_str is absolute, returns it as-is.
    If build_path=True, resolves relative to default_path.
    Otherwise returns default_path directly (used when path_str is empty/missing).

    Parameters
    ----------
    path_str     : Path string from config or CLI argument.
    default_path : Fallback / base path (typically a project constant).
    build_path   : If True, join default_path / path_str when path_str is relative.

    Returns
    -------
    Path : Resolved absolute Path.
    """

    p = Path(path_str)

    if p.is_absolute():
        return p.resolve()
    elif build_path:
        return (default_path / path_str).resolve()
    else:
        return default_path


# =====| YAML Loading |=========================================================

def load_config(path_str: str) -> dict:
    """
    Loads a single YAML config file and returns its contents as a flat dict.
    Relative paths are resolved relative to CONFIG_DIR.

    Parameters
    ----------
    path_str : Path to a YAML configuration file (absolute or relative).

    Returns
    -------
    dict : Raw YAML contents (may be nested).

    Raises
    ------
    FileNotFoundError : If the file does not exist.
    ValueError        : If the file does not parse as a dict.
    """

    path = resolve_path(path_str, CONFIG_DIR, build_path=True)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} did not parse as a dict.")

    return config


# =====| Config Merging |=======================================================

def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Recursively flattens a nested dict using dot-separated keys.
    E.g. {"a": {"b": 1}} → {"a.b": 1}

    Used internally; not part of the public API.
    """

    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

_MERGE_MAP: dict[str, str] = {
    # training.yaml nested keys → flat keys consumed by train.py / callbacks
    "early_stopping.patience":  "early_stopping_patience",
    "early_stopping.min_delta": "early_stopping_min_delta",
    # loss.yaml nested keys → flat keys consumed by train.py / betaVAE
    "anneal.num_cycles":        "num_cycles",
}


def merge_configs(*configs: dict) -> dict:
    """
    Merges multiple config dicts (loaded from separate YAML files) into a
    single flat dict suitable for consumption by train.py and objective.py.

    Merge semantics
    ---------------
    - Dicts are merged left-to-right; later files override earlier ones on
      key collision. This allows e.g. data.yaml to set a default batch_size
      that training.yaml can override.
    - Nested dicts are first flattened with dot-separated keys, then remapped
      via _MERGE_MAP to the canonical flat keys expected by train.py.
    - List-valued keys (encoder_dims, batch_size, num_cycles, etc.) are
      preserved as-is for consumption by objective.py's suggest_* calls.

    Parameters
    ----------
    *configs : Any number of dicts returned by load_config().

    Returns
    -------
    dict : Single flat dict with all keys train.py and objective.py expect.

    Example
    -------
    >>> cfg = merge_configs(
    ...     load_config("data.yaml"),
    ...     load_config("training.yaml"),
    ...     load_config("loss.yaml"),
    ... )
    >>> cfg["input_dim"]               # from data.yaml
    >>> cfg["max_epochs"]              # from training.yaml
    >>> cfg["free_bits"]               # from loss.yaml
    >>> cfg["num_cycles"]              # remapped from loss.yaml anneal.num_cycles
    >>> cfg["early_stopping_patience"] # remapped from training.yaml early_stopping.patience
    """

    merged: dict = {}
    for cfg in configs:
        flat = _flatten(cfg)
        for k, v in flat.items():
            canonical = _MERGE_MAP.get(k, k)
            merged[canonical] = v

    return merged


def merge_configs_with_search_space(*configs: dict, search_space: dict) -> dict:
    """
    Merges fixed configs and attaches the search space as a sub-dict under
    the key "search_space". Used by sweep entry points so that objective.py
    can access both the fixed runtime config and the sweep ranges in one object.

    Parameters
    ----------
    *configs     : Fixed config dicts (data, training, loss).
    search_space : Dict loaded from search_space.yaml.

    Returns
    -------
    dict : Merged flat config with search_space attached.
    """

    merged = merge_configs(*configs)
    merged["search_space"] = search_space
    return merged

# [END]