"""Load YAML config files for the benchmark."""

import yaml


def load_yaml(path):
    """Read a YAML file into a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_path):
    """Load experiment YAML and inline its referenced model config."""
    cfg = load_yaml(experiment_path)
    model_cfg_path = cfg.get("model_config")
    if model_cfg_path:
        cfg["model"] = load_yaml(model_cfg_path)
    return cfg
