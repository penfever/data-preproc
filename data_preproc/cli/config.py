"""Configuration loading and processing."""

import json
import os
import tempfile
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
import yaml

from data_preproc.utils.dict import DictDefault
from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)


def check_remote_config(config: Union[str, Path]) -> Union[str, Path]:
    """
    Downloads remote configuration if URL provided.

    Args:
        config: Local path or HTTPS URL to a YAML or JSON file.

    Returns:
        Either the original config if it's not a valid HTTPS URL, or the path to the
        downloaded remote config.
    """
    # Check if the config is a valid HTTPS URL
    if not (isinstance(config, str) and config.startswith("https://")):
        return config

    filename = os.path.basename(urlparse(config).path)
    temp_dir = tempfile.mkdtemp()

    try:
        response = requests.get(config, timeout=30)
        response.raise_for_status()

        content = response.content
        # Verify it's valid YAML
        yaml.safe_load(content)

        # Write the content to a file
        output_path = Path(temp_dir) / filename
        with open(output_path, "wb") as file:
            file.write(content)
        LOG.info(f"Downloaded config from {config}")
        return output_path

    except Exception as err:
        raise RuntimeError(f"Failed to download {config}: {err}") from err


def load_cfg(
    config: Union[str, Path] = Path("config.yaml"),
    **kwargs,
) -> DictDefault:
    """
    Loads configuration from YAML file.

    Args:
        config: Path to config YAML file.
        kwargs: Additional config overrides.

    Returns:
        Loaded configuration as DictDefault.
    """
    config = check_remote_config(config)
    
    # Load the config file
    with open(config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    
    # Apply any kwargs overrides
    cfg.update(kwargs)
    
    # Convert to DictDefault for attribute access
    cfg = DictDefault(cfg)
    
    # Normalize paths
    if cfg.get("output_dir"):
        cfg.output_dir = Path(cfg.output_dir).resolve()
    
    return cfg