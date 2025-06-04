import numpy as np
import pickle
import yaml
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def load_pickle(file_path: str) -> Any:
    """Load .pkl configuration file"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise

def load_npz(file_path: str) -> Dict[str, np.ndarray]:
    """Load .npz configuration file"""
    try:
        return np.load(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise