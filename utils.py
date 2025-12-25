import logging
import yaml
from pathlib import Path

# Function to setup logging
def setup_logging(level=logging.INFO):
  logging.basicConfig(
      format='[%(levelname)s] %(message)s',
      level=level
  )

# Function to load configuration from YAML file, returns a dictionary with the config
def load_config(config_path='config/base.yaml'):
  config_file = Path(config_path)
  if not config_file.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
  with open(config_file, 'r') as f:
    return yaml.safe_load(f)
