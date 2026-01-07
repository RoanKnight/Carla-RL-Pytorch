import logging
import yaml
import glob
import os
from pathlib import Path

def setup_logging(level=logging.INFO):
  """Setup logging for the application."""
  logging.basicConfig(
      format='[%(levelname)s] %(asctime)s - %(message)s',
      datefmt='%H:%M:%S',
      level=level
  )

def load_config(config_path='config/base.yaml'):
  """Load configuration from YAML file."""
  config_file = Path(config_path)
  if not config_file.exists():
    raise FileNotFoundError(f"Config file not found: {config_path}")
  with open(config_file, 'r') as f:
    return yaml.safe_load(f)

def find_latest_checkpoint(checkpoint_dir: str) -> tuple:
  """Find the most recent checkpoint in a directory.

  Args:
    checkpoint_dir: Path to checkpoint directory

  Returns:
    Tuple of (checkpoint_path, checkpoint_steps) or (None, None) if no checkpoint exists
  """
  if not os.path.exists(checkpoint_dir):
    return None, None

  checkpoint_files = glob.glob(os.path.join(
      checkpoint_dir, "sac_carla_*_steps.zip"))
  if not checkpoint_files:
    return None, None

  # Sort by steps (extract number from filename)
  checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))
  latest_checkpoint = checkpoint_files[-1]
  checkpoint_steps = int(latest_checkpoint.split('_')[-2])

  return latest_checkpoint, checkpoint_steps
