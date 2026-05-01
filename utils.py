import glob
import logging
import os
import re
from pathlib import Path

import yaml

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

def find_latest_checkpoint(checkpoint_dir: str,
                           name_prefix: str = "sac_carla") -> tuple:
  """Find the most recent checkpoint in a directory.

  Args:
    checkpoint_dir: Path to checkpoint directory
    name_prefix: Checkpoint filename prefix

  Returns:
    Tuple of (checkpoint_path, checkpoint_steps) or (None, None) if no checkpoint exists
  """
  if not os.path.exists(checkpoint_dir):
    return None, None

  checkpoint_files = glob.glob(os.path.join(
      checkpoint_dir, f"{name_prefix}_*_steps.zip"))
  if not checkpoint_files:
    return None, None

  filename_regex = re.compile(
      rf"^{re.escape(name_prefix)}_(\d+)_steps\.zip$")
  parsed_checkpoints = []
  for checkpoint_path in checkpoint_files:
    filename = os.path.basename(checkpoint_path)
    match = filename_regex.match(filename)
    if match is None:
      continue
    parsed_checkpoints.append((int(match.group(1)), checkpoint_path))

  if not parsed_checkpoints:
    return None, None

  parsed_checkpoints.sort(key=lambda item: item[0])
  checkpoint_steps, latest_checkpoint = parsed_checkpoints[-1]

  return latest_checkpoint, checkpoint_steps
