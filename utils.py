import logging
import ctypes
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

def get_monitor_refresh_rate():
  try:
    user32 = ctypes.windll.user32
    dc = user32.GetDC(0)
    refresh_rate = ctypes.windll.gdi32.GetDeviceCaps(dc, 116)
    user32.ReleaseDC(0, dc)
    return refresh_rate if refresh_rate > 0 else 60
  except Exception:
    logging.warning("No monitor refresh rate detected, setting to 60 Hz")
    return 60
