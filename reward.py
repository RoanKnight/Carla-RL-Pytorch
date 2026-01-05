import numpy as np
from utils import load_config

# Load configuration settings
CONFIG = load_config('config/phase1.yaml')
WEIGHTS = CONFIG['reward_weights']
TARGET_SPEED_MIN = CONFIG['speed_targets']['min']
TARGET_SPEED_MAX = CONFIG['speed_targets']['max']

def compute_reward(state: dict, action: np.ndarray, prev_state: dict = None) -> float:
  """Compute dense reward from state/action based on current state, action, and previous state. Returns the total reward as a float."""
  reward = 0.0

  # Progress toward destination
  if prev_state is not None and state['distance_to_destination'] is not None:
    prev_dist = prev_state.get('distance_to_destination')
    if prev_dist is not None:
      progress = prev_dist - state['distance_to_destination']
      reward += WEIGHTS['progress'] * progress

  # Speed control to stay within the targeted speed range
  speed = state.get('speed', 0.0)
  if TARGET_SPEED_MIN <= speed <= TARGET_SPEED_MAX:
    reward += WEIGHTS['speed_bonus']
  elif speed < TARGET_SPEED_MIN:
    reward -= WEIGHTS['speed_penalty'] * (TARGET_SPEED_MIN - speed)
  else:
    reward -= WEIGHTS['speed_penalty'] * (speed - TARGET_SPEED_MAX)

  # Lane centering penalty
  lane_deviation = state.get('lane_deviation', 0.0)
  reward -= WEIGHTS['lane_deviation'] * lane_deviation

  # Off-road penalty
  if state.get('off_road', False):
    reward -= WEIGHTS['off_road']

  # Steering smoothness to penalize sudden changes in steering angle
  if prev_state is not None:
    prev_action = prev_state.get('action')
    if prev_action is not None:
      steering_change = abs(action[0] - prev_action[0])
      reward -= WEIGHTS['steering_smoothness'] * steering_change

  # Collision penalty
  if state.get('collision', False):
    reward -= WEIGHTS['collision']

  # Success bonus for reaching destination
  dist = state.get('distance_to_destination')
  if dist is not None and dist < 2.0:
    reward += WEIGHTS['success']

  return reward

def get_reward_weights() -> dict:
  """Return current reward weights for logging/debugging."""
  return WEIGHTS.copy()

def get_phase_config() -> dict:
  """Return full configuration for current phase"""
  return CONFIG.copy()
