import numpy as np

def compute_reward(state: dict, action: np.ndarray, prev_state: dict, 
                   prev_action: np.ndarray, weights: dict, speed_targets: dict) -> float:
  """Compute dense reward from state/action. All parameters are explicit (no global state).
  
  Args:
    state: Current vehicle state (speed, distance_to_destination, collision, etc.)
    action: Current action (steering, throttle/brake)
    prev_state: Previous vehicle state
    prev_action: Previous action (for smoothness penalty)
    weights: Reward weight dictionary from config
    speed_targets: Speed target dict with 'min' and 'max' keys
  
  Returns:
    Total reward as float
  """
  reward = 0.0
  
  # Progress toward destination
  if prev_state is not None:
    prev_dist = prev_state.get('distance_to_destination')
    curr_dist = state.get('distance_to_destination')
    if prev_dist is not None and curr_dist is not None:
      progress = prev_dist - curr_dist
      reward += weights['progress'] * progress

  # Speed control to stay within the targeted speed range
  speed = state.get('speed', 0.0)
  speed_min = speed_targets['min']
  speed_max = speed_targets['max']
  if speed_min <= speed <= speed_max:
    reward += weights['speed_bonus']
  elif speed < speed_min:
    reward -= weights['speed_penalty'] * (speed_min - speed)
  else:
    reward -= weights['speed_penalty'] * (speed - speed_max)

  # Lane centering penalty
  lane_deviation = state.get('lane_deviation', 0.0)
  reward -= weights['lane_deviation'] * lane_deviation

  # Off-road penalty
  if state.get('off_road', False):
    reward -= weights['off_road']

  # Steering smoothness to penalize sudden changes in steering angle
  if prev_state is not None and prev_action is not None:
    steering_change = abs(action[0] - prev_action[0])
    reward -= weights['steering_smoothness'] * steering_change

  # Collision penalty
  if state.get('collision', False):
    reward -= weights['collision']

  # Success bonus for reaching destination
  dist = state.get('distance_to_destination')
  if dist is not None and dist < 2.0:
    reward += weights['success']

  return reward
