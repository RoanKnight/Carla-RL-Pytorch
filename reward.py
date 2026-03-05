import numpy as np

def compute_target_speed(light_state: str, distance_to_stop: float, speed_limit_kmh: float) -> float:
  """Compute target speed based on traffic light state and distance to stop line.

  Returns:
    Target speed in km/h that the vehicle should be traveling at
  """
  if light_state not in ('red', 'yellow'):
    return speed_limit_kmh

  if light_state == 'red':
    if distance_to_stop >= -2.0:
      return 0.0
    if distance_to_stop < -30.0:
      return speed_limit_kmh
    return speed_limit_kmh * (abs(distance_to_stop) - 2.0) / 28.0

  # Yellow light
  if distance_to_stop >= -5.0:
    return speed_limit_kmh * 0.7
  if distance_to_stop < -30.0:
    return speed_limit_kmh
  if distance_to_stop < -15.0:
    return speed_limit_kmh * (0.5 + 0.5 * (abs(distance_to_stop) - 15.0) / 15.0)
  return speed_limit_kmh * 0.5

def compute_reward(state: dict, action: np.ndarray, prev_state: dict,
                   prev_action: np.ndarray, weights: dict, min_speed_fraction: float) -> float:
  """Compute dense reward from state/action."""
  reward = 0.0

  # Waypoint progress reward
  if prev_state is not None:
    prev_wp_dist = prev_state.get('waypoint_distance')
    curr_wp_dist = state.get('waypoint_distance')
    if prev_wp_dist is not None and curr_wp_dist is not None:
      waypoint_progress = prev_wp_dist - curr_wp_dist
      reward += weights['waypoint_progress'] * waypoint_progress

  # Waypoint reached reward
  if prev_state is not None:
    prev_wp_idx = prev_state.get('current_waypoint_idx', 0)
    curr_wp_idx = state.get('current_waypoint_idx', 0)
    waypoints_advanced = curr_wp_idx - prev_wp_idx
    if waypoints_advanced > 0:
      reward += weights['waypoint_reached'] * waypoints_advanced

  speed = state.get('speed', 0.0)
  speed_limit_kmh = state.get('speed_limit_kmh', 50.0)
  speed_min = speed_limit_kmh * min_speed_fraction
  speed_max = speed_limit_kmh
  if speed_min <= speed <= speed_max:
    reward += weights['speed_bonus']
  elif speed < speed_min:
    reward -= weights['speed_penalty'] * (speed_min - speed)
  else:
    reward -= weights['speed_penalty'] * (speed - speed_max)

  lane_deviation = state.get('lane_deviation', 0.0)
  reward -= weights['lane_deviation'] * lane_deviation

  if state.get('off_road', False):
    reward -= weights['off_road']

  if state.get('lane_invasion', False):
    reward -= weights['lane_invasion']

  if prev_state is not None and prev_action is not None:
    steering_change = abs(action[0] - prev_action[0])
    reward -= weights['steering_smoothness'] * steering_change

  if prev_state is not None and prev_action is not None:
    throttle_brake_change = abs(action[1] - prev_action[1])
    reward -= weights['throttle_brake_smoothness'] * throttle_brake_change

  if state.get('collision', False):
    reward -= weights['collision']

  vehicle_dist = state.get('nearest_vehicle_dist', 50.0)
  if vehicle_dist < 15.0:
    reward -= weights['proximity_vehicle'] * (1.0 - vehicle_dist / 15.0)

  ped_dist = state.get('nearest_pedestrian_dist', 20.0)
  if ped_dist < 8.0:
    reward -= weights['proximity_pedestrian'] * (1.0 - ped_dist / 8.0)

  ttc = state.get('ttc', 10.0)
  if ttc < 4.0:
    reward -= weights['ttc_penalty'] * (1.0 - ttc / 4.0)

  light_state = state.get('traffic_light_state', 'none')
  distance_to_stop = state.get('distance_to_stop', 999.0)
  red_light_violation = (
      light_state == 'red' and
      distance_to_stop > 0.0 and
      speed > 5.0
  )

  if light_state != 'none':
    target_speed = compute_target_speed(
        light_state, distance_to_stop, speed_limit_kmh)
    speed_error = abs(speed - target_speed)
    reward -= weights['traffic_light_compliance'] * speed_error

    if red_light_violation:
      reward -= weights['red_light_violation']

    if light_state == 'red' and -3.0 < distance_to_stop <= 0.0 and speed < 2.0:
      reward += weights['holding_at_red']

  dist = state.get('distance_to_destination')
  reached_destination = dist is not None and dist < 2.0
  terminal_failure = state.get('collision', False) or red_light_violation
  if reached_destination and not terminal_failure:
    reward += weights['success']

  return reward
