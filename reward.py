import numpy as np

def get_speed_limit_kmh(speed_limit_kmh: float, speed_config: dict) -> float:
  """Resolve a usable speed limit in km/h for reward shaping."""
  max_forward_speed_kmh = float(speed_config['max_forward_speed_kmh'])

  if speed_limit_kmh is None or speed_limit_kmh <= 0.0:
    speed_limit_kmh = max_forward_speed_kmh

  return float(np.clip(speed_limit_kmh, 0.0, max_forward_speed_kmh))

def compute_target_speed(light_state: str, distance_to_stop: float,
                         speed_limit_kmh: float, speed_config: dict,
                         traffic_lights_enabled: bool = True) -> float:
  """Compute reward target speed using speed limit and traffic light context.

  Returns:
    Target speed in km/h that the vehicle should be traveling at
  """
  effective_limit_kmh = get_speed_limit_kmh(speed_limit_kmh, speed_config)

  max_forward_speed_kmh = float(speed_config['max_forward_speed_kmh'])
  target_fraction = float(speed_config['target_fraction'])

  cruise_target_kmh = float(np.clip(
      effective_limit_kmh * target_fraction,
      0.0,
      max_forward_speed_kmh,
  ))

  if not traffic_lights_enabled:
    return cruise_target_kmh

  if light_state not in ('red', 'yellow'):
    return cruise_target_kmh

  ramp_m_per_kmh = float(speed_config['red_light_ramp_m_per_kmh'])
  ramp_distance_m = ramp_m_per_kmh * cruise_target_kmh
  if ramp_distance_m <= 0.0:
    return 0.0

  if distance_to_stop >= 0.0:
    return 0.0

  distance_ahead = abs(distance_to_stop)
  ramp_scale = float(np.clip(distance_ahead / ramp_distance_m, 0.0, 1.0))
  return cruise_target_kmh * ramp_scale

def is_holding_at_red(state: dict | None, stop_reward_config: dict) -> bool:
  """Return whether a state qualifies for stop-hold reward."""
  if not isinstance(state, dict):
    return False
  if not bool(state.get('traffic_lights_enabled', True)):
    return False
  if bool(state.get('traffic_light_violation', False)):
    return False
  if state.get('traffic_light_state', 'none') not in ('red', 'yellow'):
    return False

  distance_to_stop = state.get('distance_to_stop')
  if distance_to_stop is None:
    return False

  range_cfg = stop_reward_config['stop_distance_range']
  zone_min_m = float(range_cfg['min_m'])
  zone_max_m = float(range_cfg['max_m'])
  if not (zone_min_m <= float(distance_to_stop) <= zone_max_m):
    return False

  max_hold_speed_kmh = float(stop_reward_config['max_hold_speed_kmh'])
  return float(state.get('speed', 0.0)) <= max_hold_speed_kmh

def compute_reward_with_components(state: dict, action: np.ndarray, prev_state: dict,
                                   prev_action: np.ndarray, weights: dict, speed_config: dict,
                                   stop_reward_config: dict | None = None) -> tuple:
  """Compute dense reward and return a breakdown of signed component contributions."""
  stop_reward_config = dict(stop_reward_config)
  components = {
      'waypoint_progress': 0.0,
      'target_speed_compliance': 0.0,
      'lane_deviation': 0.0,
      'smoothness_steer': 0.0,
      'smoothness_accel_brake': 0.0,
      'collision': 0.0,
      'traffic_light_violation': 0.0,
      'holding_at_red': 0.0,
      'success': 0.0,
  }

  # Waypoint progress is based on the change in distance to the current waypoint
  if prev_state is not None:
    prev_wp_dist = prev_state.get('waypoint_distance')
    curr_wp_dist = state.get('waypoint_distance')
    prev_wp_idx = prev_state.get('current_waypoint_idx', 0)
    curr_wp_idx = state.get('current_waypoint_idx', 0)
    if prev_wp_dist is not None and curr_wp_dist is not None:
      if curr_wp_idx == prev_wp_idx:
        waypoint_progress = prev_wp_dist - curr_wp_dist
        components['waypoint_progress'] = (
            weights['waypoint_progress'] * waypoint_progress)

  # Traffic light compliance is based on the error between the target speed and actual speed.
  traffic_lights_enabled = bool(state.get('traffic_lights_enabled', True))
  traffic_light_violation = state.get('traffic_light_violation', False)
  speed_error_kmh = min(abs(state.get('speed_error_kmh', 0.0)), 20.0)
  components['target_speed_compliance'] = (
      -weights['target_speed_compliance'] * speed_error_kmh)

  lane_error_abs = abs(state.get('lane_error_signed', 0.0))
  components['lane_deviation'] = -weights['lane_deviation'] * lane_error_abs

  # Smoothness penalties compare the current action to the previous one.
  if prev_state is not None and prev_action is not None:
    steer_change = abs(action[0] - prev_action[0])
    accel_brake_change = abs(action[1] - prev_action[1])
    components['smoothness_steer'] = - \
        weights['smoothness_steer'] * steer_change
    components['smoothness_accel_brake'] = - \
        weights['smoothness_accel_brake'] * accel_brake_change

  # One time penalties are applied when certain events occur, such as collisions or traffic light violations.
  if bool(state.get('just_collided', False)):
    components['collision'] = -weights['collision']

  if traffic_lights_enabled and bool(state.get('just_red_light_violation', False)):
    components['traffic_light_violation'] = -weights['traffic_light_violation']

  # Holding correctly at a red or yellow light gives a small sustained reward.
  current_is_valid_stop_hold = is_holding_at_red(
      state, stop_reward_config)
  if current_is_valid_stop_hold:
    components['holding_at_red'] = float(stop_reward_config['sustain_per_step'])

  # One-time success reward for just reaching the destination.
  if bool(state.get('just_reached_destination', False)):
    components['success'] = weights['success']

  reward = float(sum(components.values()))
  return reward, components

def compute_reward(state: dict, action: np.ndarray, prev_state: dict,
                   prev_action: np.ndarray, weights: dict, speed_config: dict,
                   stop_reward_config: dict | None = None) -> float:
  """Compute dense reward from state/action."""
  reward, _ = compute_reward_with_components(
      state, action, prev_state, prev_action, weights, speed_config,
      stop_reward_config)
  return reward
