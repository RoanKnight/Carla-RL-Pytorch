import numpy as np

def _get_effective_speed_limit(speed_limit_kmh: float, speed_config: dict) -> float:
  """Resolve a usable, reachable speed limit in km/h for reward shaping."""
  max_forward_speed_kmh = float(
      speed_config.get('max_forward_speed_kmh', 40.0))

  if speed_limit_kmh is None or speed_limit_kmh <= 0.0:
    speed_limit_kmh = max_forward_speed_kmh

  return float(np.clip(speed_limit_kmh, 0.0, max_forward_speed_kmh))

def compute_speed_band(speed_limit_kmh: float, speed_config: dict) -> tuple[float, float, float]:
  """Compute min, target, and max desired speeds from the current road speed limit."""
  effective_limit_kmh = _get_effective_speed_limit(
      speed_limit_kmh, speed_config)
  absolute_min_kmh = float(speed_config.get('absolute_min_kmh', 5.0))
  max_forward_speed_kmh = float(speed_config.get(
      'max_forward_speed_kmh', effective_limit_kmh))
  min_fraction = float(speed_config.get('min_fraction', 0.60))
  target_fraction = float(speed_config.get('target_fraction', 0.90))
  max_fraction = float(speed_config.get('max_fraction', 1.00))

  min_speed_kmh = float(np.clip(
      effective_limit_kmh * min_fraction,
      absolute_min_kmh,
      max_forward_speed_kmh,
  ))
  target_speed_kmh = float(np.clip(
      effective_limit_kmh * target_fraction,
      min_speed_kmh,
      max_forward_speed_kmh,
  ))
  max_speed_kmh = float(np.clip(
      effective_limit_kmh * max_fraction,
      target_speed_kmh,
      max_forward_speed_kmh,
  ))
  return min_speed_kmh, target_speed_kmh, max_speed_kmh

def compute_cruise_speed_target(speed_limit_kmh: float, speed_config: dict) -> float:
  """Compute the normal cruise target speed from the current road speed limit."""
  _, cruise_target_kmh, _ = compute_speed_band(speed_limit_kmh, speed_config)
  return cruise_target_kmh

def compute_target_speed(light_state: str, distance_to_stop: float,
                         speed_limit_kmh: float, speed_config: dict,
                         traffic_lights_enabled: bool = True) -> float:
  """Compute reward target speed using speed limit and traffic light context.

  Returns:
    Target speed in km/h that the vehicle should be traveling at
  """
  cruise_target_kmh = compute_cruise_speed_target(
      speed_limit_kmh, speed_config)

  if not traffic_lights_enabled:
    return cruise_target_kmh

  if light_state not in ('red', 'yellow'):
    return cruise_target_kmh

  ramp_m_per_kmh = float(speed_config.get('red_light_ramp_m_per_kmh', 1.48))
  ramp_distance_m = ramp_m_per_kmh * cruise_target_kmh
  if ramp_distance_m <= 0.0:
    return 0.0

  if distance_to_stop >= 0.0:
    return 0.0

  distance_ahead = abs(distance_to_stop)
  ramp_scale = float(np.clip(distance_ahead / ramp_distance_m, 0.0, 1.0))
  return cruise_target_kmh * ramp_scale

def compute_reward_with_components(state: dict, action: np.ndarray, prev_state: dict,
                                   prev_action: np.ndarray, weights: dict, speed_config: dict) -> tuple:
  """Compute dense reward and return a breakdown of signed component contributions."""
  components = {
      'waypoint_progress': 0.0,
      'target_speed_compliance': 0.0,
      'lane_deviation': 0.0,
      'smoothness_steer': 0.0,
      'smoothness_accel_brake': 0.0,
      'collision': 0.0,
      'traffic_light_violation': 0.0,
      'success': 0.0,
  }

  # Waypoint progress reward
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

  traffic_lights_enabled = bool(state.get('traffic_lights_enabled', True))
  traffic_light_violation = state.get('traffic_light_violation', False)
  speed_error_kmh = min(abs(state.get('speed_error_kmh', 0.0)), 20.0)
  components['target_speed_compliance'] = (
      -weights['target_speed_compliance'] * speed_error_kmh)

  lane_error_abs = abs(state.get('lane_error_signed', 0.0))
  components['lane_deviation'] = -weights['lane_deviation'] * lane_error_abs

  if prev_state is not None and prev_action is not None:
    steer_change = abs(action[0] - prev_action[0])
    accel_brake_change = abs(action[1] - prev_action[1])
    components['smoothness_steer'] = - \
        weights['smoothness_steer'] * steer_change
    components['smoothness_accel_brake'] = - \
        weights['smoothness_accel_brake'] * accel_brake_change

  if state.get('collision', False):
    components['collision'] = -weights['collision']

  if traffic_lights_enabled and traffic_light_violation:
    components['traffic_light_violation'] = -weights['traffic_light_violation']

  dist = state.get('distance_to_destination')
  if dist is not None and dist < 2.0:
    components['success'] = weights['success']

  reward = float(sum(components.values()))
  return reward, components

def compute_reward(state: dict, action: np.ndarray, prev_state: dict,
                   prev_action: np.ndarray, weights: dict, speed_config: dict) -> float:
  """Compute dense reward from state/action."""
  reward, _ = compute_reward_with_components(
      state, action, prev_state, prev_action, weights, speed_config)
  return reward
