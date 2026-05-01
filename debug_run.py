import sys

sys.path.append(r'C:\Carla\PythonAPI\carla')

import logging
from pathlib import Path
from typing import Optional

import carla
import cv2
import numpy as np
import torch

from augmentation import RandomShiftAug
from environment import CarlaEnv
from reward import compute_reward
from sac_carla import get_agent_type, get_checkpoint_dir, load_agent
from utils import find_latest_checkpoint, load_config, setup_logging

DEBUG_CONFIG = {
    'map': 'Town02_Opt',
    'weather': 'clear_noon',
    'max_steps': 10000,
    'episodes': 5,
    'control_mode': 'idle',  # 'idle' or 'rl'
    'agent_path': None,
    'traffic_lights_enabled_override': True,
    'log_every_n_steps': 50,
    'save_image_on_first_step': True,
    'save_drq_image_on_first_step': True,
    'camera_snapshot_max_wait_steps': 30,
    'image_dir': 'debug_images',
    'base_reset_seed': 12345,
    'spectator_distance_behind': 6.0,
    'spectator_height_above': 2.5,
    'drq_pad': 40,
}

_OBS_KEYS = ('goal', 'traffic_light', 'distance_to_stop', 'speed', 'target_speed', 'speed_error', 'last_action', 'lane_error_signed')
_SCALAR_OBS = {
    'distance_to_stop': ('distance_to_stop', 'm', True, 1, lambda e: max(float(getattr(e, 'traffic_light_context_range_m', 1.0)), 1.0)),
    'speed': ('speed', 'kmh', False, 1, lambda e: max(float(e._get_speed_observation_clip_kmh()), 1.0)),
    'target_speed': ('reward_target_speed_kmh', 'kmh', False, 1, lambda e: max(float(e._get_speed_observation_clip_kmh()), 1.0)),
    'speed_error': ('speed_error_kmh', 'kmh', True, 1, lambda e: max(float(e._get_speed_observation_clip_kmh()), 1.0)),
    'lane_error_signed': ('lane_error_signed', 'm', True, 2, lambda e: max(float(e.config.get('lane_detection', {}).get('max_lane_deviation', 5.0)), 1e-6)),
}

def _to_vec(value):
  try:
    return np.asarray(value, dtype=np.float32).reshape(-1)
  except (TypeError, ValueError):
    return np.asarray([], dtype=np.float32)

def _format_obs_value(value) -> str:
  array = np.asarray(value)
  return repr(array.item()) if array.ndim == 0 else np.array2string(array, precision=3, floatmode='maxprec_equal')

def _format_obs_annotation(key: str, value, env: Optional[CarlaEnv] = None, state: Optional[dict] = None) -> str:
  state = state or {}
  if key in _SCALAR_OBS:
    state_key, unit, signed, precision, scale_getter = _SCALAR_OBS[key]
    vector = _to_vec(value)
    if state_key not in state and (env is None or vector.size == 0):
      return ''
    real_value = float(state[state_key]) if state_key in state else float(vector[0]) * float(scale_getter(env))
    return f"({real_value:{'+' if signed else ''}.{precision}f}{unit})"

  vector = _to_vec(value)
  if key == 'goal' and vector.size >= 2:
    return f"(fwd={vector[0] * 10.0:+.1f}m, lat={vector[1] * 10.0:+.1f}m)"
  if key == 'traffic_light' and vector.size >= 4:
    return f"({('none', 'red', 'yellow', 'green')[int(np.argmax(vector[:4]))]})"
  if key == 'last_action' and vector.size >= 2:
    steer, accel_brake = float(vector[0]), float(vector[1])
    return f"(steer={steer:+.2f}, accel_brake={accel_brake:+.2f}, throttle={max(accel_brake, 0.0):.2f}, brake={max(-accel_brake, 0.0):.2f})"
  return ''

def _format_observation(obs: dict, env: Optional[CarlaEnv] = None, state: Optional[dict] = None) -> str:
  if obs is None:
    return 'obs=<none>'

  parts = []
  for key in _OBS_KEYS:
    if key in obs:
      annotation = _format_obs_annotation(key, obs[key], env=env, state=state)
      parts.append(f"{key}={_format_obs_value(obs[key])}{' ' + annotation if annotation else ''}")

  if 'last_action' not in obs and 'prev_action' in obs:
    annotation = _format_obs_annotation('last_action', obs['prev_action'], env=env, state=state)
    parts.append(f"prev_action={_format_obs_value(obs['prev_action'])}{' ' + annotation if annotation else ''}")

  if 'image' in obs:
    image = np.asarray(obs['image'])
    parts.append(f"image=shape{image.shape}, dtype={image.dtype}, min={int(image.min())}, max={int(image.max())}")

  return 'obs={' + ', '.join(parts) + '}'

def _format_reward_components(components: dict) -> str:
  top = [] if not isinstance(components, dict) else sorted(
      ((name, value) for name, value in components.items() if abs(value) >= 0.01),
      key=lambda item: abs(item[1]), reverse=True,
  )[:5]
  return '' if not top else ' | ' + ', '.join(f"{name}={value:+.2f}" for name, value in top)

def _save_camera_image(episode: int, env: CarlaEnv, config: dict) -> bool:
  if not getattr(env, 'camera_enabled', True) or env._rgb_image is None:
    return False
  out_dir = Path(config['image_dir'])
  out_dir.mkdir(parents=True, exist_ok=True)
  image_path = out_dir / f'ep{episode:03d}.png'
  cv2.imwrite(str(image_path), cv2.cvtColor(env._rgb_image, cv2.COLOR_RGB2BGR))
  logging.info("Saved camera image: %s", image_path)
  return True

def _save_drq_image(episode: int, env: CarlaEnv, config: dict) -> bool:
  if not getattr(env, 'camera_enabled', True) or env._rgb_image is None:
    return False
  image_tensor = torch.from_numpy(env._rgb_image.copy()).permute(2, 0, 1).unsqueeze(0).float()
  shifted = RandomShiftAug(pad=config['drq_pad'])(image_tensor).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
  out_dir = Path(config['image_dir'])
  out_dir.mkdir(parents=True, exist_ok=True)
  image_path = out_dir / f'ep{episode:03d}_drq.png'
  cv2.imwrite(str(image_path), cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR))
  logging.info("Saved DrQ-shifted image: %s", image_path)
  return True

def _log_route_stop_waypoint(env: CarlaEnv, state: dict, prefix: str) -> None:
  tracked_stop_waypoint = env._tracked_stop_waypoint
  tracked_tl_id = int(state.get('tracked_traffic_light_id', -1))
  if tracked_stop_waypoint is None:
    logging.info("%s: none (tl_id=%s)", prefix, tracked_tl_id if tracked_tl_id >= 0 else "none")
    return
  logging.info("%s: tl_id=%s state=%s dist=%5.1fm", prefix, tracked_tl_id if tracked_tl_id >= 0 else "none", state.get('traffic_light_state', 'none'), state.get('distance_to_stop', 999.0))

def _draw_route_stop_waypoint(env: CarlaEnv) -> None:
  tracked_stop_waypoint = env._tracked_stop_waypoint
  if tracked_stop_waypoint is not None:
    stop_location = tracked_stop_waypoint.transform.location
    env.world.debug.draw_line(stop_location, stop_location + carla.Location(z=1.5), thickness=0.20, color=carla.Color(255, 255, 0), life_time=0.1)

def _log_step(step: int, state: dict, reward: float, action: np.ndarray, obs: dict, env: Optional[CarlaEnv] = None) -> None:
  logging.debug(
      f"step={step:4d} | tl_active={state.get('traffic_lights_enabled', True)} | "
      f"{_format_observation(obs, env=env, state=state)} | "
      f"action=[steer={action[0]:+.2f}, accel_brake={action[1]:+.2f}] | "
      f"reward={reward:+7.2f}{_format_reward_components(state.get('reward_components', {}))}"
  )

def _print_waypoint_vectors(env: CarlaEnv) -> None:
  if len(env.route) == 0:
    logging.info('No route available.')
    return

  transform = env.vehicle.get_transform()
  location = transform.location
  yaw = np.radians(transform.rotation.yaw)
  forward_x, forward_y = np.cos(yaw), np.sin(yaw)
  right_x, right_y = np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)
  start_idx = env.current_waypoint_idx
  logging.info("Waypoint vectors (first %d starting from WP %d of %d):", min(10, len(env.route) - start_idx), start_idx, len(env.route))

  for idx in range(start_idx, min(start_idx + 10, len(env.route))):
    waypoint_location = env._route_waypoint_locations[idx]
    dx, dy = waypoint_location.x - location.x, waypoint_location.y - location.y
    logging.info("  WP %2d: (%+7.2f fwd, %+7.2f lat)", idx, dx * forward_x + dy * forward_y, dx * right_x + dy * right_y)

def _update_transition_logs(step: int, info: dict, env: CarlaEnv, prev_waypoint_timeout_active: bool, prev_tl_state: str, prev_tracked_tl_id: int):
  current_waypoint_timeout_steps = int(info.get('waypoint_timeout_steps', 0))
  current_waypoint_timeout_active = current_waypoint_timeout_steps > 0
  current_tl_state = info.get('traffic_light_state', 'none')
  current_tracked_tl_id = int(info.get('tracked_traffic_light_id', -1))

  if current_waypoint_timeout_active != prev_waypoint_timeout_active and current_waypoint_timeout_active:
    logging.info(
        "Step %4d: waypoint-timeout tracking started (dist_to_wp=%.1fm, counter=%d/%d, threshold=%.1fm)",
        step, float(info.get('waypoint_distance', 0.0)), current_waypoint_timeout_steps,
        int(getattr(env, 'waypoint_distance_timeout_steps', 0)),
        float(getattr(env, 'waypoint_distance_timeout_threshold_meters', 0.0)),
    )
  elif current_waypoint_timeout_active != prev_waypoint_timeout_active:
    logging.info("Step %4d: waypoint-timeout tracking cleared (dist_to_wp=%.1fm)", step, float(info.get('waypoint_distance', 0.0)))

  if current_tracked_tl_id != prev_tracked_tl_id:
    logging.info(
        "Step %4d: tracked_tl %s -> %s (state=%s, dist=%.1fm)", step,
        prev_tracked_tl_id if prev_tracked_tl_id >= 0 else "none",
        current_tracked_tl_id if current_tracked_tl_id >= 0 else "none",
        current_tl_state, float(info.get('distance_to_stop', 999.0)),
    )
  if current_tl_state != prev_tl_state:
    logging.info(
        "Step %4d: traffic_light %s -> %s at %.1fm (tl_id=%s)", step, prev_tl_state, current_tl_state,
        info.get('distance_to_stop', 999.0), current_tracked_tl_id if current_tracked_tl_id >= 0 else "none",
    )

  return current_waypoint_timeout_active, current_tl_state, current_tracked_tl_id

def run(config: Optional[dict] = None) -> None:
  debug_config = dict(DEBUG_CONFIG)
  if config:
    debug_config.update(config)

  setup_logging(level=logging.DEBUG)
  control_mode = debug_config['control_mode']
  if control_mode not in {'idle', 'rl'}:
    raise ValueError(f"Unsupported control_mode='{control_mode}'. Use 'idle' or 'rl'.")

  training_config = load_config('config/training.yaml')
  agent_variant = get_agent_type()
  checkpoint_dir = get_checkpoint_dir(training_config, agent_variant)

  logging.info('=' * 70)
  logging.info('DEBUG RUN')
  logging.info("map=%s weather=%s steps=%s episodes=%s control_mode=%s", debug_config['map'], debug_config['weather'], debug_config['max_steps'], debug_config['episodes'], control_mode)
  logging.info('=' * 70)

  env = CarlaEnv(config_path='config/base.yaml', phase_config_path='config/training.yaml', reward_fn=compute_reward, mode='test')
  available_maps = [map_name.split('/')[-1] for map_name in env.client.get_available_maps()]
  if debug_config['map'] not in available_maps:
    raise ValueError(f"Map '{debug_config['map']}' not found. Available: {sorted(available_maps)}")

  env.phase_config['distribution'] = {'maps': [debug_config['map']], 'weathers': [debug_config['weather']]}
  env.map_change_frequency = 1
  env.max_steps = debug_config['max_steps']
  if debug_config['traffic_lights_enabled_override'] is not None:
    env.set_traffic_lights_enabled(bool(debug_config['traffic_lights_enabled_override']))
  logging.debug("Overrides: map=%s weather=%s mode=%s traffic_lights_override=%s", debug_config['map'], debug_config['weather'], control_mode, debug_config['traffic_lights_enabled_override'])

  agent = None
  if control_mode == 'rl':
    model_path = debug_config['agent_path']
    if model_path is None:
      model_path, steps = find_latest_checkpoint(checkpoint_dir)
      if model_path is None:
        raise FileNotFoundError("control_mode='rl' but no checkpoint found in " f"{checkpoint_dir}/")
      logging.info("Auto-detected checkpoint: %s (%s steps)", model_path, steps)
    else:
      logging.info("Loading checkpoint: %s", model_path)
    agent = load_agent(model_path, env=env, agent_variant=agent_variant)

  try:
    for episode in range(1, debug_config['episodes'] + 1):
      episode_seed = debug_config['base_reset_seed'] + episode - 1
      logging.info("\n%s", '-' * 70)
      logging.info("Episode %d/%d seed=%d", episode, debug_config['episodes'], episode_seed)

      obs, info = env.reset(seed=episode_seed)
      logging.info("Reset: map=%s weather=%s dist=%.1fm traffic_lights_enabled=%s", info['map'], info['weather'], info['initial_distance'], info.get('traffic_lights_enabled', True))
      logging.info("Initial observation: %s", _format_observation(obs, env=env, state=info))
      logging.info("Initial current_waypoint_idx: %s", info.get('current_waypoint_idx', -1))
      _log_route_stop_waypoint(env, info, 'Initial route stop waypoint')

      spectator = env.world.get_spectator()
      vehicle_transform = env.vehicle.get_transform()
      forward = vehicle_transform.get_forward_vector()
      spectator.set_transform(carla.Transform(vehicle_transform.location + carla.Location(
          x=-forward.x * debug_config['spectator_distance_behind'],
          y=-forward.y * debug_config['spectator_distance_behind'],
          z=debug_config['spectator_height_above'],
      ), vehicle_transform.rotation))
      _print_waypoint_vectors(env)

      done = False
      step = 0
      pending_camera_snapshot = debug_config['save_image_on_first_step'] and getattr(env, 'camera_enabled', False)
      pending_drq_snapshot = debug_config['save_drq_image_on_first_step'] and getattr(env, 'camera_enabled', False)
      prev_waypoint_timeout_active = int(info.get('waypoint_timeout_steps', 0)) > 0
      prev_tl_state = info.get('traffic_light_state', 'none')
      prev_tracked_tl_id = int(info.get('tracked_traffic_light_id', -1))

      while not done:
        action = np.asarray(agent.predict(obs, deterministic=True)[0], dtype=np.float32) if agent is not None else np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if (pending_camera_snapshot or pending_drq_snapshot) and env._rgb_image is not None:
          pending_camera_snapshot = pending_camera_snapshot and not _save_camera_image(episode, env, debug_config)
          pending_drq_snapshot = pending_drq_snapshot and not _save_drq_image(episode, env, debug_config)
        elif (pending_camera_snapshot or pending_drq_snapshot) and step >= debug_config['camera_snapshot_max_wait_steps']:
          logging.warning("Camera image not available by step %d, skipping first-frame save for this episode.", step)
          pending_camera_snapshot = pending_drq_snapshot = False

        if spectator is not None and control_mode == 'rl':
          spectator_transform = spectator.get_transform()
          vehicle_transform = env.vehicle.get_transform()
          forward = spectator_transform.get_forward_vector()
          location = vehicle_transform.location - forward * debug_config['spectator_distance_behind'] + carla.Vector3D(z=debug_config['spectator_height_above'])
          spectator.set_transform(carla.Transform(location, spectator_transform.rotation))

        destination = env.spawn_points[env.dest_idx].location
        env.world.debug.draw_line(destination, destination + carla.Location(z=150), thickness=0.4, color=carla.Color(255, 0, 0), life_time=0.1)
        for waypoint_location in env._route_waypoint_locations:
          env.world.debug.draw_line(waypoint_location, waypoint_location + carla.Location(z=1.5), thickness=0.15, color=carla.Color(0, 255, 0), life_time=0.1)
        _draw_route_stop_waypoint(env)

        if step % debug_config['log_every_n_steps'] == 0 or done:
          _log_step(step, info, reward, action, obs, env=env)

        prev_waypoint_timeout_active, prev_tl_state, prev_tracked_tl_id = _update_transition_logs(
            step, info, env, prev_waypoint_timeout_active, prev_tl_state, prev_tracked_tl_id)

      logging.info(
          "Episode %d ended: steps=%d collision=%s red_violation=%s waypoint_timeout_steps=%d waypoint_distance=%.1fm dest=%.1fm",
          episode, step, info.get('collision', False), info.get('traffic_light_violation', False),
          int(info.get('waypoint_timeout_steps', 0)), float(info.get('waypoint_distance', 0.0)),
          info.get('distance_to_destination', 0.0),
      )

  except KeyboardInterrupt:
    logging.info('Interrupted by user.')
  finally:
    env.close()
    logging.info('Closed.')

if __name__ == '__main__':
  run()
