import sys
sys.path.append(r'C:\Carla\PythonAPI\carla')

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import carla

from environment import CarlaEnv
from reward import compute_reward
from sac_carla import load_agent
from utils import setup_logging, find_latest_checkpoint
from augmentation import RandomShiftAug

DEBUG_CONFIG = {
    'map': 'Town02',
    'weather': 'clear_noon',
    'max_steps': 5000,
    'episodes': 3,
    'control_mode': 'idle',  # 'idle' or 'rl'
    'agent_path': None,
    'traffic_lights_enabled_override': False,
    'log_every_n_steps': 1,
    'save_image_on_first_step': True,
    'save_drq_image_on_first_step': True,
    'camera_snapshot_max_wait_steps': 30,
    'image_dir': 'debug_images',
    'base_reset_seed': 12345,
    'spectator_distance_behind': 6.0,
    'spectator_height_above': 2.5,
    'drq_pad': 4,
}

def _apply_overrides(env: CarlaEnv, config: dict) -> None:
  env.phase_config['distribution'] = {
      'maps': [config['map']],
      'weathers': [config['weather']],
  }
  env.map_change_frequency = 1
  env.max_steps = config['max_steps']

  if config['traffic_lights_enabled_override'] is not None:
    env.set_traffic_lights_enabled(
        bool(config['traffic_lights_enabled_override']))

  logging.debug(
      "Overrides: map=%s weather=%s mode=%s traffic_lights_override=%s",
      config['map'],
      config['weather'],
      config['control_mode'],
      config['traffic_lights_enabled_override'],
  )

def _resolve_control_mode(config: dict) -> str:
  control_mode = config['control_mode']
  if control_mode not in {'idle', 'rl'}:
    raise ValueError(
        f"Unsupported control_mode='{control_mode}'. Use 'idle' or 'rl'.")
  return control_mode

def _set_spectator_at_vehicle(env: CarlaEnv, spectator, config: dict) -> None:
  vehicle_transform = env.vehicle.get_transform()
  forward = vehicle_transform.get_forward_vector()
  cam_loc = vehicle_transform.location + carla.Location(
      x=-forward.x * config['spectator_distance_behind'],
      y=-forward.y * config['spectator_distance_behind'],
      z=config['spectator_height_above'],
  )
  spectator.set_transform(carla.Transform(cam_loc, vehicle_transform.rotation))

def _format_reward_components(components: dict) -> str:
  if not isinstance(components, dict):
    return ''

  non_zero_components = [
      (name, value) for name, value in components.items() if abs(value) >= 0.01]
  if not non_zero_components:
    return ''

  top_components = sorted(
      non_zero_components, key=lambda item: abs(item[1]), reverse=True)[:5]
  return ' | ' + ', '.join(
      f"{name}={value:+.2f}" for name, value in top_components)

def _format_obs_value(value) -> str:
  array = np.asarray(value)
  if array.ndim == 0:
    return repr(array.item())
  return np.array2string(array, precision=3, floatmode='maxprec_equal')

def _format_observation(obs: dict) -> str:
  if obs is None:
    return 'obs=<none>'

  parts = []
  for key in ('goal', 'traffic_light', 'distance_to_stop', 'speed',
              'target_speed', 'speed_error', 'last_action',
              'lane_error_signed'):
    if key in obs:
      parts.append(f"{key}={_format_obs_value(obs[key])}")

  if 'last_action' not in obs and 'prev_action' in obs:
    parts.append(f"prev_action={_format_obs_value(obs['prev_action'])}")

  if 'image' in obs:
    image = np.asarray(obs['image'])
    parts.append(
        f"image=shape{image.shape}, dtype={image.dtype}, "
        f"min={int(image.min())}, max={int(image.max())}")

  return 'obs={' + ', '.join(parts) + '}'

def _save_camera_image(episode: int, env: CarlaEnv, config: dict) -> bool:
  if not getattr(env, 'camera_enabled', True):
    return False
  if env._rgb_image is None:
    return False

  out_dir = Path(config['image_dir'])
  out_dir.mkdir(parents=True, exist_ok=True)

  image_bgr = cv2.cvtColor(env._rgb_image, cv2.COLOR_RGB2BGR)
  image_path = out_dir / f'ep{episode:03d}.png'
  cv2.imwrite(str(image_path), image_bgr)
  logging.info("Saved camera image: %s", image_path)
  return True

def _save_drq_image(episode: int, env: CarlaEnv, config: dict) -> bool:
  if not getattr(env, 'camera_enabled', True):
    return False
  if env._rgb_image is None:
    return False

  image_rgb = env._rgb_image.copy()
  image_tensor = torch.from_numpy(
      image_rgb).permute(2, 0, 1).unsqueeze(0).float()

  random_shift = RandomShiftAug(pad=config['drq_pad'])
  shifted_tensor = random_shift(image_tensor)

  shifted_image = shifted_tensor.squeeze(
      0).permute(1, 2, 0).numpy().astype(np.uint8)
  shifted_image_bgr = cv2.cvtColor(shifted_image, cv2.COLOR_RGB2BGR)

  out_dir = Path(config['image_dir'])
  out_dir.mkdir(parents=True, exist_ok=True)

  drq_image_path = out_dir / f'ep{episode:03d}_drq.png'
  cv2.imwrite(str(drq_image_path), shifted_image_bgr)
  logging.info("Saved DrQ-shifted image: %s", drq_image_path)
  return True

def _log_route_stop_waypoint(env: CarlaEnv, state: dict, prefix: str) -> None:
  tracked_stop_waypoint = env._tracked_stop_waypoint
  if tracked_stop_waypoint is None:
    logging.info("%s: none", prefix)
    return

  tl_state = state.get('traffic_light_state', 'none')
  tl_distance = state.get('distance_to_stop', 999.0)
  logging.info("%s: state=%s dist=%5.1fm", prefix, tl_state, tl_distance)


def _draw_route_stop_waypoint(env: CarlaEnv) -> None:
  tracked_stop_waypoint = env._tracked_stop_waypoint
  if tracked_stop_waypoint is None:
    return

  stop_location = tracked_stop_waypoint.transform.location
  env.world.debug.draw_line(
      stop_location,
      stop_location + carla.Location(z=1.5),
      thickness=0.20,
      color=carla.Color(255, 255, 0),
      life_time=0.1,
  )

def _log_step(step: int, state: dict, reward: float, action: np.ndarray, obs: dict) -> None:
  obs_summary = _format_observation(obs)
  traffic_lights_active = state.get('traffic_lights_enabled', True)
  components = _format_reward_components(state.get('reward_components', {}))

  logging.debug(
      f"step={step:4d} | "
      f"tl_active={traffic_lights_active} | "
      f"{obs_summary} | "
      f"action=[steer={action[0]:+.2f}, accel_brake={action[1]:+.2f}] | "
      f"reward={reward:+7.2f}{components}"
  )

def _print_waypoint_vectors(env: CarlaEnv) -> None:
  if len(env.route) == 0:
    logging.info('No route available.')
    return

  vehicle_transform = env.vehicle.get_transform()
  vehicle_location = vehicle_transform.location
  yaw = np.radians(vehicle_transform.rotation.yaw)
  forward_x = np.cos(yaw)
  forward_y = np.sin(yaw)
  right_x = np.cos(yaw + np.pi / 2)
  right_y = np.sin(yaw + np.pi / 2)

  start_idx = env.current_waypoint_idx
  num_waypoints = min(10, len(env.route) - start_idx)
  logging.info("Waypoint vectors (first %d starting from WP %d of %d):",
               num_waypoints, start_idx, len(env.route))

  for i in range(num_waypoints):
    idx = start_idx + i
    waypoint_location = env.route[idx][0].transform.location
    dx = waypoint_location.x - vehicle_location.x
    dy = waypoint_location.y - vehicle_location.y
    forward_distance = dx * forward_x + dy * forward_y
    lateral_offset = dx * right_x + dy * right_y
    logging.info(
        "  WP %2d: (%+7.2f fwd, %+7.2f lat)",
        idx,
        forward_distance,
        lateral_offset,
    )

def run(config: dict | None = None) -> None:
  debug_config = dict(DEBUG_CONFIG)
  if config:
    debug_config.update(config)

  setup_logging(level=logging.DEBUG)
  control_mode = _resolve_control_mode(debug_config)

  logging.info('=' * 70)
  logging.info('DEBUG RUN')
  logging.info(
      "map=%s weather=%s steps=%s episodes=%s control_mode=%s",
      debug_config['map'],
      debug_config['weather'],
      debug_config['max_steps'],
      debug_config['episodes'],
      control_mode,
  )
  logging.info('=' * 70)

  env = CarlaEnv(
      config_path='config/base.yaml',
      phase_config_path='config/training.yaml',
      reward_fn=compute_reward,
      mode='test',
  )

  available_maps = [map_name.split('/')[-1]
                    for map_name in env.client.get_available_maps()]
  if debug_config['map'] not in available_maps:
    raise ValueError(
        f"Map '{debug_config['map']}' not found. Available: {sorted(available_maps)}")

  _apply_overrides(env, debug_config)

  agent = None
  if control_mode == 'rl':
    model_path = debug_config['agent_path']
    if model_path is None:
      model_path, steps = find_latest_checkpoint('checkpoints')
      if model_path is None:
        raise FileNotFoundError(
            "control_mode='rl' but no checkpoint found in checkpoints/")
      logging.info("Auto-detected checkpoint: %s (%s steps)",
                   model_path, steps)
    else:
      logging.info("Loading checkpoint: %s", model_path)

    agent = load_agent(model_path, env=env)

  spectator = None

  try:
    for episode in range(1, debug_config['episodes'] + 1):
      episode_seed = debug_config['base_reset_seed'] + episode - 1
      logging.info("\n%s", '-' * 70)
      logging.info("Episode %d/%d seed=%d", episode,
                   debug_config['episodes'], episode_seed)

      obs, info = env.reset(seed=episode_seed)
      logging.info(
          "Reset: map=%s weather=%s dist=%.1fm traffic_lights_enabled=%s",
          info['map'],
          info['weather'],
          info['initial_distance'],
          info.get('traffic_lights_enabled', True),
      )
      logging.info("Initial observation: %s", _format_observation(obs))
      logging.info("Initial current_waypoint_idx: %s",
                   info.get('current_waypoint_idx', -1))
      _log_route_stop_waypoint(env, info, 'Initial route stop waypoint')

      if spectator is None:
        spectator = env.world.get_spectator()
      _set_spectator_at_vehicle(env, spectator)

      _print_waypoint_vectors(env)

      done = False
      step = 0
      pending_camera_snapshot = bool(
          debug_config['save_image_on_first_step'] and getattr(env, 'camera_enabled', False))
      pending_drq_snapshot = bool(
          debug_config['save_drq_image_on_first_step'] and getattr(env, 'camera_enabled', False))
      snapshot_warned = False
      prev_off_road = bool(info.get('off_road', False))
      prev_tl_state = info.get('traffic_light_state', 'none')

      while not done:
        if control_mode == 'rl' and agent is not None:
          action, _ = agent.predict(obs, deterministic=True)
          action = np.asarray(action, dtype=np.float32)
        else:
          action = np.array([0.0, 0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if pending_camera_snapshot or pending_drq_snapshot:
          if env._rgb_image is not None:
            if pending_camera_snapshot:
              pending_camera_snapshot = not _save_camera_image(
                  episode, env, debug_config)
            if pending_drq_snapshot:
              pending_drq_snapshot = not _save_drq_image(
                  episode, env, debug_config)
          elif step >= debug_config['camera_snapshot_max_wait_steps'] and not snapshot_warned:
            logging.warning(
                "Camera image not available by step %d, skipping first-frame save for this episode.",
                step,
            )
            pending_camera_snapshot = False
            pending_drq_snapshot = False
            snapshot_warned = True

        if spectator is not None and control_mode == 'rl':
          spectator_transform = spectator.get_transform()
          vehicle_transform = env.vehicle.get_transform()
          forward = spectator_transform.get_forward_vector()
          location = vehicle_transform.location - \
              forward * debug_config['spectator_distance_behind'] + \
              carla.Vector3D(z=debug_config['spectator_height_above'])
          spectator.set_transform(carla.Transform(
              location, spectator_transform.rotation))

        destination = env.spawn_points[env.dest_idx].location
        env.world.debug.draw_line(
            destination,
            destination + carla.Location(z=150),
            thickness=0.4,
            color=carla.Color(255, 0, 0),
            life_time=0.1,
        )

        for waypoint, _ in env.route:
          waypoint_location = waypoint.transform.location
          env.world.debug.draw_line(
              waypoint_location,
              waypoint_location + carla.Location(z=1.5),
              thickness=0.15,
              color=carla.Color(0, 255, 0),
              life_time=0.1,
          )

        _draw_route_stop_waypoint(env)

        if step % debug_config['log_every_n_steps'] == 0 or done:
          _log_step(step, info, reward, action, obs)

        current_off_road = bool(info.get('off_road', False))
        if current_off_road != prev_off_road:
          logging.info(
              "Step %4d: off_road %s -> %s (off_road_steps=%d)",
              step,
              prev_off_road,
              current_off_road,
              int(info.get('off_road_steps', 0)),
          )
          prev_off_road = current_off_road

        current_tl_state = info.get('traffic_light_state', 'none')
        if current_tl_state != prev_tl_state:
          logging.info(
              "Step %4d: traffic_light %s -> %s at %.1fm",
              step,
              prev_tl_state,
              current_tl_state,
              info.get('distance_to_stop', 999.0),
          )
          prev_tl_state = current_tl_state

      logging.info(
          "Episode %d ended: steps=%d collision=%s red_violation=%s "
          "off_road_steps=%d dest=%.1fm",
          episode,
          step,
          info.get('collision', False),
          info.get('traffic_light_violation', False),
          int(info.get('off_road_steps', 0)),
          info.get('distance_to_destination', 0.0),
      )

  except KeyboardInterrupt:
    logging.info('Interrupted by user.')
  finally:
    env.close()
    logging.info('Closed.')

if __name__ == '__main__':
  run()
