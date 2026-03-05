import argparse
import carla
import logging
import re
from sac_carla import create_env, load_agent, apply_curriculum_for_timestep
from utils import load_config, setup_logging, find_latest_checkpoint

def _extract_checkpoint_steps(model_path: str):
  """Extract training step count from checkpoint filename."""
  match = re.search(r'_(\d+)_steps\.zip$', model_path)
  if match is None:
    return None
  return int(match.group(1))

def _format_episode_outcome(end_reason: str, end_note: str) -> str:
  """Format episode end reason and note into concise display string."""
  if end_reason == 'success':
    return 'SUCCESS'
  elif end_reason == 'failure_collision':
    return 'FAIL: Collision'
  elif end_reason == 'failure_red_light_violation':
    return f'FAIL: {end_note}'.replace('Crossed stop line on red at ', 'Crossed red at ')
  elif end_reason == 'failure_timeout':
    return f'FAIL: {end_note}'.replace('Reached max steps ', 'Timeout ')
  else:
    return f'FAIL: {end_note}' if end_note else 'FAIL: Unknown'

def test(model_path: str = None, episodes: int = 5):
  """Test trained agent with visual feedback."""
  setup_logging()
  base_config = load_config('config/base.yaml')
  training_config = load_config('config/training.yaml')

  env = None
  try:
    if not model_path:
      model_path, checkpoint_steps = find_latest_checkpoint('checkpoints')
      if not model_path:
        raise FileNotFoundError(
            "No checkpoint found in 'checkpoints' directory")
    else:
      checkpoint_steps = _extract_checkpoint_steps(model_path)
      if checkpoint_steps is None:
        checkpoint_steps = 0
        logging.warning(
            "Could not parse checkpoint steps from '%s'; defaulting curriculum timestep to 0.",
            model_path
        )

    logging.info(f"Testing agent from: {model_path}")
    logging.info(
        f"Checkpoint steps: {checkpoint_steps} — applying matching curriculum")
    logging.info(f"Episodes: {episodes}")

    # Create non-vectorized env for direct access to CARLA
    env = create_env(vectorize=False, mode='test')
    agent = load_agent(model_path, env=env)

    # Unwrap Monitor to access CarlaEnv directly
    carla_env = env.unwrapped

    # Apply the curriculum settings that were active at checkpoint_steps during training
    apply_curriculum_for_timestep(
        carla_env, training_config, checkpoint_steps, agent=agent)
    logging.info(
        f"Curriculum applied: maps={carla_env.world_config.available_maps}, "
        f"weathers={carla_env.world_config.available_weathers}, "
        f"traffic={carla_env.world_config.available_traffic_choices}, "
        f"max_steps={carla_env.max_steps}"
    )
    world = carla_env.world

    for ep in range(episodes):
      obs, info = env.reset()
      vehicle = carla_env.vehicle
      spectator = world.get_spectator()
      destination = carla_env.spawn_points[carla_env.dest_idx].location

      episode_finished = False
      episode_reward = 0.0
      episode_steps = 0
      episode_lane_invasions = 0
      prev_lane_invasion_count = 0

      while not episode_finished:
        # Spectator camera following vehicle with free look
        spectator_transform = spectator.get_transform()
        vehicle_transform = vehicle.get_transform()
        forward = spectator_transform.get_forward_vector()
        location = vehicle_transform.location - forward * base_config['camera']['distance_behind'] + \
            carla.Vector3D(z=base_config['camera']['height_above'])
        spectator.set_transform(carla.Transform(
            location, spectator_transform.rotation))

        # Draw destination
        world.debug.draw_line(
            destination,
            destination + carla.Location(z=150),
            thickness=0.4,
            color=carla.Color(255, 0, 0),
            life_time=0.1
        )

        # Use policy for testing, no randomness
        action, _ = agent.predict(obs, deterministic=True)
        obs, step_reward, terminated, truncated, info = env.step(action)

        # Track lane invasions
        lane_invasion_count = int(info.get('lane_invasion_count', 0))
        new_lane_invasions = max(
            0, lane_invasion_count - prev_lane_invasion_count)
        episode_lane_invasions += new_lane_invasions
        prev_lane_invasion_count = lane_invasion_count

        episode_finished = terminated or truncated
        episode_reward += step_reward
        episode_steps += 1

      end_reason = info.get('episode_end_reason') or 'unknown'
      end_note = info.get('episode_end_note') or ''
      outcome = _format_episode_outcome(end_reason, end_note)
      logging.info(
          f"Episode {ep + 1:3d} | Reward {episode_reward:8.2f} | "
          f"Steps {episode_steps:4d} | {outcome}"
      )
  except KeyboardInterrupt:
    logging.info(
        "Testing interrupted by user (KeyboardInterrupt). Cleaning up...")
  finally:
    if env is not None:
      env.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Test trained SAC agent on CARLA")
  parser.add_argument(
      "--model", type=str, help="Path to model checkpoint (defaults to most recent)")
  parser.add_argument("--episodes", type=int, default=5,
                      help="Number of episodes")

  args = parser.parse_args()
  test(model_path=args.model, episodes=args.episodes)
