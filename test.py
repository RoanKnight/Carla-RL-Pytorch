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

def test(model_path: str = None, episodes: int = 5):
  """Test trained agent with visual feedback."""
  setup_logging()
  config = load_config('config/base.yaml')
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
            model_path,
        )

    logging.info(f"Testing agent from: {model_path}")
    logging.info(
        f"Checkpoint steps: {checkpoint_steps} - applying matching curriculum")
    logging.info(f"Episodes: {episodes}")

    # Create non-vectorized env for direct access to CARLA
    env = create_env(vectorize=False, mode='test')
    agent = load_agent(model_path, env=env)

    # Unwrap Monitor to access CarlaEnv directly
    carla_env = env.unwrapped
    apply_curriculum_for_timestep(carla_env, training_config, checkpoint_steps, agent=agent)
    applied_distribution = carla_env.phase_config.get('distribution', {})
    logging.info(
      "Curriculum applied: maps=%s, weathers=%s, max_steps=%s",
      applied_distribution.get('maps', []),
      applied_distribution.get('weathers', []),
      carla_env.max_steps,
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

      while not episode_finished:
        # Spectator camera following vehicle with free look
        spectator_transform = spectator.get_transform()
        vehicle_transform = vehicle.get_transform()
        forward = spectator_transform.get_forward_vector()
        location = vehicle_transform.location - forward * config['camera']['distance_behind'] + \
            carla.Vector3D(z=config['camera']['height_above'])
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

        episode_finished = terminated or truncated
        episode_reward += step_reward
        episode_steps += 1

      logging.info(
          f"Episode {ep + 1}: Reward={episode_reward:8.2f}, Steps={episode_steps:4d}")

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