import argparse
import logging
import re

import carla

from sac_carla import (
  apply_curriculum_for_timestep,
  create_env,
  get_agent_type,
  get_checkpoint_dir,
  load_agent,
)
from utils import find_latest_checkpoint, load_config, setup_logging

def _extract_checkpoint_steps(model_path: str):
  """Extract training step count from checkpoint filename."""
  match = re.search(r'_(\d+)_steps\.zip$', model_path)
  if match is None:
    return None
  return int(match.group(1))

def test(model_path: str = None, episodes: int = 5, agent_choice: str = "auto"):
  """Test trained agent with visual feedback."""
  setup_logging()
  config = load_config('config/base.yaml')
  training_config = load_config('config/training.yaml')
  requested_variant = get_agent_type(agent_choice=agent_choice)
  active_env_variant = get_agent_type(agent_choice="auto")
  checkpoint_dir = get_checkpoint_dir(training_config, requested_variant)

  if requested_variant != active_env_variant:
    raise ValueError(
        "Agent choice does not match current base camera setting. "
        f"Requested variant '{requested_variant}', but base config resolves to '{active_env_variant}'. "
        "Use '--agent auto' or update observation.use_camera in config/base.yaml."
    )

  env = None
  try:
    if not model_path:
      model_path, checkpoint_steps = find_latest_checkpoint(checkpoint_dir)
      if not model_path:
        raise FileNotFoundError(
            f"No checkpoint found in '{checkpoint_dir}' directory")
    else:
      checkpoint_steps = _extract_checkpoint_steps(model_path)
      if checkpoint_steps is None:
        checkpoint_steps = 0
        logging.warning(
            "Could not parse checkpoint steps from '%s'; defaulting curriculum timestep to 0.",
            model_path,
        )

    logging.info(f"Agent variant: {requested_variant}")
    logging.info(f"Checkpoint directory: {checkpoint_dir}")
    logging.info(f"Testing agent from: {model_path}")
    logging.info(
        f"Checkpoint steps: {checkpoint_steps} - applying matching curriculum")
    logging.info(f"Episodes: {episodes}")

    # Create non-vectorized env for direct access to CARLA
    env = create_env(vectorize=False, mode='test',
                     agent_variant=requested_variant)
    agent = load_agent(model_path, env=env, agent_variant=requested_variant)

    # Unwrap Monitor to access CarlaEnv directly
    carla_env = env.unwrapped
    apply_curriculum_for_timestep(carla_env, training_config, checkpoint_steps)
    applied_distribution = carla_env.phase_config['distribution']
    logging.info(
      "Curriculum applied: maps=%s, weathers=%s, max_steps=%s",
      applied_distribution['maps'],
      applied_distribution['weathers'],
      carla_env.max_steps,
    )
    for ep in range(episodes):
      obs, info = env.reset()
      world = carla_env.world
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
  parser.add_argument(
      "--agent",
      type=str,
      default="auto",
      choices=["auto", "camera", "no_camera"],
      help="Agent variant to test: auto (from base config), camera (DrQ), or no_camera (plain SAC)",
  )

  args = parser.parse_args()
  test(model_path=args.model, episodes=args.episodes, agent_choice=args.agent)
