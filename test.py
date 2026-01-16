import argparse
import carla
import logging
from sac_carla import create_env, load_agent
from utils import load_config, setup_logging, find_latest_checkpoint

def test(model_path: str = None, episodes: int = 5):
  """Test trained agent with visual feedback."""
  setup_logging()
  config = load_config('config/base.yaml')

  env = None
  try:
    if not model_path:
      model_path, _ = find_latest_checkpoint('checkpoints')
      if not model_path:
        raise FileNotFoundError(
            "No checkpoint found in 'checkpoints' directory")

    logging.info(f"Testing agent from: {model_path}")
    logging.info(f"Episodes: {episodes}")

    # Create non-vectorized env for direct access to CARLA
    env = create_env(vectorize=False, mode='test')
    agent = load_agent(model_path, env=env)

    # Unwrap Monitor to access CarlaEnv directly
    carla_env = env.unwrapped
    world = carla_env.world
    metrics = {
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'red_light_violations': 0,
        'total_reward': [],
        'episode_length': []
    }

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

        # Use policy network directly for testing, no randomness
        action, _ = agent.predict(obs, deterministic=True)
        obs, step_reward, terminated, truncated, info = env.step(action)

        # Track red-light violations
        if info.get('red_light_violation', False):
          metrics['red_light_violations'] += 1

        episode_finished = terminated or truncated
        episode_reward += step_reward
        episode_steps += 1

      # Track metrics
      metrics['total_reward'].append(episode_reward)
      metrics['episode_length'].append(episode_steps)

      # Classify episode outcome based on info dict and episode state
      if info.get('collision', False):
        metrics['collision'] += 1
      elif episode_steps >= carla_env.max_steps:
        metrics['timeout'] += 1
      else:
        # Episode ended without collision or timeout => reached destination
        metrics['success'] += 1

      logging.info(
          f"Episode {ep + 1}: Reward={episode_reward:8.2f}, Steps={episode_steps:4d}")

    # Summary
    logging.info("=" * 50)
    logging.info("TEST SUMMARY")
    logging.info("=" * 50)
    logging.info(
        f"Success:  {metrics['success']}/{episodes} ({100 * metrics['success'] / episodes:.1f}%)")
    logging.info(f"Collision: {metrics['collision']}/{episodes}")
    logging.info(f"Timeout:   {metrics['timeout']}/{episodes}")
    logging.info(f"Red Light Violations: {metrics['red_light_violations']}/{episodes}")
    logging.info(
        f"Avg Reward: {sum(metrics['total_reward']) / len(metrics['total_reward']):.2f}")
    logging.info(
        f"Avg Steps:  {sum(metrics['episode_length']) / len(metrics['episode_length']):.1f}")
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
