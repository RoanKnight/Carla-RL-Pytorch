import argparse
import carla
from sac_carla import create_env, load_agent
from utils import load_config

def test(model_path: str = None, episodes: int = 5):
  """Test trained agent with visual feedback."""
  config = load_config('config/base.yaml')
  model_path = model_path or 'logs/checkpoints/best_model.zip'

  print(f"Testing agent from: {model_path}")
  print(f"Episodes: {episodes}\n")

  env = create_env()
  agent = load_agent(model_path, env=env)

  world = env.world
  metrics = {
      'success': 0,
      'collision': 0,
      'timeout': 0,
      'total_reward': [],
      'episode_length': []
  }

  for ep in range(episodes):
    obs, info = env.reset()
    vehicle = env.vehicle
    spectator = world.get_spectator()
    destination = env.spawn_points[env.dest_idx].location

    episode_finished = False
    episode_reward = 0.0
    episode_steps = 0

    while not episode_finished:
      # Spectator camera following vehicle
      spectator_transform = spectator.get_transform()
      transform = vehicle.get_transform()
      forward = spectator_transform.get_forward_vector()
      location = transform.location - forward * config['camera']['distance_behind'] + \
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

      episode_finished = terminated or truncated
      episode_reward += step_reward
      episode_steps += 1

    # Track metrics
    metrics['total_reward'].append(episode_reward)
    metrics['episode_length'].append(episode_steps)

    if terminated and not info.get('collision', False):
      metrics['success'] += 1
    elif info.get('collision', False):
      metrics['collision'] += 1
    else:
      metrics['timeout'] += 1

    print(
        f"Episode {ep + 1}: Reward={episode_reward:8.2f}, Steps={episode_steps:4d}")

  # Summary
  env.close()
  print("\n" + "=" * 50)
  print("SUMMARY")
  print("=" * 50)
  print(
      f"Success:  {metrics['success']}/{episodes} ({100 * metrics['success'] / episodes:.1f}%)")
  print(f"Collision: {metrics['collision']}/{episodes}")
  print(f"Timeout:   {metrics['timeout']}/{episodes}")
  print(
      f"Avg Reward: {sum(metrics['total_reward']) / len(metrics['total_reward']):.2f}")
  print(
      f"Avg Steps:  {sum(metrics['episode_length']) / len(metrics['episode_length']):.1f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Test trained SAC agent on CARLA")
  parser.add_argument("--model", type=str, help="Path to model checkpoint")
  parser.add_argument("--episodes", type=int, default=5,
                      help="Number of episodes")

  args = parser.parse_args()
  test(model_path=args.model, episodes=args.episodes)
