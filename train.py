import logging
from utils import load_config, setup_logging, find_latest_checkpoint
from sac_carla import create_env, create_agent, get_callbacks, load_agent

def train():
  """Training loop with checkpoint resuming."""
  setup_logging()
  config = load_config('config/phase1.yaml')
  logging.info(
      f"Training SAC on CARLA - Phase {config['phase']}: {config['name']}")
  logging.info(f"Total timesteps: {config['training']['total_timesteps']}")

  env = create_env(mode='train')
  try:
    # Check for existing checkpoints and resume if found
    checkpoint_path, checkpoint_steps = find_latest_checkpoint(
        config['logging']['checkpoint_dir'])

    if checkpoint_path:
      agent = load_agent(checkpoint_path, env=env)
      remaining_steps = config['training']['total_timesteps'] - \
          checkpoint_steps
      logging.info(f"Found checkpoint at {checkpoint_steps} steps")
      logging.info(f"Resuming from {checkpoint_steps} steps")
    else:
      agent = create_agent(env, config)
      remaining_steps = config['training']['total_timesteps']
      logging.info("Starting from scratch")

    callbacks = get_callbacks(config)

    agent.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        log_interval=config['training']['log_interval'],
        progress_bar=True,
        reset_num_timesteps=False,
    )
  except KeyboardInterrupt:
    # Enable ability to interrupt training with keyboard shortcut
    logging.info(
        "Training interrupted by user (KeyboardInterrupt). Cleaning up...")
  finally:
    env.close()

if __name__ == '__main__':
  train()
