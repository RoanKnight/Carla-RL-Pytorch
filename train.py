import logging

from sac_carla import (
  create_agent,
  create_env,
  get_agent_type,
  get_callbacks,
  get_checkpoint_dir,
  load_agent,
)
from utils import find_latest_checkpoint, load_config, setup_logging

def train():
  """Training loop with checkpoint resuming."""
  setup_logging()
  config = load_config('config/training.yaml')
  agent_variant = get_agent_type()
  checkpoint_dir = get_checkpoint_dir(config, agent_variant)
  logging.info(
      f"Training SAC on CARLA - {config['description']}")
  logging.info(f"Total timesteps: {config['training']['total_timesteps']}")
  logging.info(f"Agent variant: {agent_variant}")
  logging.info(f"Checkpoint directory: {checkpoint_dir}")

  env = create_env(mode='train', agent_variant=agent_variant)
  try:
    # Check for existing checkpoints and resume if found
    checkpoint_path, checkpoint_steps = find_latest_checkpoint(
        checkpoint_dir)

    if checkpoint_path:
      agent = load_agent(checkpoint_path, env=env, agent_variant=agent_variant)
      remaining_steps = config['training']['total_timesteps'] - \
          checkpoint_steps
      logging.info(f"Found checkpoint at {checkpoint_steps} steps")
      logging.info(f"Resuming from {checkpoint_steps} steps")
    else:
      agent = create_agent(env, config, agent_variant=agent_variant)
      remaining_steps = config['training']['total_timesteps']
      logging.info("Starting from scratch")

    callbacks = get_callbacks(config, checkpoint_dir=checkpoint_dir)

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
