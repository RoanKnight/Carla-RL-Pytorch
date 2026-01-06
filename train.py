import os
import glob
from utils import load_config
from sac_carla import create_env, create_agent, get_callbacks, load_agent

def train():
  """Training loop with checkpoint resuming."""
  config = load_config('config/phase1.yaml')
  print(f"Training SAC on CARLA - Phase {config['phase']}: {config['name']}")
  print(f"Total timesteps: {config['training']['total_timesteps']}")

  env = create_env()

  # Check for existing checkpoints
  checkpoint_path = None
  if os.path.exists(config['logging']['checkpoint_dir']):
    checkpoint_files = glob.glob(
        os.path.join(config['logging']['checkpoint_dir'],
                     "sac_carla_*_steps.zip")
    )
    if checkpoint_files:
      checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))
      checkpoint_path = checkpoint_files[-1]
      print(f"Found checkpoint: {checkpoint_path}")

  if checkpoint_path:
    agent = load_agent(checkpoint_path, env=env)
    checkpoint_steps = int(checkpoint_path.split('_')[-2])
    remaining_steps = config['training']['total_timesteps'] - checkpoint_steps
    print(f"Resuming from {checkpoint_steps} steps")
  else:
    agent = create_agent(env, config)
    remaining_steps = config['training']['total_timesteps']
    print("Starting from scratch")

  callbacks = get_callbacks(config)

  print("Tensorboard: tensorboard --logdir " +
        config['logging']['tensorboard_dir'])
  print("-" * 50)

  agent.learn(
      total_timesteps=remaining_steps,
      callback=callbacks,
      log_interval=config['training']['log_interval'],
      progress_bar=True,
      reset_num_timesteps=False,
  )

  env.close()

if __name__ == '__main__':
  train()
