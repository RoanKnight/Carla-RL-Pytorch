import os
import glob
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from environment import CarlaEnv
from utils import load_config

class EpisodeLogger(BaseCallback):
  """Log episode statistics to console during training."""
  def __init__(self, log_interval: int = 10):
    super().__init__()
    self.log_interval = log_interval
    self.episode_count = 0
    self.current_episode_reward = 0.0
    self.current_episode_steps = 0

  def _on_step(self) -> bool:
    step_reward = float(self.locals["rewards"][0])
    episode_finished = self.locals["dones"][0]
    self.current_episode_reward += step_reward
    self.current_episode_steps += 1

    if episode_finished:
      self.episode_count += 1
      if self.episode_count % self.log_interval == 0:
        print(
            f"Episode {self.episode_count:4d} | "
            f"Reward {self.current_episode_reward:8.2f} | "
            f"Steps {self.current_episode_steps:4d}"
        )
      self.current_episode_reward = 0.0
      self.current_episode_steps = 0

    return True

def create_dirs(config: dict) -> None:
  """Create logging and checkpoint directories if not already created."""
  Path(config['logging']['tensorboard_dir']).mkdir(parents=True, exist_ok=True)
  Path(config['logging']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

def make_env(config: dict, spawn_point_index: int = None) -> CarlaEnv:
  """Create and wrap CARLA environment with monitoring."""
  if spawn_point_index is not None:
    # Create a modified config for different spawn point
    import yaml
    import tempfile
    import os

    with open('config/base.yaml', 'r') as f:
      env_config = yaml.safe_load(f)
    env_config['vehicle']['spawn_point_index'] = spawn_point_index

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(env_config, f)
      temp_config = f.name

    env = CarlaEnv(config_path=temp_config)
    os.unlink(temp_config)
  else:
    env = CarlaEnv(config_path='config/base.yaml')

  env = Monitor(env)
  return env

def create_agent(env: CarlaEnv, config: dict) -> SAC:
  """Create a SAC agent with config parameters."""
  agent = SAC(
      policy="CnnPolicy",
      env=env,
      learning_rate=config['sac']['learning_rate'],
      buffer_size=config['sac']['buffer_size'],
      batch_size=config['sac']['batch_size'],
      gamma=config['sac']['gamma'],
      tau=config['sac']['tau'],
      ent_coef=config['sac']['ent_coef'],
      learning_starts=config['sac']['learning_starts'],
      tensorboard_log=config['logging']['tensorboard_dir'],
      verbose=0,
  )
  return agent

def get_callbacks(config: dict, env: CarlaEnv) -> tuple:
  """Get list of callbacks for training and eval environment."""
  create_dirs(config)

  checkpoint_cb = CheckpointCallback(
      save_freq=config['training']['eval_freq'],
      save_path=config['logging']['checkpoint_dir'],
      name_prefix="sac_carla",
  )

  eval_env = DummyVecEnv([lambda: make_env(config, spawn_point_index=config['spawn_points']['evaluation'])])
  eval_env = VecTransposeImage(eval_env)
  eval_cb = EvalCallback(
      eval_env,
      best_model_save_path=config['logging']['checkpoint_dir'],
      log_path=config['logging']['tensorboard_dir'],
      eval_freq=config['training']['eval_freq'],
      n_eval_episodes=5,
      deterministic=True,
  )

  log_cb = EpisodeLogger(log_interval=config['training']['log_interval'])
  return [checkpoint_cb, eval_cb, log_cb], eval_env

def train():
  """Training loop with callbacks and progress bar."""
  config = load_config('config/training.yaml')
  print(f"Training SAC on CARLA")
  print(f"Total timesteps: {config['training']['total_timesteps']}")

  env = make_env(config, spawn_point_index=config['spawn_points']['training'])

  checkpoint_path = None
  if os.path.exists(config['logging']['checkpoint_dir']):
    checkpoint_files = glob.glob(os.path.join(config['logging']['checkpoint_dir'], "sac_carla_*_steps.zip"))
    if checkpoint_files:
      checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]))
      checkpoint_path = checkpoint_files[-1]
      print(f"Found checkpoint: {checkpoint_path}")

  if checkpoint_path:
    agent = SAC.load(checkpoint_path, env=env)
    checkpoint_steps = int(checkpoint_path.split('_')[-2])
    remaining_steps = config['training']['total_timesteps'] - checkpoint_steps
    print(f"Resuming training from checkpoint at {checkpoint_steps} steps")
  else:
    agent = create_agent(env, config)
    remaining_steps = config['training']['total_timesteps']
    print("Starting training from scratch")

  callbacks, eval_env = get_callbacks(config, env)

  print("For tensorboard: tensorboard --logdir " + config['logging']['tensorboard_dir'])
  print("-" * 50)

  agent.learn(
      total_timesteps=remaining_steps,
      callback=callbacks,
      log_interval=config['training']['log_interval'],
      progress_bar=True,
      reset_num_timesteps=False,
  )

  env.close()
  eval_env.close()

if __name__ == '__main__':
  train()
