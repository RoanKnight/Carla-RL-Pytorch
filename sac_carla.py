from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from environment import CarlaEnv
from reward import compute_reward

def create_env(phase_config_path: str = 'config/phase1.yaml', vectorize: bool = True):
  """Create CARLA environment with reward function and phase config."""
  def _make_env():
    base = CarlaEnv(
        config_path='config/base.yaml',
        phase_config_path=phase_config_path,
        reward_fn=compute_reward
    )
    return Monitor(base)

  if not vectorize:
    return _make_env()

  # Wrap for training: DummyVecEnv -> VecTransposeImage
  env = DummyVecEnv([_make_env])
  env = VecTransposeImage(env)
  return env

def create_agent(env: CarlaEnv, config: dict) -> SAC:
  """Create SAC agent with config parameters."""
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

def get_callbacks(config: dict, phase_config_path: str = 'config/phase1.yaml') -> list:
  """Create callbacks for training: checkpointing and logging."""
  Path(config['logging']['tensorboard_dir']).mkdir(parents=True, exist_ok=True)
  Path(config['logging']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

  checkpoint_cb = CheckpointCallback(
      save_freq=config['training']['eval_freq'],
      save_path=config['logging']['checkpoint_dir'],
      name_prefix="sac_carla",
  )

  log_cb = EpisodeLogger(log_interval=config['training']['log_interval'])
  return [checkpoint_cb, log_cb]

def load_agent(model_path: str, env: CarlaEnv = None) -> SAC:
  """Load trained SAC agent from checkpoint."""
  return SAC.load(model_path, env=env)

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
