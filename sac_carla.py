import logging
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from environment import CarlaEnv
from reward import compute_reward
from utils import load_config
from augmentation import DrQDictFeaturesExtractor

def create_env(phase_config_path: str = 'config/training.yaml', vectorize: bool = True, mode: str = 'train'):
  """Create CARLA environment with reward function and phase config."""
  def _make_env():
    base = CarlaEnv(
        config_path='config/base.yaml',
        phase_config_path=phase_config_path,
        reward_fn=compute_reward,
        mode=mode
    )
    return Monitor(base)

  if not vectorize:
    return _make_env()

  env = DummyVecEnv([_make_env])

  # VecTransposeImage requires at least one image observation; skip when camera is off
  base_config = load_config('config/base.yaml')
  camera_enabled = bool(base_config.get('observation', {}).get('use_camera', True))
  if camera_enabled:
    env = VecTransposeImage(env)

  return env

def create_agent(env, config: dict) -> SAC:
  """Create SAC agent with config parameters.

  Args:
    env: Gymnasium environment, either vectorized or raw
    config: Configuration dictionary with SAC hyperparameters

  Returns:
    Initialized SAC agent
  """
  base_config = load_config('config/base.yaml')
  camera_enabled = bool(base_config.get('observation', {}).get('use_camera', False))
  drq_enabled = bool(config.get('augmentation', {}).get('drq_enabled', False))
  drq_pad = config.get('augmentation', {}).get('drq_pad', 4)
  
  policy_kwargs = {}
  if camera_enabled and drq_enabled:
    policy_kwargs['features_extractor_class'] = DrQDictFeaturesExtractor
    policy_kwargs['features_extractor_kwargs'] = {
        'cnn_output_dim': 256,
        'drq_enabled': True,
        'drq_pad': drq_pad,
        'normalized_image': False,
    }
  
  agent = SAC(
      policy="MultiInputPolicy",
      env=env,
      learning_rate=config['sac']['learning_rate'],
      buffer_size=config['sac']['buffer_size'],
      batch_size=config['sac']['batch_size'],
      gamma=config['sac']['gamma'],
      tau=config['sac']['tau'],
      ent_coef=config['sac']['ent_coef'],
      learning_starts=config['sac']['learning_starts'],
      policy_kwargs=policy_kwargs if policy_kwargs else None,
      verbose=0,
  )
  return agent

def get_callbacks(config: dict) -> list:
  """Create callbacks for training: checkpointing, logging, and curriculum."""
  Path(config['logging']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

  checkpoint_cb = CheckpointCallback(
      save_freq=config['training']['eval_freq'],
      save_path=config['logging']['checkpoint_dir'],
      name_prefix="sac_carla",
  )

  log_cb = EpisodeLogger(log_interval=config['training']['log_interval'])
  callbacks = [checkpoint_cb, log_cb]

  curriculum_config = config.get('curriculum', {})
  if curriculum_config:
    callbacks.append(CurriculumManager(curriculum_config, verbose=1))

  return callbacks

def load_agent(model_path: str, env: CarlaEnv = None) -> SAC:
  """Load trained SAC agent from checkpoint."""
  return SAC.load(model_path, env=env)

def apply_curriculum_for_timestep(base_env, config: dict, timestep: int, agent=None) -> None:
  """Apply curriculum settings for a given timestep to a non-vectorized env."""
  curriculum = config.get('curriculum', {})
  for dimension, schedule in curriculum.items():
    if not isinstance(schedule, list) or len(schedule) == 0:
      continue

    schedule_sorted = sorted(schedule, key=lambda x: x['timesteps'])
    active_entry = schedule_sorted[0]
    for entry in schedule_sorted:
      if timestep >= entry['timesteps']:
        active_entry = entry
      else:
        break

    if dimension == 'episode_length':
      base_env.max_steps = int(
          active_entry.get('max_steps', base_env.max_steps))
    elif dimension == 'maps':
      distribution = base_env.phase_config.setdefault('distribution', {})
      distribution['maps'] = list(active_entry.get(
          'choices', distribution.get('maps', [])))
    elif dimension == 'traffic_lights':
      base_env.set_traffic_lights_enabled(active_entry.get('enabled', True))
    elif dimension == 'weathers':
      distribution = base_env.phase_config.setdefault('distribution', {})
      distribution['weathers'] = list(active_entry.get(
          'choices', distribution.get('weathers', [])))

class EpisodeLogger(BaseCallback):
  """Log episode statistics to console during training."""

  def __init__(self, log_interval: int = 10):
    super().__init__()
    self.log_interval = log_interval
    self.episode_count = 0
    self.current_episode_reward = 0.0
    self.current_episode_steps = 0
    self.current_episode_components = {}

  def _accumulate_reward_components(self, step_components: dict):
    """Accumulate per-step reward components into episode totals."""
    if not isinstance(step_components, dict):
      return
    for name, value in step_components.items():
      self.current_episode_components[name] = (
          self.current_episode_components.get(name, 0.0) + float(value))

  def _format_episode_components(self) -> str:
    """Return top signed reward contributors for the current episode."""
    non_zero = [
        (name, value)
        for name, value in self.current_episode_components.items()
        if abs(value) >= 0.01
    ]
    if not non_zero:
      return "components: none"
    top = sorted(non_zero, key=lambda item: abs(item[1]), reverse=True)[:5]
    rendered = ", ".join(f"{name}={value:+.2f}" for name, value in top)
    return f"components: {rendered}"

  def _on_step(self) -> bool:
    step_reward = float(self.locals["rewards"][0])
    episode_finished = self.locals["dones"][0]
    self.current_episode_reward += step_reward
    self.current_episode_steps += 1
    infos = self.locals.get("infos")
    if isinstance(infos, (list, tuple)) and infos:
      first_info = infos[0]
      if isinstance(first_info, dict):
        self._accumulate_reward_components(first_info.get("reward_components"))

    if episode_finished:
      self.episode_count += 1
      if self.episode_count % self.log_interval == 0:
        component_summary = self._format_episode_components()
        print(
            f"Episode {self.episode_count:4d} | "
            f"Reward {self.current_episode_reward:8.2f} | "
            f"Steps {self.current_episode_steps:4d} | {component_summary}",
            flush=True
        )
      self.current_episode_reward = 0.0
      self.current_episode_steps = 0
      self.current_episode_components = {}

    return True

class CurriculumManager(BaseCallback):
  """Manage curriculum dimensions based on training progress."""

  def __init__(self, curriculum_config: dict, verbose: int = 1):
    super().__init__(verbose)
    self.curriculum = {}
    self.current_values = {}

    for dimension, schedule in curriculum_config.items():
      if isinstance(schedule, list) and len(schedule) > 0:
        self.curriculum[dimension] = sorted(
            schedule, key=lambda x: x['timesteps'])
        self.current_values[dimension] = self._get_initial_value(
            dimension, schedule[0])

  @staticmethod
  def _get_base_env(training_env):
    """Unwrap VecTransposeImage -> DummyVecEnv -> Monitor -> CarlaEnv."""
    env = training_env
    if hasattr(env, 'venv'):
      env = env.venv
    env = env.envs[0]
    return getattr(env, 'env', env)

  def _get_initial_value(self, dimension: str, entry: dict):
    if dimension == 'episode_length':
      return int(entry.get('max_steps')) if entry.get('max_steps') is not None else None
    if dimension == 'traffic_lights':
      return bool(entry.get('enabled', True))
    if dimension in ('maps', 'weathers'):
      return list(entry.get('choices', []))
    return None

  def _get_value_for_timestep(self, schedule: list, timesteps: int, dimension: str):
    value = self._get_initial_value(dimension, schedule[0])
    for entry in schedule:
      if timesteps >= entry['timesteps']:
        value = self._get_initial_value(dimension, entry)
      else:
        break
    return value

  @staticmethod
  def _apply_change(base_env, dimension: str, value):
    if dimension == 'episode_length' and value is not None:
      base_env.max_steps = int(value)
    elif dimension == 'maps':
      distribution = base_env.phase_config.setdefault('distribution', {})
      distribution['maps'] = list(value or [])
    elif dimension == 'traffic_lights':
      base_env.set_traffic_lights_enabled(value)
    elif dimension == 'weathers':
      distribution = base_env.phase_config.setdefault('distribution', {})
      distribution['weathers'] = list(value or [])

  def _on_training_start(self) -> None:
    base_env = self._get_base_env(self.training_env)
    for dimension, value in self.current_values.items():
      self._apply_change(base_env, dimension, value)

  def _on_step(self) -> bool:
    timesteps = self.num_timesteps
    base_env = self._get_base_env(self.training_env)

    for dimension, schedule in self.curriculum.items():
      new_value = self._get_value_for_timestep(schedule, timesteps, dimension)
      if new_value != self.current_values[dimension]:
        if hasattr(base_env, 'queue_curriculum_change'):
          base_env.queue_curriculum_change(dimension, new_value)
        else:
          self._apply_change(base_env, dimension, new_value)
        self.current_values[dimension] = new_value
        if self.verbose >= 1:
          print(
              f"\n[Curriculum] Timestep {timesteps}: {dimension} -> {new_value} (applied next reset)\n",
              flush=True,
          )

    return True
