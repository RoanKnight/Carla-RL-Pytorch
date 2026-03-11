import logging
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from environment import CarlaEnv
from augmentation import DrQDictFeaturesExtractor
from reward import compute_reward

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

  # Wrap for training: DummyVecEnv -> VecTransposeImage
  env = DummyVecEnv([_make_env])
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
  drq_config = config.get('drq', {})
  policy_kwargs = dict(
      features_extractor_class=DrQDictFeaturesExtractor,
      features_extractor_kwargs={
          'cnn_output_dim': int(drq_config.get('cnn_output_dim', 256)),
          'drq_pad': int(drq_config.get('pad', 4)),
      },
  )

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
      policy_kwargs=policy_kwargs,
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

  # Add curriculum manager for all dimensions
  curriculum_config = config.get('curriculum', {})
  if curriculum_config:
    curriculum_cb = CurriculumManager(curriculum_config, verbose=1)
    callbacks.append(curriculum_cb)

  return callbacks

def load_agent(model_path: str, env: CarlaEnv = None) -> SAC:
  """Load trained SAC agent from checkpoint."""
  return SAC.load(model_path, env=env)

def apply_curriculum_for_timestep(base_env, config: dict, timestep: int, agent=None) -> None:
  """Apply curriculum settings for a given timestep to a non-vectorized env.

  Replicates what CurriculumManager does during training, so testing a checkpoint uses the same maps/weather/episode_length settings that were active
  for that checkpoint during training.
  """
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
      base_env.max_steps = active_entry.get('max_steps', base_env.max_steps)
    elif dimension == 'maps':
      base_env.world_config.update_map_choices(active_entry.get('choices', []))
    elif dimension == 'weathers':
      base_env.world_config.update_weather_choices(
          active_entry.get('choices', []))
    elif dimension == 'drq' and agent is not None:
      drq_enabled = bool(active_entry.get('enabled', False))
      curriculum_mgr = CurriculumManager(curriculum)
      curriculum_mgr.model = agent
      curriculum_mgr._set_drq_enabled(drq_enabled)

class EpisodeLogger(BaseCallback):
  """Log episode statistics to console during training."""

  def __init__(self, log_interval: int = 10):
    super().__init__()
    self.log_interval = log_interval
    self.episode_count = 0
    self.current_episode_reward = 0.0
    self.current_episode_steps = 0

  def _format_episode_outcome(self, end_reason: str, end_note: str) -> str:
    """Format episode end reason and note into concise display string."""
    if end_reason == 'success':
      return 'SUCCESS'
    elif end_reason == 'failure_collision':
      return 'FAIL: Collision'
    elif end_reason == 'failure_red_light_violation':
      return f'FAIL: {end_note}'.replace('Crossed stop line on red at ', 'Crossed red at ')
    elif end_reason == 'failure_timeout':
      return f'FAIL: {end_note}'.replace('Reached max steps ', 'Timeout ')
    else:
      return f'FAIL: {end_note}' if end_note else 'FAIL: Unknown'

  def _on_step(self) -> bool:
    step_reward = float(self.locals["rewards"][0])
    episode_finished = self.locals["dones"][0]
    self.current_episode_reward += step_reward
    self.current_episode_steps += 1

    if episode_finished:
      self.episode_count += 1
      episode_info = {}
      infos = self.locals.get("infos")
      if isinstance(infos, (list, tuple)) and infos:
        first_info = infos[0]
        if isinstance(first_info, dict):
          episode_info = first_info

      if self.episode_count % self.log_interval == 0:
        end_reason = episode_info.get("episode_end_reason") or "unknown"
        end_note = episode_info.get("episode_end_note") or ""
        outcome = self._format_episode_outcome(end_reason, end_note)
        print(
            f"Episode {self.episode_count:4d} | "
            f"Reward {self.current_episode_reward:8.2f} | "
            f"Steps {self.current_episode_steps:4d} | {outcome}",
            flush=True
        )
      self.current_episode_reward = 0.0
      self.current_episode_steps = 0

    return True

class CurriculumManager(BaseCallback):
  """Manage multiple curriculum dimensions based on training progress."""

  def __init__(self, curriculum_config: dict, verbose: int = 1):
    super().__init__(verbose)
    self.curriculum = {}
    self.current_values = {}

    # Parse each dimension's schedule
    for dimension, schedule in curriculum_config.items():
      if isinstance(schedule, list) and len(schedule) > 0:
        self.curriculum[dimension] = sorted(
            schedule, key=lambda x: x['timesteps'])
        # Initialize with first value
        self.current_values[dimension] = self._get_initial_value(
            dimension, schedule[0])

  def _get_initial_value(self, dimension: str, entry: dict):
    """Extract the value from a schedule entry based on dimension type."""
    if dimension == 'episode_length':
      return entry.get('max_steps')
    elif dimension == 'maps':
      return entry.get('choices', [])
    elif dimension == 'weathers':
      return entry.get('choices', [])
    elif dimension == 'drq':
      return bool(entry.get('enabled', False))
    return None

  def _get_value_for_timestep(self, schedule: list, timesteps: int, dimension: str):
    """Find appropriate value for current timestep."""
    value = self._get_initial_value(dimension, schedule[0])
    for entry in schedule:
      if timesteps >= entry['timesteps']:
        value = self._get_initial_value(dimension, entry)
      else:
        break
    return value

  def _set_drq_enabled(self, enabled: bool):
    """Toggle DrQ augmentation on all policy feature extractors."""
    policy = getattr(self.model, 'policy', None)
    if policy is None:
      return

    seen = set()
    for module_name in ('actor', 'critic', 'critic_target'):
      module = getattr(policy, module_name, None)
      extractor = getattr(module, 'features_extractor',
                          None) if module else None
      if extractor is None or not hasattr(extractor, 'set_drq_enabled'):
        continue
      extractor_id = id(extractor)
      if extractor_id in seen:
        continue
      extractor.set_drq_enabled(enabled)
      seen.add(extractor_id)

  def _apply_change(self, base_env, dimension: str, value):
    """Apply a curriculum change to environment or model."""
    if dimension == 'episode_length':
      base_env.max_steps = value
    elif dimension == 'maps':
      base_env.world_config.update_map_choices(value)
    elif dimension == 'weathers':
      base_env.world_config.update_weather_choices(value)
    elif dimension == 'drq':
      self._set_drq_enabled(bool(value))

  def _on_training_start(self) -> None:
    """Apply initial curriculum values so model/env start in sync."""
    base_env = self.training_env.venv.envs[0].env
    for dimension, value in self.current_values.items():
      self._apply_change(base_env, dimension, value)

  def _on_step(self) -> bool:
    timesteps = self.num_timesteps
    base_env = self.training_env.venv.envs[0].env

    for dimension, schedule in self.curriculum.items():
      new_value = self._get_value_for_timestep(schedule, timesteps, dimension)

      if new_value != self.current_values[dimension]:
        self._apply_change(base_env, dimension, new_value)
        self.current_values[dimension] = new_value

        if self.verbose >= 1:
          print(
              f"\n[Curriculum] Timestep {timesteps}: {dimension} → {new_value}\n", flush=True)

    return True
