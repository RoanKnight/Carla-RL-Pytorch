from pathlib import Path

from drq.drq_replay_buffer import DrQDictReplayBuffer
from drq.drq_sac import DrQSAC
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from augmentation import DrQDictFeaturesExtractor
from environment import CarlaEnv
from reward import compute_reward
from utils import load_config

AGENT_VARIANT_CAMERA = "camera_drq"
AGENT_VARIANT_NO_CAMERA = "no_camera"
_SUPPORTED_AGENT_CHOICES = {"auto", "camera", "no_camera"}

def _validate_agent_variant(agent_variant: str) -> str:
  variant = str(agent_variant).strip().lower()
  if variant not in (AGENT_VARIANT_CAMERA, AGENT_VARIANT_NO_CAMERA):
    raise ValueError(f"Unsupported agent variant '{agent_variant}'.")
  return variant

def get_agent_type(agent_choice: str = 'auto', config_path: str = 'config/base.yaml') -> str:
  """Resolve agent type from explicit choice or base camera setting."""
  choice = str(agent_choice).strip().lower()
  if choice not in _SUPPORTED_AGENT_CHOICES:
    raise ValueError(
      f"Invalid agent choice '{agent_choice}'. Expected one of: {sorted(_SUPPORTED_AGENT_CHOICES)}")
  if choice == "auto":
    base_config = load_config(config_path)
    use_camera = bool(base_config['observation']['use_camera'])
    return AGENT_VARIANT_CAMERA if use_camera else AGENT_VARIANT_NO_CAMERA
  if choice == "camera":
    return AGENT_VARIANT_CAMERA
  return AGENT_VARIANT_NO_CAMERA

def get_checkpoint_dir(config: dict, agent_variant: str) -> str:
  """Return variant-specific checkpoint directory."""
  checkpoint_root = config['logging']['checkpoint_dir']
  return str(Path(checkpoint_root) / agent_variant)

def create_env(phase_config_path: str = 'config/training.yaml',
               vectorize: bool = True,
               mode: str = 'train',
               agent_variant: str = AGENT_VARIANT_NO_CAMERA):
  """Create CARLA environment with reward function and phase config."""
  active_variant = _validate_agent_variant(agent_variant)

  def _make_env():
    base = CarlaEnv(config_path='config/base.yaml',
                    phase_config_path=phase_config_path,
                    reward_fn=compute_reward,
                    mode=mode)
    return Monitor(base)

  if not vectorize:
    return _make_env()

  env = DummyVecEnv([_make_env])

  # VecTransposeImage requires at least one image observation; skip when camera is off
  if active_variant == AGENT_VARIANT_CAMERA:
    env = VecTransposeImage(env)

  return env

def create_agent(env, config: dict, agent_variant: str) -> SAC:
  """Create SAC agent with config parameters.

  Args:
    env: Gymnasium environment, either vectorized or raw
    config: Configuration dictionary with SAC hyperparameters
    agent_variant: Explicit agent variant ("camera_drq" or "no_camera")

  Returns:
    Initialized SAC agent
  """
  active_variant = _validate_agent_variant(agent_variant)
  camera_enabled = (active_variant == AGENT_VARIANT_CAMERA)

  policy_kwargs = {}
  if camera_enabled:
    drq_config = config['drq']
    drq_pad = int(drq_config['pad'])
    drq_num_views = int(drq_config['num_views'])
    policy_kwargs['features_extractor_class'] = DrQDictFeaturesExtractor
    policy_kwargs['features_extractor_kwargs'] = {
      'cnn_output_dim': 256,
      'normalized_image': False,
    }

  sac_kwargs = dict(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=config['sac']['learning_rate'],
    buffer_size=config['sac']['buffer_size'],
    batch_size=config['sac']['batch_size'],
    gamma=config['sac']['gamma'],
    tau=config['sac']['tau'],
    ent_coef=config['sac']['ent_coef'],
    train_freq=config['sac']['train_freq'],
    gradient_steps=config['sac']['gradient_steps'],
    learning_starts=config['sac']['learning_starts'],
    policy_kwargs=policy_kwargs if policy_kwargs else None,
    verbose=0,
  )

  if active_variant == AGENT_VARIANT_CAMERA:
    return DrQSAC(
      replay_buffer_class=DrQDictReplayBuffer,
      drq_pad=drq_pad,
      drq_num_views=drq_num_views,
      **sac_kwargs,
    )

  return SAC(**sac_kwargs)

def get_callbacks(config: dict, checkpoint_dir: str | None = None) -> list:
  """Create callbacks for training: checkpointing, logging, and curriculum."""
  active_checkpoint_dir = checkpoint_dir or config['logging']['checkpoint_dir']
  Path(active_checkpoint_dir).mkdir(parents=True, exist_ok=True)

  checkpoint_cb = CheckpointCallback(
    save_freq=config['training']['eval_freq'],
    save_path=active_checkpoint_dir,
    name_prefix="sac_carla",
  )

  log_cb = EpisodeLogger(log_interval=config['training']['log_interval'])
  callbacks = [checkpoint_cb, log_cb]

  callbacks.append(CurriculumManager(config['curriculum'], verbose=1))

  return callbacks

def load_agent(model_path: str,
               env: CarlaEnv = None,
               agent_variant: str = AGENT_VARIANT_NO_CAMERA) -> SAC:
  """Load trained SAC agent from checkpoint."""
  active_variant = _validate_agent_variant(agent_variant)
  if active_variant == AGENT_VARIANT_CAMERA:
    return DrQSAC.load(model_path, env=env)
  if active_variant == AGENT_VARIANT_NO_CAMERA:
    return SAC.load(model_path, env=env)
  raise ValueError(f"Unsupported agent variant '{agent_variant}'.")

def _sort_curriculum_schedule(schedule: list) -> list:
  return sorted(schedule, key=lambda x: x['timesteps'])

def _curriculum_value_from_entry(dimension: str, entry: dict):
  if dimension == 'episode_length':
    return int(entry['max_steps'])
  if dimension == 'traffic_lights':
    return bool(entry['enabled'])
  if dimension in ('maps', 'weathers'):
    return list(entry['choices'])
  return None

def _resolve_curriculum_value_for_timestep(schedule: list, timestep: int, dimension: str):
  active_entry = schedule[0]
  for entry in schedule:
    if timestep >= entry['timesteps']:
      active_entry = entry
    else:
      break
  return _curriculum_value_from_entry(dimension, active_entry)

def _apply_curriculum_dimension(base_env, dimension: str, value) -> None:
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

def apply_curriculum_for_timestep(base_env, config: dict, timestep: int) -> None:
  """Apply curriculum settings for a given timestep to a non-vectorized env."""
  curriculum = config['curriculum']
  for dimension, schedule in curriculum.items():
    if not isinstance(schedule, list) or len(schedule) == 0:
      continue

    schedule_sorted = _sort_curriculum_schedule(schedule)
    value = _resolve_curriculum_value_for_timestep(schedule_sorted, timestep, dimension)
    _apply_curriculum_dimension(base_env, dimension, value)

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
      self.current_episode_components[name] = (self.current_episode_components.get(name, 0.0) +
                                               float(value))

  def _format_episode_components(self) -> str:
    """Return top signed reward contributors for the current episode."""
    non_zero = [(name, value) for name, value in self.current_episode_components.items()
                if abs(value) >= 0.01]
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
          flush=True)
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

    # Store the curriculum schedule for each dimension, sorted by timesteps, and initialize current values.
    for dimension, schedule in curriculum_config.items():
      if isinstance(schedule, list) and len(schedule) > 0:
        schedule_sorted = _sort_curriculum_schedule(schedule)
        self.curriculum[dimension] = schedule_sorted
        self.current_values[dimension] = _curriculum_value_from_entry(dimension, schedule_sorted[0])

  @staticmethod
  def _get_base_env(training_env):
    """Unwrap VecTransposeImage -> DummyVecEnv -> Monitor -> CarlaEnv."""
    env = training_env
    # Get the first env from the vectorized wrapper if applicable
    if hasattr(env, 'venv'):
      env = env.venv
    env = env.envs[0]
    return getattr(env, 'env', env)

  def _on_training_start(self) -> None:
    # Apply the initial curriculum values before learning begins.
    base_env = self._get_base_env(self.training_env)
    for dimension, value in self.current_values.items():
      _apply_curriculum_dimension(base_env, dimension, value)

  def _on_step(self) -> bool:
    # Check whether any curriculum dimension should advance at this timestep.
    timesteps = self.num_timesteps
    base_env = self._get_base_env(self.training_env)

    for dimension, schedule in self.curriculum.items():
      new_value = _resolve_curriculum_value_for_timestep(schedule, timesteps, dimension)
      if new_value != self.current_values[dimension]:
        # Queue the change when possible so it is applied cleanly at the next reset.
        if hasattr(base_env, 'queue_curriculum_change'):
          base_env.queue_curriculum_change(dimension, new_value)
        else:
          _apply_curriculum_dimension(base_env, dimension, new_value)
        self.current_values[dimension] = new_value
        if self.verbose >= 1:
          print(
            f"\n[Curriculum] Timestep {timesteps}: {dimension} -> {new_value} (applied next reset)\n",
            flush=True,
          )

    return True
