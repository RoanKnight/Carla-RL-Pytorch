import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import weakref
from utils import load_config

class CarlaEnv(gym.Env):
  """Gymnasium wrapper for CARLA simulator, handles CARLA setup, observation/action spaces,
  and episode lifecycle
  """
  metadata = {'render_modes': ['human', 'rgb_array']}

  def __init__(self, config_path='config/base.yaml', reward_fn=None):
    super().__init__()
    self.config = load_config(config_path)
    self.reward_fn = reward_fn

    # Episode state
    self.step_count = 0
    self.max_steps = 1000
    self.collision_occurred = False

    # CARLA objects to be initialized
    self.client = None
    self.world = None
    self.vehicle = None
    self.rgb_camera = None
    self.collision_sensor = None
    self._original_settings = None
    self._rgb_image = None

    self._setup_carla()
    self._setup_spaces()

  def _setup_carla(self):
    """Initialize CARLA client and world settings."""
    self.client = carla.Client(
        self.config['carla']['host'],
        self.config['carla']['port']
    )
    self.client.set_timeout(self.config['carla']['timeout'])

    self.world = self.client.load_world(self.config['world']['map'])
    self._original_settings = self.world.get_settings()

    settings = self.world.get_settings()
    settings.synchronous_mode = self.config['world']['synchronous_mode']
    settings.fixed_delta_seconds = 1.0 / 60.0  # 60Hz refresh rate
    self.world.apply_settings(settings)

    self.world.set_weather(carla.WeatherParameters(**self.config['weather']))
    self.spawn_points = self.world.get_map().get_spawn_points()

  def _setup_spaces(self):
    """Define available actions like steering, throttle, and brake and observation space as RGB image"""
    self.action_space = spaces.Box(
        low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32
    )

    # Observation: RGB image from front camera
    obs_config = self.config.get('observation', {})
    width = obs_config.get('width', 640)
    height = obs_config.get('height', 480)

    self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(height, width, 3),
        dtype=np.uint8
    )

  def _spawn_vehicle(self):
    """Spawn the ego vehicle at configured spawn point."""
    vehicle_blueprint = self.world.get_blueprint_library().filter(
        self.config['vehicle']['model']
    )[0]

    spawn_idx = self.config['vehicle']['spawn_point_index']
    if spawn_idx >= len(self.spawn_points):
      spawn_idx = 0

    self.vehicle = self.world.spawn_actor(
        vehicle_blueprint,
        self.spawn_points[spawn_idx]
    )

  def _setup_sensors(self):
    """Attach RGB camera and collision sensor to vehicle."""
    obs_config = self.config.get('observation', {})
    width = obs_config.get('width', 640)
    height = obs_config.get('height', 480)
    fov = obs_config.get('fov', 90)

    # RGB Camera
    camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(fov))

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    self.rgb_camera = self.world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=self.vehicle
    )

    current_env_instance = weakref.ref(self)
    self.rgb_camera.listen(lambda img: CarlaEnv._on_rgb_image(current_env_instance, img))

    # Collision Sensor
    collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
    self.collision_sensor = self.world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=self.vehicle
    )
    self.collision_sensor.listen(
        lambda evt: CarlaEnv._on_collision(current_env_instance, evt))

  @staticmethod
  def _on_rgb_image(current_env_instance, image):
    """Callback for RGB camera sensor."""
    self = current_env_instance()
    if self is None:
      return
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    # Convert BGRA to RGB
    self._rgb_image = array[:, :, :3][:, :, ::-1]

  @staticmethod
  def _on_collision(current_env_instance, event):
    """Callback for collision sensor."""
    self = current_env_instance()
    if self is None:
      return
    self.collision_occurred = True

  def _get_observation(self):
    """Return current RGB observation."""
    if self._rgb_image is None:
      return np.zeros(self.observation_space.shape, dtype=np.uint8)
    return self._rgb_image.copy()

  def _get_vehicle_state(self):
    """Get current vehicle state for reward calculation and info."""
    
    # Get vehicle transform (location and rotation), velocity, and speed
    vehicle_transform = self.vehicle.get_transform()
    velocity = self.vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y **
                    2 + velocity.z**2) * 3.6

    # Get destination location
    dest_idx = self.config['vehicle']['destination_index']
    destination = self.spawn_points[dest_idx].location if dest_idx < len(
        self.spawn_points) else None
    distance_to_dest = None
    if destination:
      distance_to_dest = vehicle_transform.location.distance(destination)

    return {
        'location': vehicle_transform.location,
        'rotation': vehicle_transform.rotation,
        'speed': speed,
        'destination': destination,
        'distance_to_destination': distance_to_dest,
        'collision': self.collision_occurred
    }

  def _cleanup_actors(self):
    """Destroy all spawned actors."""
    for actor in [self.collision_sensor, self.rgb_camera, self.vehicle]:
      if actor is not None:
        actor.destroy()
    self.vehicle = None
    self.rgb_camera = None
    self.collision_sensor = None
    self._rgb_image = None

  def reset(self, seed=None, options=None):
    """Reset the environment for a new episode."""
    super().reset(seed=seed)

    self._cleanup_actors()
    self._spawn_vehicle()
    self._setup_sensors()

    self.step_count = 0
    self.collision_occurred = False

    # Tick to get first observation
    self.world.tick()

    observation = self._get_observation()
    info = self._get_vehicle_state()

    return observation, info

  def step(self, action):
    """Execute one environment step."""
    # Apply action to vehicle
    control = carla.VehicleControl(
        steer=float(action[0]),
        throttle=float(action[1]),
        brake=float(action[2])
    )
    self.vehicle.apply_control(control)
    self.world.tick()
    self.step_count += 1

    observation = self._get_observation()
    state = self._get_vehicle_state()

    # Reward calculation (delegated to external function)
    reward = 0.0
    if self.reward_fn is not None:
      reward = self.reward_fn(state, action)

    # Termination: collision or reached destination
    terminated = self.collision_occurred
    if state['distance_to_destination'] is not None and state['distance_to_destination'] < 2.0:
      terminated = True

    # Episode timeout: max steps reached
    episode_timeout = self.step_count >= self.max_steps

    info = state
    info['step'] = self.step_count

    return observation, reward, terminated, episode_timeout, info

  def close(self):
    """Clean up CARLA resources."""
    self._cleanup_actors()
    if self._original_settings is not None:
      self.world.apply_settings(self._original_settings)
