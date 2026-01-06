import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import weakref
from utils import load_config

class CarlaEnv(gym.Env):
  """Gymnasium wrapper for CARLA simulator."""
  metadata = {'render_modes': ['human', 'rgb_array']}

  def __init__(self, config_path='config/base.yaml', phase_config_path='config/phase1.yaml', reward_fn=None):
    super().__init__()
    self.config = load_config(config_path)
    self.phase_config = load_config(phase_config_path)
    self.weather_presets = load_config(
        'config/presets/weathers.yaml')['presets']
    self.reward_fn = reward_fn

    # Episode state
    self.step_count = 0
    self.max_steps = self.phase_config.get(
        'episode', {}).get('max_steps', 1000)
    self.collision_occurred = False

    # Spawn/destination tracking, randomised per episode
    self.spawn_idx = None
    self.dest_idx = None
    self.initial_distance = None
    self.current_map = None
    self.current_weather = None

    # Previous state tracking for reward calculation
    self.prev_state = None
    self.prev_action = None

    # CARLA objects to be initialized
    self.client = None
    self.world = None
    self.carla_map = None
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

    # Load random map from phase distribution
    maps = self.phase_config['distribution']['maps']
    self.current_map = np.random.choice(maps)
    self.world = self.client.load_world(self.current_map)
    self._original_settings = self.world.get_settings()

    settings = self.world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 60.0
    self.world.apply_settings(settings)

    # Apply random weather from presets
    self._apply_random_weather()

    self.carla_map = self.world.get_map()
    self.spawn_points = self.carla_map.get_spawn_points()

  def _apply_random_weather(self):
    """Apply a random weather preset from phase distribution."""
    weather_list = self.phase_config['distribution'].get('weathers', [
                                                         'clear_noon'])
    if weather_list:
      preset_name = np.random.choice(weather_list)
      self.current_weather = preset_name
      weather_params = self.weather_presets[preset_name]
      self.world.set_weather(carla.WeatherParameters(**weather_params))

  def _setup_spaces(self):
    """Define available actions like steering, and coupled throttle_brake, and observation space as RGB image"""
    # Action space: steering [-1,1], left/right, throttle_brake [-1,1], forward/backward
    self.action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
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
    """Spawn the vehicle at the current spawn point, with retry fallback."""
    vehicle_blueprint = self.world.get_blueprint_library().filter(
        self.config['vehicle']['model']
    )[0]

    self.vehicle = self.world.try_spawn_actor(
        vehicle_blueprint,
        self.spawn_points[self.spawn_idx]
    )

    if self.vehicle is None:
      for _ in range(5):
        self.spawn_idx = self.np_random.integers(0, len(self.spawn_points))
        self.vehicle = self.world.try_spawn_actor(
            vehicle_blueprint,
            self.spawn_points[self.spawn_idx]
        )
        if self.vehicle is not None:
          break

    if self.vehicle is None:
      raise RuntimeError("Failed to spawn vehicle after retries")

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
    self.rgb_camera.listen(
        lambda img: CarlaEnv._on_rgb_image(current_env_instance, img))

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
    vehicle_transform = self.vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    velocity = self.vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2) * 3.6

    destination = self.spawn_points[self.dest_idx].location
    distance_to_dest = np.sqrt(
        (vehicle_location.x - destination.x)**2 +
        (vehicle_location.y - destination.y)**2
    )

    waypoint = self.carla_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint is None:
      lane_deviation = 5.0
      off_road = True
    else:
      wp_location = waypoint.transform.location
      lane_deviation_2d = np.sqrt(
          (vehicle_location.x - wp_location.x)**2 +
          (vehicle_location.y - wp_location.y)**2
      )
      lane_deviation = min(lane_deviation_2d, 5.0)
      off_road = lane_deviation_2d > (waypoint.lane_width / 2.0)

    return {
        'location': (vehicle_location.x, vehicle_location.y, vehicle_location.z),
        'rotation': (vehicle_transform.rotation.pitch, vehicle_transform.rotation.yaw, vehicle_transform.rotation.roll),
        'speed': speed,
        'distance_to_destination': distance_to_dest,
        'collision': self.collision_occurred,
        'lane_deviation': lane_deviation,
        'off_road': off_road,
        'action': self.prev_action,
    }

  def _cleanup_actors(self):
    """Destroy all spawned actors in safe order: stop sensors, tick, destroy."""
    if self.rgb_camera is not None:
      try:
        self.rgb_camera.stop()
      except:
        pass

    if self.collision_sensor is not None:
      try:
        self.collision_sensor.stop()
      except:
        pass

    if self.world is not None:
      try:
        self.world.tick()
      except:
        pass

    if self.collision_sensor is not None:
      try:
        if self.collision_sensor.is_alive:
          self.collision_sensor.destroy()
      except:
        pass

    if self.rgb_camera is not None:
      try:
        if self.rgb_camera.is_alive:
          self.rgb_camera.destroy()
      except:
        pass

    if self.vehicle is not None:
      try:
        if self.vehicle.is_alive:
          self.vehicle.destroy()
      except:
        pass

    self.vehicle = None
    self.rgb_camera = None
    self.collision_sensor = None
    self._rgb_image = None
    self.prev_state = None
    self.prev_action = None

  def reset(self, seed=None, options=None):
    """Reset the environment for a new episode with randomized spawn/destination/weather."""
    super().reset(seed=seed)

    self._cleanup_actors()

    # Randomize weather for new episode
    self._apply_random_weather()

    # Randomize spawn and destination points
    num_spawn_points = len(self.spawn_points)
    self.spawn_idx = self.np_random.integers(0, num_spawn_points)
    self.dest_idx = self.np_random.integers(0, num_spawn_points)
    while self.dest_idx == self.spawn_idx:
      self.dest_idx = self.np_random.integers(0, num_spawn_points)

    self._spawn_vehicle()
    self._setup_sensors()

    # Reset episode state
    self.step_count = 0
    self.collision_occurred = False
    self.prev_state = None
    self.prev_action = None

    # Calculate initial distance for reference
    spawn_loc = self.spawn_points[self.spawn_idx].location
    dest_loc = self.spawn_points[self.dest_idx].location
    self.initial_distance = spawn_loc.distance(dest_loc)

    # Tick to get first observation
    self.world.tick()

    observation = self._get_observation()
    info = self._get_vehicle_state()
    info['initial_distance'] = self.initial_distance
    info['map'] = self.current_map
    info['weather'] = self.current_weather

    return observation, info

  def step(self, action):
    """Execute one environment step."""
    # Convert 2D action to CARLA controls, steering [-1,1], throttle_brake [-1,1]
    throttle = max(0.0, float(action[1]))
    brake = max(0.0, -float(action[1]))

    control = carla.VehicleControl(
        steer=float(action[0]),
        throttle=throttle,
        brake=brake
    )
    self.vehicle.apply_control(control)
    self.world.tick()
    self.step_count += 1

    observation = self._get_observation()
    state = self._get_vehicle_state()

    # Reward calculation with previous state for progress/smoothness
    reward = 0.0
    if self.reward_fn is not None:
      reward = self.reward_fn(state, action, self.prev_state)

    # Store current state/action for next step
    self.prev_state = state.copy()
    self.prev_action = action.copy()

    # Termination: collision or reached destination
    terminated = self.collision_occurred
    if state['distance_to_destination'] < 2.0:
      terminated = True

    # Episode timeout: max steps reached
    episode_timeout = self.step_count >= self.max_steps

    info = state
    info['step'] = self.step_count
    info['initial_distance'] = self.initial_distance

    return observation, reward, terminated, episode_timeout, info

  def close(self):
    """Clean up CARLA resources."""
    self._cleanup_actors()
    if self._original_settings is not None:
      self.world.apply_settings(self._original_settings)
