import sys
sys.path.append(r'C:\Carla\PythonAPI\carla')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
import weakref
from utils import load_config

class CarlaEnv(gym.Env):
  """Gymnasium wrapper for CARLA simulator."""
  metadata = {'render_modes': ['human', 'rgb_array']}

  def __init__(self, config_path='config/base.yaml', phase_config_path='config/training.yaml', reward_fn=None, mode=None):
    super().__init__()
    self.config = load_config(config_path)
    self.phase_config = load_config(phase_config_path)
    self.weather_presets = load_config(
        'config/presets/weathers.yaml')['presets']
    self.reward_fn = reward_fn
    self.mode = mode

    # Extract reward configuration for passing to reward function
    self.reward_weights = self.phase_config.get('reward_weights', {})
    self.speed_targets = self.phase_config.get('speed_targets', {})

    # Map randomization frequency based on mode
    map_config = self.phase_config.get('map_randomization', {})
    if self.mode == 'train':
      self.map_change_frequency = map_config.get(
          'train_map_randomness_frequency', 1)
    elif self.mode == 'test':
      self.map_change_frequency = map_config.get(
          'test_map_randomness_frequency', 1)

    # Episode state
    self.step_count = 0
    # Get initial max_steps from first curriculum entry
    curriculum = self.phase_config.get('curriculum', {})
    schedule = curriculum.get('episode_length', [])
    if schedule:
      self.max_steps = schedule[0]['max_steps']
    else:
      self.max_steps = 1000
    self.collision_occurred = False
    self.episode_count = 0

    # Curriculum state (updated by CurriculumManager callback)
    self.available_maps = []
    self.available_weathers = []
    self.traffic_density = 'none'

    # Spawn/destination tracking, randomised per episode
    self.spawn_idx = None
    self.dest_idx = None
    self.initial_distance = None
    self.current_map = None
    self.current_weather = None

    # Route planning
    self.route_planner = None
    self.route = []
    self.current_waypoint_idx = 0

    # Previous state tracking for reward calculation
    self.prev_state = None
    self.prev_action = None

    # CARLA objects to be initialized
    self.client = None
    self.world = None
    self.carla_map = None
    self.vehicle = None
    self.rgb_camera = None
    self.rgb_camera_rear = None
    self.collision_sensor = None
    self.lane_invasion_sensor = None
    self._original_settings = None
    self._rgb_image = None
    self._rgb_image_rear = None
    self.lane_invasion = False
    self.lane_invasion_count = 0
    self.drive_mode = 'forward'
    self.prev_drive_mode = 'forward'
    self.obs_width = None
    self.obs_height = None

    self._setup_carla()
    self._setup_spaces()

  def _setup_carla(self):
    """Initialize CARLA client and world settings."""
    self.client = carla.Client(
        self.config['carla']['host'],
        self.config['carla']['port']
    )
    self.client.set_timeout(self.config['carla']['timeout'])

    # Load random map from curriculum initial choices
    curriculum = self.phase_config.get('curriculum', {})
    maps = curriculum.get('maps', [{}])[0].get('choices', [])
    self.available_maps = maps
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

    # Initialize route planner
    self.route_planner = GlobalRoutePlanner(
        self.carla_map, sampling_resolution=2.0)

  def _apply_random_weather(self):
    """Apply a random weather preset from curriculum choices."""
    curriculum = self.phase_config.get('curriculum', {})
    weather_list = curriculum.get('weathers', [{}])[
        0].get('choices', ['clear_noon'])
    if not weather_list:
      weather_list = ['clear_noon']
    preset_name = np.random.choice(weather_list)
    self.current_weather = preset_name
    weather_params = self.weather_presets[preset_name]
    self.world.set_weather(carla.WeatherParameters(**weather_params))

  def _setup_spaces(self):
    """Define available actions like steering, and coupled throttle_brake, and observation space as Dict with image + goal."""
    # Action space: steering [-1,1], left/right, throttle_brake [-1,1], forward/backward
    self.action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32
    )

    # Observation: Dict with RGB image from front camera and goal vector
    obs_config = self.config.get('observation', {})
    width = obs_config.get('width', 640)
    height = obs_config.get('height', 480)
    self.obs_width = width
    self.obs_height = height

    self.observation_space = spaces.Dict({
        "image_front": spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8
        ),
        "image_rear": spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8
        ),
        "speed_limit": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        ),
        "goal": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        ),
        "traffic_light": spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        ),
        "distance_to_stop": spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
    })

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

  def _spawn_camera(self, x, z, yaw, fov, width, height, buffer_attr):
    """Helper function to spawn a camera actor, used for front and rear cameras.

    Returns:
      Spawned camera actor
    """
    camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(fov))

    transform = carla.Transform(
        carla.Location(x=x, z=z),
        carla.Rotation(yaw=yaw)
    )
    camera = self.world.spawn_actor(
        camera_bp, transform, attach_to=self.vehicle)

    current_env_instance = weakref.ref(self)
    camera.listen(
        lambda img: CarlaEnv._on_camera_image(current_env_instance, img, buffer_attr))

    return camera

  def _setup_sensors(self):
    """Attach front/rear RGB cameras, collision sensor, and lane invasion sensor to vehicle."""
    obs_config = self.config.get('observation', {})
    width = obs_config.get('width', 640)
    height = obs_config.get('height', 480)
    fov = obs_config.get('fov', 90)
    rear_fov = obs_config.get('rear_fov', 110)

    self.rgb_camera = self._spawn_camera(
        x=1.5, z=2.4, yaw=0.0, fov=fov,
        width=width, height=height,
        buffer_attr='_rgb_image'
    )

    self.rgb_camera_rear = self._spawn_camera(
        x=-2.0, z=2.4, yaw=180.0, fov=rear_fov,
        width=width, height=height,
        buffer_attr='_rgb_image_rear'
    )

    current_env_instance = weakref.ref(self)

    collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
    self.collision_sensor = self.world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=self.vehicle
    )
    self.collision_sensor.listen(
        lambda evt: CarlaEnv._on_collision(current_env_instance, evt))

    lane_invasion_bp = self.world.get_blueprint_library().find(
        'sensor.other.lane_invasion')
    self.lane_invasion_sensor = self.world.spawn_actor(
        lane_invasion_bp,
        carla.Transform(),
        attach_to=self.vehicle
    )
    self.lane_invasion_sensor.listen(
        lambda evt: CarlaEnv._on_lane_invasion(current_env_instance, evt))

  @staticmethod
  def _on_camera_image(current_env_instance, image, buffer_attr):
    """Unified callback for all RGB camera sensors."""
    self = current_env_instance()
    if self is None:
      return
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb_image = array[:, :, :3][:, :, ::-1]
    setattr(self, buffer_attr, rgb_image)

  @staticmethod
  def _on_collision(current_env_instance, event):
    """Callback for collision sensor."""
    self = current_env_instance()
    if self is None:
      return
    self.collision_occurred = True

  @staticmethod
  def _on_lane_invasion(current_env_instance, event):
    """Callback for lane invasion sensor."""
    self = current_env_instance()
    if self is None:
      return
    self.lane_invasion = True
    self.lane_invasion_count += 1

  def _compute_goal_vector(self):
    """Computes the goal vector to the next waypoint, returns the forward and lateral distance from the next waypoint."""
    if len(self.route) == 0 or self.current_waypoint_idx >= len(self.route):
      return np.array([0.0, 0.0], dtype=np.float32)

    vehicle_transform = self.vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation

    target_waypoint_loc = self.route[self.current_waypoint_idx][0].transform.location

    dx = target_waypoint_loc.x - vehicle_location.x
    dy = target_waypoint_loc.y - vehicle_location.y

    # Get the forward and lateral distance from the next waypoint based on the vehicle's rotation
    yaw = np.radians(vehicle_rotation.yaw)
    forward_x = np.cos(yaw)
    forward_y = np.sin(yaw)
    right_x = np.cos(yaw + np.pi / 2)
    right_y = np.sin(yaw + np.pi / 2)

    forward_distance = dx * forward_x + dy * forward_y
    lateral_offset = dx * right_x + dy * right_y

    return np.array([forward_distance, lateral_offset], dtype=np.float32)

  def _compute_signed_distance_to_stop(self, vehicle_location, stop_waypoint):
    """Compute signed distance along lane direction to stop waypoint.

    Returns:
      Signed distance in meters: negative = before stop line, positive = crossed.
    """
    stop_transform = stop_waypoint.transform
    stop_forward = stop_transform.get_forward_vector()

    dx = vehicle_location.x - stop_transform.location.x
    dy = vehicle_location.y - stop_transform.location.y

    # Use dot product to get the signed distance to the stop waypoint
    signed_distance = dx * stop_forward.x + dy * stop_forward.y

    return signed_distance

  def _get_signed_speed(self):
    """Get vehicle speed with sign (positive=forward, negative=reverse) in km/h."""
    velocity = self.vehicle.get_velocity()
    speed_magnitude = ((velocity.x**2 + velocity.y **
                       2 + velocity.z**2) ** 0.5) * 3.6

    vehicle_transform = self.vehicle.get_transform()
    forward = vehicle_transform.get_forward_vector()
    velocity_dot = velocity.x * forward.x + \
        velocity.y * forward.y + velocity.z * forward.z

    return float(-speed_magnitude if velocity_dot < 0 else speed_magnitude)

  def _detect_traffic_light(self, vehicle_location):
    """Detect traffic light state and compute distance to stop line.

    Returns:
      (traffic_light_state, distance_to_stop): tuple of (str, float)
    """
    traffic_light_state = 'none'
    distance_to_stop = 999.0

    if self.vehicle is not None and self.vehicle.is_at_traffic_light():
      traffic_light = self.vehicle.get_traffic_light()
      tl_state = traffic_light.get_state()

      if tl_state == carla.TrafficLightState.Red:
        traffic_light_state = 'red'
      elif tl_state == carla.TrafficLightState.Yellow:
        traffic_light_state = 'yellow'
      elif tl_state == carla.TrafficLightState.Green:
        traffic_light_state = 'green'

      stop_waypoints = traffic_light.get_stop_waypoints()
      if stop_waypoints:
        stop_waypoint = stop_waypoints[0]
        distance_to_stop = self._compute_signed_distance_to_stop(
            vehicle_location, stop_waypoint
        )

    return traffic_light_state, distance_to_stop

  def _get_observation(self):
    """Return Dict observation with front/rear images, goal vector, traffic light state, and distance to stop."""
    height = self.obs_height if self.obs_height is not None else 84
    width = self.obs_width if self.obs_width is not None else 84
    image_front = self._rgb_image.copy() if self._rgb_image is not None else np.zeros(
        (height, width, 3), dtype=np.uint8)

    image_rear = self._rgb_image_rear.copy() if self._rgb_image_rear is not None else np.zeros(
        (height, width, 3), dtype=np.uint8)

    goal = self._compute_goal_vector()

    speed_limit_kmh = float(self.vehicle.get_speed_limit()
                            ) if self.vehicle is not None else 0.0
    max_speed_limit_kmh = 120.0
    speed_limit_norm = np.array(
        [np.clip(speed_limit_kmh, 0.0, max_speed_limit_kmh) /
         max_speed_limit_kmh],
        dtype=np.float32
    )

    vehicle_location = self.vehicle.get_transform().location if self.vehicle else None
    traffic_light_state, distance_to_stop = self._detect_traffic_light(
        vehicle_location)

    tl_one_hot_map = {
        'none': np.array([1, 0, 0, 0], dtype=np.float32),
        'red': np.array([0, 1, 0, 0], dtype=np.float32),
        'yellow': np.array([0, 0, 1, 0], dtype=np.float32),
        'green': np.array([0, 0, 0, 1], dtype=np.float32),
    }
    tl_one_hot = tl_one_hot_map.get(
        traffic_light_state, tl_one_hot_map['none'])

    distance_clipped = np.clip(distance_to_stop, 0.0, 50.0)
    distance_normalized = np.array([distance_clipped / 50.0], dtype=np.float32)

    return {
        "image_front": image_front,
        "image_rear": image_rear,
        "speed_limit": speed_limit_norm,
        "goal": goal,
        "traffic_light": tl_one_hot,
        "distance_to_stop": distance_normalized
    }

  def _get_vehicle_state(self):
    """Get current vehicle state for reward calculation and info."""
    vehicle_transform = self.vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    velocity = self.vehicle.get_velocity()
    speed = self._get_signed_speed()

    destination = self.spawn_points[self.dest_idx].location
    distance_to_dest = np.sqrt(
        (vehicle_location.x - destination.x)**2 +
        (vehicle_location.y - destination.y)**2
    )

    # Compute distance to current waypoint
    waypoint_distance = 0.0
    if len(self.route) > 0 and self.current_waypoint_idx < len(self.route):
      current_waypoint_loc = self.route[self.current_waypoint_idx][0].transform.location
      waypoint_distance = np.sqrt(
          (vehicle_location.x - current_waypoint_loc.x)**2 +
          (vehicle_location.y - current_waypoint_loc.y)**2
      )

    waypoint = self.carla_map.get_waypoint(
        vehicle_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    max_lane_deviation = self.config.get(
        'lane_detection', {}).get('max_lane_deviation', 5.0)

    if waypoint is None:
      lane_deviation = max_lane_deviation
      off_road = True
    else:
      wp_location = waypoint.transform.location
      lane_deviation_2d = np.sqrt(
          (vehicle_location.x - wp_location.x)**2 +
          (vehicle_location.y - wp_location.y)**2
      )
      lane_deviation = min(lane_deviation_2d, max_lane_deviation)
      off_road = lane_deviation_2d > (waypoint.lane_width / 2.0)

    vehicle_location = self.vehicle.get_transform().location
    traffic_light_state, distance_to_stop = self._detect_traffic_light(
        vehicle_location)

    speed_limit_kmh = float(self.vehicle.get_speed_limit()
                            ) if self.vehicle is not None else 0.0

    return {
        'location': (vehicle_location.x, vehicle_location.y, vehicle_location.z),
        'rotation': (vehicle_transform.rotation.pitch, vehicle_transform.rotation.yaw, vehicle_transform.rotation.roll),
        'speed': speed,
        'speed_limit_kmh': speed_limit_kmh,
        'distance_to_destination': distance_to_dest,
        'waypoint_distance': waypoint_distance,
        'current_waypoint_idx': self.current_waypoint_idx,
        'collision': self.collision_occurred,
        'lane_deviation': lane_deviation,
        'off_road': off_road,
        'traffic_light_state': traffic_light_state,
        'distance_to_stop': distance_to_stop,
        'lane_invasion': self.lane_invasion,
        'lane_invasion_count': self.lane_invasion_count,
        'direction_changed': getattr(self, '_direction_changed_flag', False),
    }

  def _cleanup_actors(self):
    """Destroy all spawned actors in safe order to prevent memory leaks and stop callbacks: stop sensors, tick, destroy."""
    if self.rgb_camera is not None:
      self.rgb_camera.stop()
    if self.rgb_camera_rear is not None:
      self.rgb_camera_rear.stop()
    if self.collision_sensor is not None:
      self.collision_sensor.stop()
    if self.lane_invasion_sensor is not None:
      self.lane_invasion_sensor.stop()

    for actor in [self.lane_invasion_sensor, self.collision_sensor, self.rgb_camera_rear, self.rgb_camera, self.vehicle]:
      if actor is not None and actor.is_alive:
        actor.destroy()

    self.vehicle = None
    self.rgb_camera = None
    self.rgb_camera_rear = None
    self.collision_sensor = None
    self.lane_invasion_sensor = None
    self._rgb_image = None
    self._rgb_image_rear = None
    self.prev_state = None
    self.prev_action = None
    self.lane_invasion = False
    self.lane_invasion_count = 0

  def update_map_choices(self, maps: list):
    """Update available maps for curriculum progression."""
    self.available_maps = maps

  def update_weather_choices(self, weathers: list):
    """Update available weathers for curriculum progression."""
    self.available_weathers = weathers

  def update_traffic_density(self, density: str):
    """Update traffic density for curriculum progression."""
    self.traffic_density = density

  def reset(self, seed=None, options=None):
    """Reset the environment every episode with randomized elements each time."""
    super().reset(seed=seed)

    self._cleanup_actors()

    # Increment episode counter
    self.episode_count += 1

    # Check if map should be changed based on frequency
    if self.episode_count % self.map_change_frequency == 0:
      maps = self.available_maps if self.available_maps else []
      if not maps:
        curriculum = self.phase_config.get('curriculum', {})
        maps = curriculum.get('maps', [{}])[0].get('choices', [])
        self.available_maps = maps

      new_map = np.random.choice(maps)

      # Only reload world if map actually changed
      if new_map != self.current_map:
        self.current_map = new_map
        self.world = self.client.load_world(self.current_map)

        # Apply synchronous settings after loading new world
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 60.0
        self.world.apply_settings(settings)
        self._original_settings = self.world.get_settings()

        # Refresh map-dependent objects
        self.carla_map = self.world.get_map()
        self.spawn_points = self.carla_map.get_spawn_points()

        # Recreate route planner for new map
        self.route_planner = GlobalRoutePlanner(
            self.carla_map, sampling_resolution=2.0)

    # Apply random weather for new episode
    self._apply_random_weather()

    # Randomize spawn and destination indices
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
    self.lane_invasion = False
    self.lane_invasion_count = 0
    self.drive_mode = 'forward'
    self.prev_drive_mode = 'forward'

    # Calculate initial distance to destination
    spawn_loc = self.spawn_points[self.spawn_idx].location
    dest_loc = self.spawn_points[self.dest_idx].location
    self.initial_distance = spawn_loc.distance(dest_loc)

    # Compute route from spawn to destination
    self.route = self.route_planner.trace_route(spawn_loc, dest_loc)
    self.current_waypoint_idx = 0

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
    steer_command = float(action[0])
    target_speed_normalized = float(action[1])

    ctrl_config = self.config.get('control', {})
    max_forward = float(ctrl_config.get('max_forward_speed', 60.0))
    max_reverse = ctrl_config.get('max_reverse_speed', 10.0)
    speed_tolerance = ctrl_config.get('speed_tolerance', 1.0)
    gain = ctrl_config.get('proportional_gain', 0.1)

    if target_speed_normalized >= 0:
      target_speed_kmh = target_speed_normalized * max_forward
    else:
      target_speed_kmh = target_speed_normalized * max_reverse

    current_speed_kmh = self._get_signed_speed()

    # Value thresholds for mode switching
    REVERSE_ENTRY_THRESHOLD = -3.0
    FORWARD_ENTRY_THRESHOLD = 3.0
    STAY_THRESHOLD = 1.0

    if self.drive_mode == 'forward':
      if target_speed_kmh < REVERSE_ENTRY_THRESHOLD:
        requested_mode = 'reverse'
      elif target_speed_kmh > STAY_THRESHOLD:
        requested_mode = 'forward'
      else:
        requested_mode = 'forward'
    elif self.drive_mode == 'reverse':
      if target_speed_kmh > FORWARD_ENTRY_THRESHOLD:
        requested_mode = 'forward'
      elif target_speed_kmh < -STAY_THRESHOLD:
        requested_mode = 'reverse'
      else:
        requested_mode = 'reverse'
    else:  # in stopping modes
      if target_speed_kmh < -STAY_THRESHOLD:
        requested_mode = 'reverse'
      elif target_speed_kmh > STAY_THRESHOLD:
        requested_mode = 'forward'
      else:
        requested_mode = self.drive_mode if 'reverse' in self.drive_mode else 'forward'

    if self.drive_mode in ('forward', 'reverse') and requested_mode != self.drive_mode:
      self.drive_mode = 'stopping_to_reverse' if requested_mode == 'reverse' else 'stopping_to_forward'

    if self.drive_mode == 'stopping_to_reverse' and requested_mode == 'forward':
      self.drive_mode = 'stopping_to_forward'
    elif self.drive_mode == 'stopping_to_forward' and requested_mode == 'reverse':
      self.drive_mode = 'stopping_to_reverse'

    if self.drive_mode in ('stopping_to_reverse', 'stopping_to_forward'):
      if abs(current_speed_kmh) > speed_tolerance:
        throttle = 0.0
        brake = min(max(abs(current_speed_kmh) * gain, 0.0), 1.0)
        reverse_flag = current_speed_kmh < 0.0
      else:
        self.drive_mode = 'reverse' if self.drive_mode == 'stopping_to_reverse' else 'forward'
        throttle = 0.0
        brake = 0.0
        reverse_flag = self.drive_mode == 'reverse'
    else:
      reverse_flag = self.drive_mode == 'reverse'

      mode_sign = -1.0 if reverse_flag else 1.0
      target_speed_mode = mode_sign * target_speed_kmh
      current_speed_mode = mode_sign * current_speed_kmh
      speed_error_mode = target_speed_mode - current_speed_mode

      if speed_error_mode > speed_tolerance:
        throttle = min(max(speed_error_mode * gain, 0.0), 1.0)
        brake = 0.0
      elif speed_error_mode < -speed_tolerance:
        throttle = 0.0
        brake = min(max(abs(speed_error_mode) * gain, 0.0), 1.0)
      else:
        throttle = 0.0
        brake = 0.0

    control = carla.VehicleControl(
        steer=steer_command,
        throttle=throttle,
        brake=brake,
        reverse=reverse_flag
    )
    self.vehicle.apply_control(control)
    self.world.tick()
    self.step_count += 1

    # Detect direction change for reward penalty
    direction_changed = False
    if (self.prev_drive_mode in ('forward', 'reverse') and
        self.drive_mode in ('forward', 'reverse') and
            self.prev_drive_mode != self.drive_mode):
      direction_changed = True

    self._direction_changed_flag = direction_changed
    self.prev_drive_mode = self.drive_mode

    observation = self._get_observation()
    state = self._get_vehicle_state()

    # Update route progress: advance to next waypoint if close enough
    if len(self.route) > 0 and self.current_waypoint_idx < len(self.route):
      vehicle_location = self.vehicle.get_transform().location
      current_waypoint_loc = self.route[self.current_waypoint_idx][0].transform.location
      distance_to_waypoint = vehicle_location.distance(current_waypoint_loc)

      # Advance to next waypoint if within 5 meters
      if distance_to_waypoint < 5.0 and self.current_waypoint_idx < len(self.route) - 1:
        self.current_waypoint_idx += 1

    # Reward calculation with previous state for progress/smoothness
    reward = 0.0
    if self.reward_fn is not None:
      # Pass action and prev_action separately; state is physical state only
      prev_action = self.prev_action if self.prev_action is not None else np.zeros(
          2, dtype=np.float32)
      reward = self.reward_fn(state, action, self.prev_state, prev_action,
                              self.reward_weights, self.speed_targets)

    # Store current state/action for next step
    self.prev_state = state.copy()
    self.prev_action = action.copy()

    # Reset lane invasion flag for next step
    self.lane_invasion = False

    # Termination: collision or reached destination
    terminated = self.collision_occurred
    if state['distance_to_destination'] < 2.0:
      terminated = True

    # Red-light violation termination
    red_light_violation = False
    if (state['traffic_light_state'] == 'red' and
        state['distance_to_stop'] > 0.0 and
            state['speed'] > 5.0):
      terminated = True
      red_light_violation = True

    # Episode timeout: max steps reached
    episode_timeout = self.step_count >= self.max_steps

    info = state
    info['step'] = self.step_count
    info['initial_distance'] = self.initial_distance
    info['red_light_violation'] = red_light_violation

    return observation, reward, terminated, episode_timeout, info

  def close(self):
    """Clean up CARLA resources."""
    self._cleanup_actors()
    if self._original_settings is not None:
      self.world.apply_settings(self._original_settings)
