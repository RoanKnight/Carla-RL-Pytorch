import sys
sys.path.append(r'C:\Carla\PythonAPI\carla')

import logging

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
import weakref
from utils import load_config
from reward import compute_reward, compute_reward_with_components, compute_target_speed, compute_cruise_speed_target

_GOAL_NORMALIZATION_METERS = 10.0
_STOP_DISTANCE_CLIP_METERS = 100.0

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
    self.debug_env_info = bool(
        self.config.get('debug', {}).get('env_info', False))

    # Extract reward configuration for passing to reward function
    self.reward_weights = self.phase_config.get('reward_weights', {})
    control_config = self.config.get('control', {})
    observation_config = self.config.get('observation', {})
    environment_config = self.config.get('environment', {})
    max_forward_speed_kmh = float(
        control_config.get('max_forward_speed', 40.0))
    self.speed_config = dict(self.phase_config.get('speed_limit', {}))
    self.speed_config.setdefault(
        'max_forward_speed_kmh', max_forward_speed_kmh)
    self.camera_enabled = bool(observation_config.get('use_camera', True))
    self.camera_warmup_ticks = int(
        max(0, observation_config.get('camera_warmup_ticks', 3)))
    self.route_sampling_resolution = float(
        environment_config.get('route_sampling_resolution', 4.0))
    self.route_progress_reached_distance_meters = float(
        environment_config.get('route_progress_reached_distance_meters', 2.0))
    self.destination_success_distance_meters = float(
        environment_config.get('destination_success_distance_meters', 2.0))
    self.traffic_light_lookahead_waypoints = int(
        environment_config.get('traffic_light_lookahead_waypoints', 30))
    self.traffic_light_detection_max_distance_meters = float(
        environment_config.get('traffic_light_detection_max_distance_meters', 200.0))
    self.off_road_margin_meters = float(
        environment_config.get('off_road_margin_meters', 0.5))
    self.traffic_light_release_distance_meters = float(
        environment_config.get('traffic_light_release_distance_meters', 2.0))
    self.traffic_lights_enabled = self._get_initial_traffic_lights_enabled()

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
    self.max_steps = self.phase_config.get(
        'episode', {}).get('max_steps', 1000)
    self.collision_occurred = False
    self.episode_count = 0
    self._off_road_steps = 0
    self._off_road_termination_steps = int(
        self.config.get('lane_detection', {}).get('off_road_termination_steps', 200))

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
    self._tl_index = {}

    # Previous state tracking for reward calculation and action history.
    self.prev_state = None
    self.last_action = None

    # Per-step cached values to avoid redundant CARLA API calls
    self._step_transform = None
    self._step_lane_metrics = None

    # CARLA objects to be initialized
    self.client = None
    self.world = None
    self.carla_map = None
    self.vehicle = None
    self.rgb_camera = None
    self.collision_sensor = None
    self._original_settings = None
    self._rgb_image = None
    self._tl_state = 'none'
    self._tl_distance = 999.0
    self._tracked_traffic_light = None
    self._tracked_stop_waypoint = None
    self._pending_curriculum_changes = {}
    self._last_control = {
        'accel_brake': 0.0,
        'throttle': 0.0,
        'brake': 0.0,
        'steer': 0.0,
    }

    self._setup_carla()
    self._setup_spaces()

  def _setup_carla(self):
    """Initialize CARLA client and world settings."""
    self.client = carla.Client(
        self.config['carla']['host'],
        self.config['carla']['port']
    )
    self.client.set_timeout(self.config['carla']['timeout'])

    # Load random map from phase distribution/curriculum
    maps = self._get_world_choices('maps')
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
        self.carla_map, sampling_resolution=self.route_sampling_resolution)

  def _apply_random_weather(self):
    """Apply a random weather preset from phase distribution."""
    weather_list = self._get_world_choices('weathers')
    if weather_list:
      preset_name = np.random.choice(weather_list)
      self.current_weather = preset_name
      weather_params = self.weather_presets[preset_name]
      self.world.set_weather(carla.WeatherParameters(**weather_params))

  def _get_world_choices(self, dimension: str):
    """Resolve map/weather choices from distribution, else curriculum at timestep 0."""
    distribution = self.phase_config.get('distribution', {})
    choices = distribution.get(dimension)
    if isinstance(choices, list) and len(choices) > 0:
      return choices

    curriculum = self.phase_config.get('curriculum', {})
    schedule = curriculum.get(dimension, [])
    if isinstance(schedule, list) and len(schedule) > 0:
      first_entry = sorted(schedule, key=lambda x: x['timesteps'])[0]
      scheduled_choices = first_entry.get('choices', [])
      if isinstance(scheduled_choices, list) and len(scheduled_choices) > 0:
        return scheduled_choices

    if dimension == 'maps':
      raise ValueError(
          "No map choices configured in distribution or curriculum")

    return ['clear_noon']

  def _get_initial_traffic_lights_enabled(self):
    """Return whether traffic lights should be active at timestep 0."""
    schedule = self.phase_config.get(
        'curriculum', {}).get('traffic_lights', [])
    if isinstance(schedule, list) and len(schedule) > 0:
      first_entry = sorted(schedule, key=lambda x: x['timesteps'])[0]
      return bool(first_entry.get('enabled', True))
    return True

  def queue_curriculum_change(self, dimension: str, value):
    """Queue a curriculum change to apply on the next reset."""
    self._pending_curriculum_changes[dimension] = value

  def apply_pending_curriculum_changes(self):
    """Apply all queued curriculum changes at the start of a new episode."""
    if not self._pending_curriculum_changes:
      return

    pending_changes = self._pending_curriculum_changes
    self._pending_curriculum_changes = {}

    for dimension, value in pending_changes.items():
      if dimension == 'episode_length' and value is not None:
        self.max_steps = int(value)
      elif dimension == 'maps':
        distribution = self.phase_config.setdefault('distribution', {})
        distribution['maps'] = list(value or [])
      elif dimension == 'traffic_lights':
        self.set_traffic_lights_enabled(value)
      elif dimension == 'weathers':
        distribution = self.phase_config.setdefault('distribution', {})
        distribution['weathers'] = list(value or [])

  def _traffic_lights_active(self):
    """Return whether traffic-light observations and penalties are active."""
    return bool(self.traffic_lights_enabled)

  def set_traffic_lights_enabled(self, enabled: bool):
    """Enable or disable all traffic-light effects without changing observation schema."""
    enabled = bool(enabled)
    if enabled == self.traffic_lights_enabled:
      return

    self.traffic_lights_enabled = enabled
    self._clear_tracked_traffic_light()
    self._tl_state = 'none'
    self._tl_distance = 999.0

    if not enabled:
      self._tl_index = {}
      return

    if self.world is not None:
      self._build_tl_index()

  def _setup_spaces(self):
    """Define direct-control action space and Dict observation space."""
    # Action space: steering [-1,1], signed accel/brake [-1,1]
    self.action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32
    )

    obs_config = self.config.get('observation', {})
    width = obs_config.get('width', 84)
    height = obs_config.get('height', 84)

    obs_dict = {
        "goal": spaces.Box(
            low=-1.0,
            high=1.0,
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
        ),
        "speed": spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        ),
        "target_speed": spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        ),
        "speed_error": spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        ),
        "last_action": spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        ),
        "lane_error_signed": spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        ),
    }

    if self.camera_enabled:
      obs_dict["image"] = spaces.Box(
          low=0,
          high=255,
          shape=(height, width, 3),
          dtype=np.uint8
      )

    self.observation_space = spaces.Dict(obs_dict)

  def _get_speed_observation_clip_kmh(self):
    """Return the clip value used for normalized speed observations."""
    return float(self.speed_config.get('max_forward_speed_kmh', 40.0))

  def _get_current_speed_kmh(self):
    """Return the vehicle's current speed in km/h."""
    if self.vehicle is None:
      return 0.0

    velocity = self.vehicle.get_velocity()
    return float(np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6)

  def _get_road_speed_limit_kmh(self):
    """Return the raw road speed limit in km/h with config fallback."""
    max_forward_speed_kmh = float(
        self.speed_config.get('max_forward_speed_kmh', 40.0))

    if self.vehicle is None:
      return max_forward_speed_kmh

    speed_limit_kmh = float(self.vehicle.get_speed_limit())
    if speed_limit_kmh <= 0.0:
      return max_forward_speed_kmh

    return speed_limit_kmh

  def _get_current_speed_limit_kmh(self):
    """Return the effective speed limit in km/h capped by the agent max speed."""
    max_forward_speed_kmh = float(
        self.speed_config.get('max_forward_speed_kmh', 40.0))
    return float(min(self._get_road_speed_limit_kmh(), max_forward_speed_kmh))

  def _normalize_speed_observation(self, speed_kmh):
    """Normalize a speed value to [0, 1] for observation usage."""
    clip_kmh = max(self._get_speed_observation_clip_kmh(), 1.0)
    return np.array(
        [float(np.clip(speed_kmh / clip_kmh, 0.0, 1.0))],
        dtype=np.float32,
    )

  def _get_reward_target_speed_kmh(self, traffic_light_state: str,
                                   distance_to_stop: float,
                                   speed_limit_kmh: float):
    """Return the reward target speed, using light context only when active."""
    return float(compute_target_speed(
        traffic_light_state,
        distance_to_stop,
        speed_limit_kmh,
        self.speed_config,
        traffic_lights_enabled=self._traffic_lights_active(),
    ))

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
    width = obs_config.get('width', 84)
    height = obs_config.get('height', 84)
    fov = obs_config.get('fov', 90)

    current_env_instance = weakref.ref(self)

    if self.camera_enabled:
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

  def _warmup_camera_if_needed(self):
    """Tick a few frames after reset so camera data is available."""
    if not self.camera_enabled or self.rgb_camera is None:
      return
    if self._rgb_image is not None:
      return

    for _ in range(self.camera_warmup_ticks):
      self.world.tick()
      if self._rgb_image is not None:
        return

    logging.debug(
        "Camera frame still unavailable after %d warmup ticks.",
        self.camera_warmup_ticks,
    )

  def _build_tl_index(self):
    """Build a lane-matched traffic-light index once per episode."""
    self._tl_index = {}
    if not self._traffic_lights_active():
      return

    for tl in self.world.get_actors().filter('traffic.traffic_light'):
      for stop_wp in tl.get_stop_waypoints():
        key = (stop_wp.road_id, stop_wp.lane_id)
        if key not in self._tl_index:
          self._tl_index[key] = (tl, stop_wp)

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

  def _compute_goal_vector(self):
    """Computes the goal vector to the next waypoint, returns the forward and lateral distance from the next waypoint."""
    if len(self.route) == 0 or self.current_waypoint_idx >= len(self.route):
      return np.array([0.0, 0.0], dtype=np.float32)

    vehicle_transform = self._step_transform
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

    forward_norm = float(np.clip(
        forward_distance / _GOAL_NORMALIZATION_METERS, -1.0, 1.0))
    lateral_norm = float(np.clip(
        lateral_offset / _GOAL_NORMALIZATION_METERS, -1.0, 1.0))

    return np.array([forward_norm, lateral_norm], dtype=np.float32)

  def _advance_route_progress(self, vehicle_location):
    """Advance past all waypoints already within reach using the pre-fetched location."""
    while (self.current_waypoint_idx < len(self.route) - 1 and
           vehicle_location.distance(
           self.route[self.current_waypoint_idx][0].transform.location) < self.route_progress_reached_distance_meters):
      self.current_waypoint_idx += 1

  def _compute_lane_metrics(self, vehicle_location):
    """Compute lane deviation, signed lateral error, and off-road status."""
    waypoint = self._get_nearest_route_waypoint(vehicle_location)
    max_lane_deviation = float(self.config.get(
        'lane_detection', {}).get('max_lane_deviation', 5.0))

    if waypoint is None:
      return max_lane_deviation, 0.0, True

    wp_location = waypoint.transform.location
    dx = float(vehicle_location.x - wp_location.x)
    dy = float(vehicle_location.y - wp_location.y)

    lane_deviation_2d = float(np.sqrt(dx**2 + dy**2))
    lane_deviation = min(lane_deviation_2d, max_lane_deviation)

    right_vector = waypoint.transform.get_right_vector()
    lane_error_signed = float(dx * right_vector.x + dy * right_vector.y)

    off_road_threshold = (waypoint.lane_width / 2.0) + \
        self.off_road_margin_meters
    off_road = lane_deviation_2d > off_road_threshold

    return lane_deviation, lane_error_signed, off_road

  def _normalize_signed_lane_error_observation(self, lane_error_signed_m):
    """Normalize signed lane error to [-1, 1]."""
    max_lane_deviation = max(float(self.config.get(
        'lane_detection', {}).get('max_lane_deviation', 5.0)), 1e-6)
    return np.array(
        [float(np.clip(lane_error_signed_m / max_lane_deviation, -1.0, 1.0))],
        dtype=np.float32,
    )

  def _get_nearest_route_waypoint_index(self, vehicle_location):
    """Return the index of the nearest waypoint on the active route."""
    if len(self.route) == 0:
      return None

    nearest_idx = None
    nearest_distance = float('inf')
    for idx in range(len(self.route)):
      route_location = self.route[idx][0].transform.location
      distance = float(np.sqrt(
          (vehicle_location.x - route_location.x)**2 +
          (vehicle_location.y - route_location.y)**2
      ))
      if distance < nearest_distance:
        nearest_distance = distance
        nearest_idx = idx

    return nearest_idx

  def _get_nearest_route_waypoint(self, vehicle_location):
    """Return the nearest waypoint anywhere on the active route."""
    nearest_idx = self._get_nearest_route_waypoint_index(vehicle_location)
    if nearest_idx is None:
      return None
    return self.route[nearest_idx][0]

  @staticmethod
  def _encode_traffic_light_state(traffic_light_state: str):
    """Encode traffic light state as one-hot: [none, red, yellow, green]."""
    tl_one_hot_map = {
        'none': np.array([1, 0, 0, 0], dtype=np.float32),
        'red': np.array([0, 1, 0, 0], dtype=np.float32),
        'yellow': np.array([0, 0, 1, 0], dtype=np.float32),
        'green': np.array([0, 0, 0, 1], dtype=np.float32),
    }
    return tl_one_hot_map.get(traffic_light_state, tl_one_hot_map['none'])

  @staticmethod
  def encode_stop_distance_observation(distance_to_stop: float):
    """Encode signed stop-line distance into [-1, 1]."""
    distance_clipped = np.clip(
        distance_to_stop,
        -_STOP_DISTANCE_CLIP_METERS,
        _STOP_DISTANCE_CLIP_METERS,
    )
    return np.array(
        [distance_clipped / _STOP_DISTANCE_CLIP_METERS],
        dtype=np.float32,
    )

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

  @staticmethod
  def _map_traffic_light_state(traffic_light):
    if traffic_light is None:
      return 'none'

    state_map = {
        carla.TrafficLightState.Red: 'red',
        carla.TrafficLightState.Yellow: 'yellow',
        carla.TrafficLightState.Green: 'green',
    }
    return state_map.get(traffic_light.get_state(), 'none')

  def _clear_tracked_traffic_light(self):
    self._tracked_traffic_light = None
    self._tracked_stop_waypoint = None

  def _update_tracked_traffic_light(self, traffic_light, stop_waypoint,
                                    vehicle_location):
    if traffic_light is None or stop_waypoint is None:
      self._clear_tracked_traffic_light()
      return 'none', 999.0

    distance_to_stop = self._compute_signed_distance_to_stop(
        vehicle_location, stop_waypoint)
    state = self._map_traffic_light_state(traffic_light)
    self._tracked_traffic_light = traffic_light
    self._tracked_stop_waypoint = stop_waypoint
    return state, distance_to_stop

  def _should_release_tracked_traffic_light(self, vehicle_location):
    if self._tracked_stop_waypoint is None:
      return True
    distance_to_stop = self._compute_signed_distance_to_stop(
        vehicle_location, self._tracked_stop_waypoint)
    return distance_to_stop > self.traffic_light_release_distance_meters

  def _find_relevant_route_traffic_light(self, vehicle_location, vehicle_forward):
    """Return the nearest route traffic light within approach range."""
    if len(self.route) == 0 or not self._tl_index:
      return None, None

    closest_dist = 999.0
    closest_tl = None
    closest_stop_waypoint = None
    lookahead_limit = min(
        self.current_waypoint_idx + self.traffic_light_lookahead_waypoints,
        len(self.route)
    )

    for idx in range(self.current_waypoint_idx, lookahead_limit):
      route_waypoint, _ = self.route[idx]
      entry = self._tl_index.get(
          (route_waypoint.road_id, route_waypoint.lane_id))
      if entry is None:
        continue

      traffic_light, stop_waypoint = entry
      stop_location = stop_waypoint.transform.location
      dx = stop_location.x - vehicle_location.x
      dy = stop_location.y - vehicle_location.y
      ahead = dx * vehicle_forward.x + dy * vehicle_forward.y
      dist = float((dx ** 2 + dy ** 2) ** 0.5)
      if ahead > 0.0 and dist < closest_dist and dist < self.traffic_light_detection_max_distance_meters:
        closest_dist = dist
        closest_tl = traffic_light
        closest_stop_waypoint = stop_waypoint

    return closest_tl, closest_stop_waypoint

  def _detect_traffic_light(self, vehicle_location):
    """Detect the nearest route traffic light and signed stop distance.

    Returns:
      (traffic_light_state, distance_to_stop): tuple of (str, float)
    """
    if not self._traffic_lights_active() or self.vehicle is None:
      self._clear_tracked_traffic_light()
      return 'none', 999.0

    vehicle_forward = self._step_transform.get_forward_vector()

    if self.vehicle.is_at_traffic_light():
      traffic_light = self.vehicle.get_traffic_light()
      stop_waypoints = traffic_light.get_stop_waypoints()
      stop_waypoint = stop_waypoints[0] if stop_waypoints else None
      if stop_waypoints:
        vehicle_waypoint = self.carla_map.get_waypoint(vehicle_location)
        if vehicle_waypoint is not None:
          for candidate in stop_waypoints:
            if candidate.road_id == vehicle_waypoint.road_id and candidate.lane_id == vehicle_waypoint.lane_id:
              stop_waypoint = candidate
              break

      return self._update_tracked_traffic_light(
          traffic_light, stop_waypoint, vehicle_location)

    if self._tracked_traffic_light is not None:
      if self._should_release_tracked_traffic_light(vehicle_location):
        self._clear_tracked_traffic_light()
      else:
        return self._update_tracked_traffic_light(
            self._tracked_traffic_light, self._tracked_stop_waypoint, vehicle_location)

    tl, stop_wp = self._find_relevant_route_traffic_light(
        vehicle_location, vehicle_forward)
    if tl is None:
      return 'none', 999.0
    return self._update_tracked_traffic_light(tl, stop_wp, vehicle_location)

  def _get_observation(self):
    """Return Dict observation with route context and target speed; image only when camera enabled."""
    goal = self._compute_goal_vector()
    current_speed_kmh = self._get_current_speed_kmh()
    speed_limit_kmh = self._get_current_speed_limit_kmh()
    _, lane_error_signed, _ = self._step_lane_metrics

    traffic_light_state = 'none'
    distance_to_stop = 999.0
    if self._traffic_lights_active():
      traffic_light_state = self._tl_state
      distance_to_stop = self._tl_distance

    reward_target_speed_kmh = self._get_reward_target_speed_kmh(
        traffic_light_state, distance_to_stop, speed_limit_kmh)

    clip_kmh = max(self._get_speed_observation_clip_kmh(), 1.0)
    speed_error_norm = np.array(
        [float(np.clip(
            (current_speed_kmh - reward_target_speed_kmh) / clip_kmh,
            -1.0,
            1.0,
        ))],
        dtype=np.float32,
    )

    tl_one_hot = self._encode_traffic_light_state(traffic_light_state)
    distance_normalized = self.encode_stop_distance_observation(
        distance_to_stop)
    last_action = (self.last_action.copy()
                   if self.last_action is not None
                   else np.array([0.0, 0.0], dtype=np.float32))

    obs = {
        "goal": goal,
        "traffic_light": tl_one_hot,
        "distance_to_stop": distance_normalized,
        "speed": self._normalize_speed_observation(current_speed_kmh),
        "target_speed": self._normalize_speed_observation(reward_target_speed_kmh),
        "speed_error": speed_error_norm,
        "last_action": last_action.astype(np.float32),
        "lane_error_signed": self._normalize_signed_lane_error_observation(lane_error_signed),
    }

    if self.camera_enabled:
      obs["image"] = (self._rgb_image.copy()
                      if self._rgb_image is not None
                      else np.zeros(self.observation_space["image"].shape, dtype=np.uint8))

    return obs

  def _get_vehicle_state(self):
    """Get current vehicle state for reward calculation and info."""
    vehicle_transform = self._step_transform
    vehicle_location = vehicle_transform.location
    speed = self._get_current_speed_kmh()
    road_speed_limit_kmh = self._get_road_speed_limit_kmh()
    effective_speed_limit_kmh = self._get_current_speed_limit_kmh()

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

    lane_deviation, lane_error_signed, off_road = self._step_lane_metrics

    traffic_light_state = 'none'
    distance_to_stop = 999.0
    if self._traffic_lights_active():
      traffic_light_state = self._tl_state
      distance_to_stop = self._tl_distance

    cruise_target_speed_kmh = float(compute_cruise_speed_target(
        effective_speed_limit_kmh, self.speed_config))
    reward_target_speed_kmh = self._get_reward_target_speed_kmh(
        traffic_light_state, distance_to_stop, effective_speed_limit_kmh)
    traffic_light_violation = (
        self._traffic_lights_active() and
        traffic_light_state in ('red', 'yellow') and
        distance_to_stop > 1.0
    )

    state = {
        'location': (vehicle_location.x, vehicle_location.y, vehicle_location.z),
        'rotation': (vehicle_transform.rotation.pitch, vehicle_transform.rotation.yaw, vehicle_transform.rotation.roll),
        'speed': speed,
        'distance_to_destination': distance_to_dest,
        'waypoint_distance': waypoint_distance,
        'current_waypoint_idx': self.current_waypoint_idx,
        'collision': self.collision_occurred,
        'speed_limit_kmh': road_speed_limit_kmh,
        'effective_speed_limit_kmh': effective_speed_limit_kmh,
        'cruise_target_speed_kmh': cruise_target_speed_kmh,
        'reward_target_speed_kmh': reward_target_speed_kmh,
        'lane_deviation': lane_deviation,
        'lane_error_signed': lane_error_signed,
        'off_road': off_road,
        'off_road_steps': self._off_road_steps,
        'traffic_light_state': traffic_light_state,
        'distance_to_stop': distance_to_stop,
        'traffic_light_violation': traffic_light_violation,
        'traffic_lights_enabled': self._traffic_lights_active(),
        'speed_error_kmh': float(speed - reward_target_speed_kmh),
    }

    if self.debug_env_info:
      state.update({
          'tracked_light_state': traffic_light_state,
          'tracked_light_distance': distance_to_stop,
      })

    return state

  def _cleanup_actors(self):
    """Destroy all spawned actors in safe order to prevent memory leaks and stop callbacks: stop sensors, tick, destroy."""
    if self.rgb_camera is not None:
      self.rgb_camera.stop()
    if self.collision_sensor is not None:
      self.collision_sensor.stop()

    # Destroy actors
    for actor in [self.collision_sensor, self.rgb_camera, self.vehicle]:
      if actor is not None and actor.is_alive:
        actor.destroy()

    # Reset references
    self.vehicle = None
    self.rgb_camera = None
    self.collision_sensor = None
    self._rgb_image = None
    self.prev_state = None
    self.last_action = None
    self._step_transform = None
    self._step_lane_metrics = None
    self._tl_index = {}
    self._tl_state = 'none'
    self._tl_distance = 999.0
    self._clear_tracked_traffic_light()
    self._off_road_steps = 0
    self._last_control = {
        'accel_brake': 0.0,
        'throttle': 0.0,
        'brake': 0.0,
        'steer': 0.0,
    }

  def reset(self, seed=None, options=None):
    """Reset the environment every episode with randomized elements each time."""
    super().reset(seed=seed)

    self._cleanup_actors()
    self.apply_pending_curriculum_changes()

    # Increment episode counter
    self.episode_count += 1

    # Check if map should be changed based on frequency
    if self.episode_count % self.map_change_frequency == 0:
      maps = self._get_world_choices('maps')
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
            self.carla_map, sampling_resolution=self.route_sampling_resolution)

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
    self._clear_tracked_traffic_light()

    # Calculate initial distance to destination
    spawn_loc = self.spawn_points[self.spawn_idx].location
    dest_loc = self.spawn_points[self.dest_idx].location
    self.initial_distance = spawn_loc.distance(dest_loc)

    # Compute route from spawn to destination
    self.route = self.route_planner.trace_route(spawn_loc, dest_loc)

    # Start at the first waypoint beyond the advance threshold so WP 0 is never silently consumed
    self.current_waypoint_idx = 0
    for idx, (wp, _) in enumerate(self.route):
      if spawn_loc.distance(wp.transform.location) > self.route_progress_reached_distance_meters:
        self.current_waypoint_idx = idx
        break

    self._build_tl_index()

    # Tick to get first observation
    self.world.tick()
    self._warmup_camera_if_needed()

    self._step_transform = self.vehicle.get_transform()
    vehicle_location = self._step_transform.location
    self._tl_state, self._tl_distance = self._detect_traffic_light(
        vehicle_location)
    self._step_lane_metrics = self._compute_lane_metrics(vehicle_location)

    observation = self._get_observation()
    info = self._get_vehicle_state()
    info['initial_distance'] = self.initial_distance
    info['map'] = self.current_map
    info['weather'] = self.current_weather
    info.update(self._last_control)

    return observation, info

  def step(self, action):
    """Execute one environment step."""
    steer_command = float(action[0])
    accel_brake_command = float(np.clip(action[1], -1.0, 1.0))
    prev_action_for_reward = (self.last_action.copy()
                              if self.last_action is not None
                              else np.zeros(2, dtype=np.float32))

    if accel_brake_command >= 0.0:
      throttle = float(accel_brake_command)
      brake = 0.0
    else:
      throttle = 0.0
      brake = float(abs(accel_brake_command))

    control = carla.VehicleControl(
        steer=steer_command,
        throttle=throttle,
        brake=brake
    )
    self._last_control = {
        'accel_brake': float(accel_brake_command),
        'throttle': float(throttle),
        'brake': float(brake),
        'steer': float(steer_command),
    }
    self.vehicle.apply_control(control)
    self.world.tick()
    self.step_count += 1

    self._step_transform = self.vehicle.get_transform()
    vehicle_location = self._step_transform.location
    self._advance_route_progress(vehicle_location)
    self._tl_state, self._tl_distance = self._detect_traffic_light(
        vehicle_location)
    self._step_lane_metrics = self._compute_lane_metrics(vehicle_location)

    _, lane_error_signed, _ = self._step_lane_metrics
    max_lane_deviation = float(self.config.get(
        'lane_detection', {}).get('max_lane_deviation', 5.0))
    lane_error_maxed = abs(lane_error_signed) >= max_lane_deviation
    self._off_road_steps = self._off_road_steps + 1 if lane_error_maxed else 0

    self.last_action = action.copy()

    observation = self._get_observation()
    state = self._get_vehicle_state()

    # Reward calculation with previous state for progress/smoothness
    reward = 0.0
    reward_components = None
    if self.reward_fn is not None:
      # Pass action and prev_action separately; state is physical state only
      if self.reward_fn is compute_reward:
        reward, reward_components = compute_reward_with_components(
            state, action, self.prev_state, prev_action_for_reward,
            self.reward_weights, self.speed_config)
      else:
        reward = self.reward_fn(state, action, self.prev_state, prev_action_for_reward,
                                self.reward_weights, self.speed_config)

    # Store current state/action for next step
    self.prev_state = state.copy()

    # Termination: collision or reached destination
    terminated = self.collision_occurred
    if state['distance_to_destination'] < self.destination_success_distance_meters:
      terminated = True

    if state['traffic_light_violation']:
      terminated = True

    if self._off_road_steps >= self._off_road_termination_steps:
      terminated = True

    # Episode timeout: max steps reached
    episode_timeout = self.step_count >= self.max_steps

    info = state
    info['step'] = self.step_count
    info['initial_distance'] = self.initial_distance
    info.update(self._last_control)
    if reward_components is not None:
      info['reward_components'] = reward_components

    return observation, reward, terminated, episode_timeout, info

  def close(self):
    """Clean up CARLA resources."""
    self._cleanup_actors()
    if self._original_settings is not None:
      self.world.apply_settings(self._original_settings)
