import numpy as np
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor

class WorldConfig:
  """Manages CARLA world randomization: maps, weather, traffic."""

  def __init__(self, client, phase_config, weather_presets, traffic_presets, mode):
    self.client = client
    self.world = None

    # World randomization state
    self.available_maps = []
    self.available_weathers = []
    self.available_traffic_choices = ['none']
    self.current_map = None
    self.current_weather = None
    self.current_traffic = 'none'
    self.weather_presets = weather_presets
    self.traffic_presets = traffic_presets

    # TrafficManager handle (initialized once in setup_initial_world)
    self.traffic_manager = None

    # NPC actor tracking for cleanup
    self.npc_vehicle_ids = []
    self.npc_walker_ids = []
    self.npc_walker_controller_ids = []

    # Map change frequency based on mode
    map_config = phase_config.get('map_randomization', {})
    if mode == 'train':
      self.map_change_frequency = map_config.get(
          'train_map_randomness_frequency', 1)
    elif mode == 'test':
      self.map_change_frequency = map_config.get(
          'test_map_randomness_frequency', 1)
    else:
      self.map_change_frequency = 1

    self.episode_counter = 0

  def setup_initial_world(self, curriculum, rng):
    """Load initial map, weather, and traffic. Returns (world, carla_map, spawn_points, route_planner)."""
    maps = curriculum.get('maps', [{}])[0].get('choices', [])
    self.available_maps = maps
    self.current_map = rng.choice(maps)

    self.world = self.client.load_world(self.current_map)

    settings = self.world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 60.0
    self.world.apply_settings(settings)

    weathers = curriculum.get('weathers', [{}])[
        0].get('choices', ['clear_noon'])
    self.available_weathers = weathers
    self.current_weather = self._apply_weather(rng)

    self.available_traffic_choices = curriculum.get(
        'traffic', [{}])[0].get('choices', ['none'])
    self._setup_traffic_manager()
    self._apply_traffic(rng)

    carla_map = self.world.get_map()
    spawn_points = carla_map.get_spawn_points()
    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

    return self.world, carla_map, spawn_points, route_planner

  def reset_episode(self, curriculum, rng):
    """Handle map/weather/traffic randomization for new episode. Returns (world, carla_map, spawn_points, route_planner) or None."""
    self.episode_counter += 1

    maps = self.available_maps if self.available_maps else []
    if not maps:
      maps = curriculum.get('maps', [{}])[0].get('choices', [])
      self.available_maps = maps

    # Check if map should change
    new_map = self._select_map(maps, rng)
    map_changed = new_map != self.current_map

    if map_changed:
      # load_world() destroys all actors — clear tracking lists before reload
      self.npc_vehicle_ids = []
      self.npc_walker_ids = []
      self.npc_walker_controller_ids = []

      self.current_map = new_map
      self.world = self.client.load_world(self.current_map)

      settings = self.world.get_settings()
      settings.synchronous_mode = True
      settings.fixed_delta_seconds = 1.0 / 60.0
      self.world.apply_settings(settings)

      # TM must be re-synced after world reload
      self._setup_traffic_manager()

      carla_map = self.world.get_map()
      spawn_points = carla_map.get_spawn_points()
      route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

      result = (self.world, carla_map, spawn_points, route_planner)
    else:
      result = None

    # Always apply weather and traffic every episode
    self.current_weather = self._apply_weather(rng)
    self._apply_traffic(rng)

    return result

  def update_map_choices(self, maps: list):
    """Update available maps for curriculum progression."""
    self.available_maps = maps

  def update_weather_choices(self, weathers: list):
    """Update available weathers for curriculum progression."""
    self.available_weathers = weathers

  def update_traffic_choices(self, choices: list):
    """Update available traffic presets for curriculum progression."""
    self.available_traffic_choices = choices

  def teardown_traffic(self):
    """Public method for environment close() to clean up all NPC actors."""
    self._teardown_traffic()

  def _setup_traffic_manager(self):
    """Initialize TrafficManager in synchronous mode with full physics for all NPC vehicles."""
    self.traffic_manager = self.client.get_trafficmanager(8000)
    self.traffic_manager.set_synchronous_mode(True)

  def _spawn_vehicles(self, preset, rng):
    """Batch-spawn NPC vehicles under TrafficManager autopilot."""
    num_vehicles = preset.get('num_vehicles', 0)
    if num_vehicles <= 0:
      return

    blueprints = self.world.get_blueprint_library().filter('vehicle.*')
    blueprints = [
        bp for bp in blueprints if bp.get_attribute('base_type') == 'car']
    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = list(self.world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    spawn_points = spawn_points[:num_vehicles]

    tm_port = self.traffic_manager.get_port()
    batch = []
    for transform in spawn_points:
      bp = blueprints[int(rng.integers(0, len(blueprints)))]
      if bp.has_attribute('color'):
        color = bp.get_attribute('color').recommended_values
        bp.set_attribute('color', color[int(rng.integers(0, len(color)))])
      bp.set_attribute('role_name', 'autopilot')
      batch.append(SpawnActor(bp, transform).then(
          SetAutopilot(FutureActor, True, tm_port)))

    for response in self.client.apply_batch_sync(batch, True):
      if not response.error:
        self.npc_vehicle_ids.append(response.actor_id)

    # Apply per-preset TM tuning
    speed_diff = preset.get('global_speed_difference', 30.0)
    lead_dist = preset.get('distance_to_leading_vehicle', 2.5)
    self.traffic_manager.global_percentage_speed_difference(speed_diff)
    self.traffic_manager.set_global_distance_to_leading_vehicle(lead_dist)

  def _spawn_walkers(self, preset, rng):
    """Batch-spawn pedestrians with AI controllers (3-phase: actors, controllers, start)."""
    num_walkers = preset.get('num_walkers', 0)
    if num_walkers <= 0:
      return

    walker_bp_library = self.world.get_blueprint_library().filter('walker.pedestrian.*')
    controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

    # Phase A: spawn walker actors
    spawn_points = []
    walker_speeds = []
    for _ in range(num_walkers):
      loc = self.world.get_random_location_from_navigation()
      if loc is not None:
        transform = carla.Transform()
        transform.location = loc
        spawn_points.append(transform)

    batch = []
    for transform in spawn_points:
      bp = walker_bp_library[int(rng.integers(0, len(walker_bp_library)))]
      if bp.has_attribute('is_invincible'):
        bp.set_attribute('is_invincible', 'false')
      if bp.has_attribute('speed'):
        walker_speeds.append(
            float(bp.get_attribute('speed').recommended_values[1]))
      else:
        walker_speeds.append(1.4)
      batch.append(SpawnActor(bp, transform))

    walker_results = self.client.apply_batch_sync(batch, True)
    successful_walker_ids = []
    successful_speeds = []
    for i, response in enumerate(walker_results):
      if not response.error:
        self.npc_walker_ids.append(response.actor_id)
        successful_walker_ids.append(response.actor_id)
        successful_speeds.append(walker_speeds[i])

    if not successful_walker_ids:
      return

    # Phase B: spawn AI controllers attached to each walker
    batch = []
    for walker_id in successful_walker_ids:
      batch.append(SpawnActor(controller_bp, carla.Transform(), walker_id))

    controller_results = self.client.apply_batch_sync(batch, True)
    successful_controller_ids = []
    final_walker_ids = []
    final_speeds = []
    for i, response in enumerate(controller_results):
      if not response.error:
        self.npc_walker_controller_ids.append(response.actor_id)
        successful_controller_ids.append(response.actor_id)
        final_walker_ids.append(successful_walker_ids[i])
        final_speeds.append(successful_speeds[i])

    # Mandatory tick: controllers must receive walker transforms before start()
    self.world.tick()

    # Phase C: start each controller and send to a random navigation point
    all_actors = self.world.get_actors(successful_controller_ids)
    for i, controller in enumerate(all_actors):
      controller.start()
      controller.go_to_location(
          self.world.get_random_location_from_navigation())
      controller.set_max_speed(final_speeds[i])

  def _teardown_traffic(self):
    """Destroy all NPC vehicles and pedestrians synchronously."""
    if self.world is None:
      self.npc_vehicle_ids = []
      self.npc_walker_ids = []
      self.npc_walker_controller_ids = []
      return

    # Stop walker controllers before destroying
    if self.npc_walker_controller_ids:
      controllers = self.world.get_actors(self.npc_walker_controller_ids)
      for controller in controllers:
        controller.stop()

    all_walker_ids = self.npc_walker_controller_ids + self.npc_walker_ids
    if all_walker_ids:
      self.client.apply_batch_sync(
          [DestroyActor(x) for x in all_walker_ids], True)

    if self.npc_vehicle_ids:
      self.client.apply_batch_sync(
          [DestroyActor(x) for x in self.npc_vehicle_ids], True)

    if self.npc_vehicle_ids or all_walker_ids:
      self.world.tick()

    self.npc_vehicle_ids = []
    self.npc_walker_ids = []
    self.npc_walker_controller_ids = []

  def _apply_traffic(self, rng):
    """Select and apply a random traffic preset from the current allowed choices."""
    choices = self.available_traffic_choices if self.available_traffic_choices else [
        'none']
    preset_name = rng.choice(choices)
    preset = self.traffic_presets.get(preset_name, {})

    self._teardown_traffic()

    if preset_name != 'none':
      self._spawn_vehicles(preset, rng)
      self._spawn_walkers(preset, rng)

    self.current_traffic = preset_name
    return preset_name

  def _select_map(self, maps, rng):
    """Sample a map, respecting change frequency."""
    if self.episode_counter % self.map_change_frequency != 0:
      return self.current_map
    return rng.choice(maps)

  def _apply_weather(self, rng):
    """Apply random weather preset and return preset name."""
    weathers = self.available_weathers if self.available_weathers else [
        'clear_noon']
    preset_name = rng.choice(weathers)
    weather_params = self.weather_presets.get(preset_name, {})
    self.world.set_weather(carla.WeatherParameters(**weather_params))
    return preset_name
