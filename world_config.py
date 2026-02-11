import numpy as np
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

class WorldConfig:
  """Manages CARLA world randomization: maps, weather, traffic."""
  
  def __init__(self, client, phase_config, weather_presets, mode):
    self.client = client
    self.world = None
    
    # World randomization state
    self.available_maps = []
    self.available_weathers = []
    self.available_traffic_densities = []
    self.current_map = None
    self.current_weather = None
    self.traffic_density = 'none'
    self.weather_presets = weather_presets
    
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
    """Load initial map and weather. Returns (world, carla_map, spawn_points, route_planner)."""
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

    carla_map = self.world.get_map()
    spawn_points = carla_map.get_spawn_points()
    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

    return self.world, carla_map, spawn_points, route_planner

  def reset_episode(self, curriculum, rng):
    """Handle map/weather randomization for new episode. Returns (world, carla_map, spawn_points, route_planner) or None."""
    self.episode_counter += 1

    maps = self.available_maps if self.available_maps else []
    if not maps:
      maps = curriculum.get('maps', [{}])[0].get('choices', [])
      self.available_maps = maps

    # Check if map should change
    new_map = self._select_map(maps, rng)
    map_changed = new_map != self.current_map

    if map_changed:
      self.current_map = new_map
      self.world = self.client.load_world(self.current_map)

      settings = self.world.get_settings()
      settings.synchronous_mode = True
      settings.fixed_delta_seconds = 1.0 / 60.0
      self.world.apply_settings(settings)

      carla_map = self.world.get_map()
      spawn_points = carla_map.get_spawn_points()
      route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

      result = (self.world, carla_map, spawn_points, route_planner)
    else:
      result = None

    # Always apply weather (every episode)
    self.current_weather = self._apply_weather(rng)

    return result

  def update_map_choices(self, maps: list):
    """Update available maps for curriculum progression."""
    self.available_maps = maps

  def update_weather_choices(self, weathers: list):
    """Update available weathers for curriculum progression."""
    self.available_weathers = weathers

  def update_traffic_density(self, density: str):
    """Update traffic density for curriculum progression."""
    self.traffic_density = density
  
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
