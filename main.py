import logging
import carla
from utils import load_config, get_monitor_refresh_rate, setup_logging

# Load configuration
CONFIG = load_config()

MONITOR_REFRESH_RATE = get_monitor_refresh_rate()
FIXED_DELTA_SECONDS = 1 / MONITOR_REFRESH_RATE

def main():
  setup_logging()
  client = carla.Client(CONFIG['carla']['host'], CONFIG['carla']['port'])
  client.set_timeout(CONFIG['carla']['timeout'])
  logging.info(
      f'Connected to CARLA (targeting {MONITOR_REFRESH_RATE}Hz refresh rate)')

  world = client.load_world(CONFIG['world']['map'])
  original_settings = world.get_settings()
  settings = world.get_settings()
  settings.synchronous_mode = CONFIG['world']['synchronous_mode']
  settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
  world.apply_settings(settings)
  world.set_weather(carla.WeatherParameters(**CONFIG['weather']))
  logging.info('World configured')

  # Get spawn points and ensure that the spawn point index is valid
  spawn_points = world.get_map().get_spawn_points()
  if CONFIG['vehicle']['spawn_point_index'] >= len(spawn_points):
    logging.warning(
        f'Spawn point index {CONFIG["vehicle"]["spawn_point_index"]} not available. Map {CONFIG["world"]["map"]} has {len(spawn_points)} spawn points. Using index 0.')
    spawn_point_index = 0
  else:
    spawn_point_index = CONFIG['vehicle']['spawn_point_index']

  # Spawn the vehicle at the spawn point
  vehicle = world.spawn_actor(world.get_blueprint_library().filter(
      CONFIG['vehicle']['model'])[0], spawn_points[spawn_point_index])
  logging.info(f'Vehicle spawned at spawn point {spawn_point_index}')

  destination_location = None
  if CONFIG['vehicle']['destination_index'] is not None:
    if CONFIG['vehicle']['destination_index'] >= len(spawn_points):
      logging.warning(
          f'Destination index {CONFIG["vehicle"]["destination_index"]} not available. Map {CONFIG["world"]["map"]} has {len(spawn_points)} spawn points.')
    else:
      destination_location = spawn_points[CONFIG['vehicle']['destination_index']].location
      logging.info(f'Destination set at spawn point {CONFIG["vehicle"]["destination_index"]}')

  spectator = world.get_spectator()
  actors = [vehicle]

  try:
    while True:
      world.tick()
      # Position spectator relatively to allow the camera to rotate freely using the mouse
      spectator_transform = spectator.get_transform()
      transform = vehicle.get_transform()
      forward = spectator_transform.get_forward_vector()
      location = transform.location - forward * CONFIG['camera']['distance_behind'] + \
          carla.Vector3D(z=CONFIG['camera']['height_above'])
      spectator.set_transform(
          carla.Transform(location, spectator_transform.rotation))

      if destination_location:
        world.debug.draw_line(
            destination_location,
            destination_location + carla.Location(z=150),
            thickness=0.4,
            color=carla.Color(255, 0, 0),
            life_time=0.1)
  except KeyboardInterrupt:
    logging.info('Simulation stopped')
  finally:
    # Cleanup actors and restore settings
    for actor in actors:
      actor.destroy()
    world.apply_settings(original_settings)

if __name__ == '__main__':
  main()
