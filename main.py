import logging
import ctypes
import carla

CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CLIENT_TIMEOUT = 10.0
WORLD_MAP = 'Town03'

# Weather configuration
WEATHER_CONFIG = {
    'cloudiness': 30.0,
    'precipitation': 70.0,
    'precipitation_deposits': 70.0,
    'wind_intensity': 30.0,
    'sun_altitude_angle': 90.0,
    'sun_azimuth_angle': 180.0,
    'fog_density': 10.0,
    'wetness': 70.0
}

VEHICLE_MODEL = 'vehicle.tesla.model3'
SPAWN_POINT_INDEX = 8
DESTINATION_INDEX = 44

CAMERA_DISTANCE_BEHIND = 6.0
CAMERA_HEIGHT_ABOVE = 2.0

def get_monitor_refresh_rate():
  user32 = ctypes.windll.user32
  dc = user32.GetDC(0)
  refresh_rate = ctypes.windll.gdi32.GetDeviceCaps(dc, 116)
  user32.ReleaseDC(0, dc)
  return refresh_rate if refresh_rate > 0 else 60

MONITOR_REFRESH_RATE = get_monitor_refresh_rate()

# Set simulation to synchronous mode and match monitor refresh rate
SYNCHRONOUS_MODE = True
FIXED_DELTA_SECONDS = 1 / MONITOR_REFRESH_RATE

def main():
  logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
  client = carla.Client(CARLA_HOST, CARLA_PORT)
  client.set_timeout(CLIENT_TIMEOUT)
  logging.info(
      f'Connected to CARLA (targeting {MONITOR_REFRESH_RATE}Hz refresh rate)')

  world = client.load_world(WORLD_MAP)
  original_settings = world.get_settings()
  settings = world.get_settings()
  settings.synchronous_mode = SYNCHRONOUS_MODE
  settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
  world.apply_settings(settings)
  world.set_weather(carla.WeatherParameters(**WEATHER_CONFIG))
  logging.info('World configured')

  # Get spawn points and ensure that the spawn point index is valid
  spawn_points = world.get_map().get_spawn_points()
  if SPAWN_POINT_INDEX >= len(spawn_points):
    logging.warning(
        f'Spawn point index {SPAWN_POINT_INDEX} not available. Map {WORLD_MAP} has {len(spawn_points)} spawn points. Using index 0.')
    spawn_point_index = 0
  else:
    spawn_point_index = SPAWN_POINT_INDEX

  # Spawn the vehicle at the spawn point
  vehicle = world.spawn_actor(world.get_blueprint_library().filter(
      VEHICLE_MODEL)[0], spawn_points[spawn_point_index])
  logging.info(f'Vehicle spawned at spawn point {spawn_point_index}')

  destination_location = None
  if DESTINATION_INDEX is not None:
    if DESTINATION_INDEX >= len(spawn_points):
      logging.warning(
          f'Destination index {DESTINATION_INDEX} not available. Map {WORLD_MAP} has {len(spawn_points)} spawn points.')
    else:
      destination_location = spawn_points[DESTINATION_INDEX].location
      logging.info(f'Destination set at spawn point {DESTINATION_INDEX}')

  spectator = world.get_spectator()
  actors = [vehicle]

  try:
    while True:
      world.tick()
      # Position spectator relatively to allow the camera to rotate freely using the mouse
      spectator_transform = spectator.get_transform()
      transform = vehicle.get_transform()
      forward = spectator_transform.get_forward_vector()
      location = transform.location - forward * CAMERA_DISTANCE_BEHIND + \
          carla.Vector3D(z=CAMERA_HEIGHT_ABOVE)
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
