import logging
import ctypes
import carla

CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CLIENT_TIMEOUT = 10.0
WORLD_MAP = 'Town05'

# Weather configuration
WEATHER_CONFIG = {
    'cloudiness': 100.0,
    'precipitation': 90.0,
    'precipitation_deposits': 80.0,
    'wind_intensity': 70.0,
    'sun_altitude_angle': 25.0,
    'sun_azimuth_angle': 70.0,
    'fog_density': 20.0,
    'wetness': 80.0
}

VEHICLE_MODEL = 'vehicle.tesla.model3'

CAMERA_DISTANCE_BEHIND = 6.0
CAMERA_HEIGHT_ABOVE = 2.0

# Function to get the monitor's refresh rate
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

  vehicle = world.spawn_actor(world.get_blueprint_library().filter(
      VEHICLE_MODEL)[0], world.get_map().get_spawn_points()[0])
  logging.info('Vehicle spawned')

  spectator = world.get_spectator()
  actors = [vehicle]

  try:
    while True:
      world.tick()
      # Position spectator behind and above vehicle
      transform = vehicle.get_transform()
      location = transform.location - transform.get_forward_vector() * CAMERA_DISTANCE_BEHIND + \
          carla.Vector3D(z=CAMERA_HEIGHT_ABOVE)
      spectator.set_transform(carla.Transform(location, transform.rotation))
  except KeyboardInterrupt:
    logging.info('Simulation stopped')
  finally:
    # Cleanup actors and restore settings
    for actor in actors:
      actor.destroy()
    world.apply_settings(original_settings)

if __name__ == '__main__':
  main()
