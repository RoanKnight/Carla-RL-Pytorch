import logging
import carla
from utils import load_config, setup_logging
from environment import CarlaEnv

# Load configuration
CONFIG = load_config()

def main():
  setup_logging()
  logging.info('Setting up CARLA environment')

  # Create environment using the CarlaEnv class
  env = CarlaEnv()
  obs, info = env.reset()

  # Get CARLA objects for visualization
  world = env.world
  vehicle = env.vehicle
  spectator = world.get_spectator()

  # Get destination location for visualization
  destination_location = None
  dest_idx = CONFIG['vehicle']['destination_index']
  if dest_idx is not None and dest_idx < len(env.spawn_points):
    destination_location = env.spawn_points[dest_idx].location
    logging.info(f'Destination set at spawn point {dest_idx}')

  try:
    while True:
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

      # Tick the world for spectator updates
      world.tick()

  except KeyboardInterrupt:
    logging.info('Simulation stopped')
  finally:
    env.close()

if __name__ == '__main__':
  main()
