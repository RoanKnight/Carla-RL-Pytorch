import carla

def main():
  client = carla.Client('localhost', 2000)
  client.set_timeout(10.0)
  print(client.get_available_maps())

  world = client.load_world('Town05')

  blueprint_library = world.get_blueprint_library()
  vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

  spawn_points = world.get_map().get_spawn_points()
  spawn_point = spawn_points[0]

  vehicle = world.spawn_actor(vehicle_bp, spawn_point)

  # Position spectator camera for third-person view behind and above the vehicle
  spectator = world.get_spectator()
  transform = vehicle.get_transform()
  # Calculate camera position: 5 meters behind vehicle, 2 meters above
  location = transform.location - transform.get_forward_vector() * 5 + \
      carla.Vector3D(z=2)
  rotation = transform.rotation
  spectator.set_transform(carla.Transform(location, rotation))

  # Run simulation loop to maintain view; interrupt with Ctrl+C
  try:
    while True:
      world.wait_for_tick()
      # Note: Spectator position could be updated here if vehicle moves
  finally:
    vehicle.destroy()

if __name__ == '__main__':
  main()
