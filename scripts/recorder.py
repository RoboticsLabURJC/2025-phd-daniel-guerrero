import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

print("Iniciando grabación...")
client.start_recorder("my_record.log")

world = client.get_world()
vehicle_bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

time.sleep(30)

print("Deteniendo grabación...")
client.stop_recorder()

vehicle.destroy()
print("Grabación finalizada: my_record.log")
