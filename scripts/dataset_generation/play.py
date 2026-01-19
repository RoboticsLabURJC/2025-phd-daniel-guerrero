import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

print("Reproduciendo grabación...")
client.replay_file("town01.log", 0, 0, 0)
