import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

print("Reproduciendo grabación...")
client.replay_file("my_record.log", 0, 0, 0)
