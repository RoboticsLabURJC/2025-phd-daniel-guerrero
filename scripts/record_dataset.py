import carla
import os
import csv
import time
import numpy as np

# Configuración
SAVE_PATH = "dataset"
IMG_DIR = os.path.join(SAVE_PATH, "images")
os.makedirs(IMG_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_PATH, "controls.csv")

csv_file = open(CSV_PATH, mode="w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "steer", "throttle", "brake"])

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.filter("model3")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

frame_id = 0
controls_dict = {}

def save_image(image):
    global frame_id
    filename = f"frame_{frame_id:05d}.png"
    image.save_to_disk(os.path.join(IMG_DIR, filename))
    control = controls_dict.get(image.frame, None)
    if control:
        writer.writerow([filename, control.steer, control.throttle, control.brake])
        print(f"[✓] Saved: {filename}")
        frame_id += 1

def capture_control():
    while True:
        control = vehicle.get_control()
        controls_dict[vehicle.get_world().get_snapshot().frame] = control
        time.sleep(0.05)

import threading
control_thread = threading.Thread(target=capture_control)
control_thread.start()

camera.listen(lambda image: save_image(image))

try:
    print("[INFO] Grabando por 60 segundos...")
    time.sleep(60*20)
finally:
    print("[INFO] Terminando grabación...")
    camera.stop()
    vehicle.destroy()
    csv_file.close()
