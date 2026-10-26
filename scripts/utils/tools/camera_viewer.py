# -*- coding: utf-8 -*-
# camera_follow_vehicle.py
import carla
import numpy as np
import cv2
import time

# Conexión con CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# Buscar vehículo existente
vehicles = world.get_actors().filter('vehicle.*')
if not vehicles:
    print("❌ No se encontró ningún vehículo activo en CARLA.")
    exit()

vehicle = vehicles[0]  # el primero (puedes mejorar esto por rol o tipo)
print(f"🚗 Siguiendo vehículo: {vehicle.type_id}")

# Crear cámara
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")
camera_bp.set_attribute("fov", "90")

# Posición detrás del vehículo (chase cam)
camera_transform = carla.Transform(
    carla.Location(x=-6, z=3),
    carla.Rotation(pitch=-10)
)

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Mostrar imagen
def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    img_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.imshow("CARLA - Cámara trasera", img_bgr)
    cv2.waitKey(1)

camera.listen(lambda image: camera_callback(image))

try:
    print("📸 Mostrando cámara trasera del vehículo. Presiona 'q' para salir.")
    while True:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    print("🛑 Cerrando visor.")
    camera.stop()
    camera.destroy()
    cv2.destroyAllWindows()
