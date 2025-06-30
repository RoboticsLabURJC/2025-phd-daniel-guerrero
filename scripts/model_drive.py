# model_drive.py
import carla
import torch
import torch.serialization
import cv2
import numpy as np
import time

from model import CarlaPilotNet  # Asegúrate de que model.py está en el mismo directorio

# 🛡️ Permitir que PyTorch cargue esta clase del archivo .pth
torch.serialization.add_safe_globals([CarlaPilotNet])

# Cargar modelo completo (estructura + pesos)
model = torch.load("carla_model.pth", map_location="cpu", weights_only=False)
model.eval()

# Conexión con CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# Crear vehículo
vehicle_bp = bp_lib.filter("model3")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Crear cámara RGB
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "200")
camera_bp.set_attribute("image_size_y", "66")
camera_bp.set_attribute("fov", "90")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Preprocesamiento
def preprocess_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]  # RGB
    array = array.astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))  # (C, H, W)
    tensor = torch.tensor(array).unsqueeze(0)
    return tensor

# Aplicar predicción al vehículo
def control_vehicle(preds):
    steer, throttle, brake = preds[0].tolist()
    control = carla.VehicleControl(
        steer=float(steer),
        throttle=max(0.0, min(1.0, float(throttle))),
        brake=max(0.0, min(1.0, float(brake)))
    )
    vehicle.apply_control(control)

# Callback principal
def camera_callback(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        preds = model(input_tensor)
    control_vehicle(preds)

# Iniciar cámara
camera.listen(lambda image: camera_callback(image))

# Ejecutar por 2 minutos
try:
    print("🚗 Conducción autónoma en curso (Ctrl+C para salir)...")
    time.sleep(500)
finally:
    print("🛑 Deteniendo vehículo.")
    camera.stop()
    vehicle.destroy()
