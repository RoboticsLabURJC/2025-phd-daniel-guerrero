import os
import time
from queue import Queue, Empty

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import carla

# ---------- Parámetros de Inferencia ----------
HOST = "127.0.0.1"
PORT = 2000
MODEL_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/model_training/pilotnet_best.pth"

# Control de Velocidad Constante
TARGET_SPEED_KMH = 10.0  # Ajusta la velocidad que desees aquí
KP_THROTTLE = 0.15       # Ganancia proporcional para acelerar
KP_BRAKE = 0.5           # Ganancia proporcional para frenar

IMG_W, IMG_H = 640, 360
CAM_FPS = 20
FIXED_DT = 1.0 / CAM_FPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------

# 1. RE-DEFINICIÓN DE LA ARQUITECTURA (Debe ser idéntica al entrenamiento)
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU()
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 1164), nn.ELU(),
            nn.Linear(1164, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x

# 2. FUNCIONES DE PROCESAMIENTO
def to_bgr(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy() # Importante el .copy()

def apply_half_mask(frame_bgr):
    h = frame_bgr.shape[0]
    frame_bgr[0:h//2, :] = 0
    return frame_bgr

def get_speed(vehicle):
    """Devuelve la velocidad actual del vehículo en km/h"""
    vel = vehicle.get_velocity()
    return 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

# 3. BUCLE PRINCIPAL
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No se encontró el modelo: {MODEL_PATH}")
        return

    # Cargar el modelo entrenado
    print(f"[INFO] Cargando modelo en {device}...")
    model = PilotNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Modo evaluación (desactiva dropout/batchnorm si los hubiera)

    # Transformaciones idénticas a las de entrenamiento
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0) # Le damos más tiempo por si tiene que cargar el mapa
    
    DESIRED_MAP = "Town03"
    current_map = client.get_world().get_map().name
    
    if DESIRED_MAP not in current_map:
        print(f"[INFO] Cambiando el mapa a {DESIRED_MAP}...")
        world = client.load_world(DESIRED_MAP)
    else:
        print(f"[INFO] Ya estás en {DESIRED_MAP}")
        world = client.get_world()
        
    original_settings = world.get_settings()

    vehicle = None
    camera = None
    image_queue = Queue()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        
        # Spawnear Vehículo
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

        # Spawnear Cámara RGB
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", str(FIXED_DT))

        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)

        print("[INFO] Autopilot Iniciado. Presiona 'Q' en la ventana de OpenCV para salir.")
        control = carla.VehicleControl()

        while True:
            world.tick()
            
            try:
                img = image_queue.get(timeout=2.0)
            except Empty:
                continue

            # 1. Procesar la imagen recibida
            frame = to_bgr(img)
            frame_masked = apply_half_mask(frame)
            
            # Preparar tensor para PyTorch
            frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device) # Añadir dimensión de Batch

            # 2. Inferencia: Predecir Ángulo del Volante
            with torch.no_grad():
                steer_pred = model(input_tensor).item()

            # 3. Control Longitudinal: Mantener Velocidad
            current_speed = get_speed(vehicle)
            speed_error = TARGET_SPEED_KMH - current_speed
            
            throttle = 0.0
            brake = 0.0
            
            if speed_error > 0:
                throttle = min(1.0, float(speed_error * KP_THROTTLE))
            else:
                brake = min(1.0, float(-speed_error * KP_BRAKE))

            # 4. Aplicar Controles al Vehículo
            control.steer = float(np.clip(steer_pred, -1.0, 1.0))
            control.throttle = throttle
            control.brake = brake
            vehicle.apply_control(control)

            # 5. Visualización
            status = f"Speed: {current_speed:.1f}/{TARGET_SPEED_KMH} km/h | Steer: {steer_pred:.3f}"
            cv2.putText(frame_masked, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("PilotNet Inference", frame_masked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[INFO] Finalizando Inferencia. Limpiando entorno...")
        world.apply_settings(original_settings)
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()