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

# Control de Velocidad Constante (Optimizado para el Loop)
TARGET_SPEED_KMH = 15.0  # Incrementado para probar estabilidad en curva
KP_THROTTLE = 0.20       
KP_BRAKE = 0.5           

IMG_W, IMG_H = 640, 360
CAM_FPS = 20
FIXED_DT = 1.0 / CAM_FPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------

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

def to_bgr(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()

def apply_half_mask(frame_bgr):
    h = frame_bgr.shape[0]
    frame_bgr[0:h//2, :] = 0
    return frame_bgr

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No se encontró el modelo: {MODEL_PATH}")
        return

    print(f"[INFO] Cargando modelo PilotNet en {device}...")
    model = PilotNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    
    # --- CAMBIO A TOWN04 (EL LOOP) ---
    DESIRED_MAP = "Town04"
    try:
        world = client.get_world()
        if DESIRED_MAP not in world.get_map().name:
            print(f"[INFO] Cargando {DESIRED_MAP} para pruebas de Loop...")
            world = client.load_world(DESIRED_MAP)
        else:
            print(f"[INFO] Ya estás en {DESIRED_MAP}")
    except Exception as e:
        print(f"[ERROR] No se pudo conectar/cargar mapa: {e}")
        return

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
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute('role_name', 'ego')

        # --- SELECCIÓN DE PUNTO DE SPAWN EN LA AUTOPISTA ---
        spawn_points = world.get_map().get_spawn_points()
        # En Town04, el índice 40 o 45 suele estar en la highway exterior
        spawn_idx = 40 if len(spawn_points) > 40 else 0
        spawn_point = spawn_points[spawn_idx]
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # Configuración de Cámara (Igual a entrenamiento)
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")

        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)

        print("[INFO] Inferencia iniciada en Town04 Highway Loop.")
        control = carla.VehicleControl()

        while True:
            world.tick()
            
            try:
                img = image_queue.get(timeout=2.0)
            except Empty:
                continue

            frame = to_bgr(img)
            frame_masked = apply_half_mask(frame)
            
            # Inferencia PyTorch
            frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                steer_pred = model(input_tensor).item()

            # Control Longitudinal (PID Simple)
            current_speed = get_speed(vehicle)
            speed_error = TARGET_SPEED_KMH - current_speed
            
            throttle = 0.0
            brake = 0.0
            
            if speed_error > 0:
                throttle = min(0.8, float(speed_error * KP_THROTTLE))
            else:
                brake = min(0.5, float(-speed_error * KP_BRAKE))

            # Aplicar Controles
            control.steer = float(np.clip(steer_pred, -1.0, 1.0))
            control.throttle = throttle
            control.brake = brake
            vehicle.apply_control(control)

            # Visualización mejorada
            status = f"Speed: {current_speed:.1f}/{TARGET_SPEED_KMH} | Steer Pred: {steer_pred:.3f}"
            cv2.putText(frame_masked, status, (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Inferencia PilotNet - Town04 Loop", frame_masked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[INFO] Cerrando sesión de inferencia...")
        world.apply_settings(original_settings)
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()