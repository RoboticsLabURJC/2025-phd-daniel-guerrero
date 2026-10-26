import os
import time
from queue import Queue, Empty
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import carla

# ---------- Parámetros de Inferencia ----------
HOST = "127.0.0.1"
PORT = 2000
MODEL_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/inferencias/pilotnet_260322.pth"

TARGET_SPEED_KMH = 5.0  
KP_THROTTLE = 0.25       
KP_BRAKE = 0.5           

IMG_W, IMG_H = 640, 360
CAM_FPS = 20
FIXED_DT = 1.0 / CAM_FPS

# --- Configuración de Suavizado y Agresividad ---
STEER_SMOOTHING_WINDOW = 5 # Filtro de media móvil
STEER_BOOST = 0.8          # Menor a 1.0 aumenta la fuerza en curvas (potencia)

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

def calculate_complex(speed_ms):
    """
    Física para Tesla Model 3.
    Ajustada para mayor autoridad de giro (Gr bajo y base_ratio alto).
    """
    m = 1611.0
    a = 1.44
    b = 1.44
    L = a + b
    k1 = -160000.0
    k2 = -175000.0
    
    K_stab = (m / (L**2)) * ((a / k2) - (b / k1))
    
    # Bajamos Gr para que el coche "quiera" girar más por cada grado de volante
    Gr = 0.095 

    if speed_ms < 0.5: return 0.7
        
    numerator = speed_ms / L
    denominator = 1 + K_stab * (speed_ms**2)
    ratio_i = (1.0 / Gr) * (numerator / denominator)
    
    # Subimos base_ratio para dar más rango de acción a la PilotNet
    base_ratio = 11.5 
    
    if ratio_i < 1.0: ratio_i = 1.0
    
    # Amortiguación final
    sensitivity = (base_ratio / ratio_i) * 0.50 
    return max(0.05, min(0.8, sensitivity))

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
        print(f"[ERROR] No modelo: {MODEL_PATH}"); return

    model = PilotNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.get_world()
    if "Town04" not in world.get_map().name:
        world = client.load_world("Town04")

    original_settings = world.get_settings()
    steer_buffer = deque(maxlen=STEER_SMOOTHING_WINDOW)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        vehicle = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), 
                                    world.get_map().get_spawn_points()[40])

        image_queue = Queue()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=0.8, z=1.3)), attach_to=vehicle)
        camera.listen(image_queue.put)

        control = carla.VehicleControl()

        print("[INFO] Tesla Model 3 listo en Town04 Highway Loop.")

        while True:
            world.tick()
            try:
                img = image_queue.get(timeout=2.0)
            except Empty: continue

            frame = to_bgr(img)
            frame_masked = apply_half_mask(frame)
            
            # Inferencia
            frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                steer_pred = model(input_tensor).item()

            # --- LÓGICA DE CONTROL AVANZADA ---
            current_speed_kmh = get_speed(vehicle)
            sens = calculate_complex(current_speed_kmh / 3.6)
            
            # 1. Boost no lineal: Exagera la intención de la red en curvas
            # Usamos signo * abs para no perder la dirección del giro
            boosted_steer = np.sign(steer_pred) * (np.abs(steer_pred) ** STEER_BOOST)
            
            # 2. Aplicar sensibilidad física
            raw_steer = boosted_steer * sens
            
            # 3. Suavizado (Filtro de media móvil)
            steer_buffer.append(raw_steer)
            final_steer = sum(steer_buffer) / len(steer_buffer)
            # ----------------------------------

            # Control Longitudinal
            speed_error = TARGET_SPEED_KMH - current_speed_kmh
            control.throttle = min(0.8, float(speed_error * KP_THROTTLE)) if speed_error > 0 else 0.0
            control.brake = min(0.5, float(-speed_error * KP_BRAKE)) if speed_error <= 0 else 0.0

            control.steer = float(np.clip(final_steer, -1.0, 1.0))
            vehicle.apply_control(control)

            # Debug UI
            status = f"Vel: {current_speed_kmh:.1f} | Sens: {sens:.2f} | Final Steer: {final_steer:.3f}"
            cv2.putText(frame_masked, status, (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Inferencia PhD - Tesla Town04", frame_masked)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        world.apply_settings(original_settings)
        if 'camera' in locals(): camera.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()