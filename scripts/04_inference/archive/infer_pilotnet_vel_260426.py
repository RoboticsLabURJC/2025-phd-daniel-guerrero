import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from queue import Queue, Empty
import carla

# ---------- CONFIGURACIÓN ----------
MODEL_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/model_training/pilotnet_20260423_corregido.pth" 
IMG_W, IMG_H = 640, 360
CAM_FPS = 15 
FIXED_DT = 1.0 / CAM_FPS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalPilotNet(nn.Module):
    def __init__(self):
        super(ConditionalPilotNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ELU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = self.features(torch.zeros(1, 3, 160, 320))
            self.flatten_size = dummy.shape[1]
            
        self.decision = nn.Sequential(
            nn.Linear(self.flatten_size + 1, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU()
        )
        self.out_steer = nn.Linear(10, 1)
        self.out_throttle = nn.Linear(10, 1)

    def forward(self, img, speed):
        x = self.features(img)
        x = torch.cat((x, speed), dim=1)
        x = self.decision(x)
        return self.out_steer(x), self.out_throttle(x)

def preprocess(image_data):
    arr = np.frombuffer(image_data.raw_data, dtype=np.uint8).reshape((image_data.height, image_data.width, 4))
    frame = arr[:, :, :3].copy() 
    h = frame.shape[0]
    frame[0:h//2, :] = 0 # Máscara idéntica al entrenamiento
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (320, 160)) 
    img_tensor = torch.from_numpy(frame_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img_tensor, frame

def main():
    model = ConditionalPilotNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    
    # FORZAR CARGA DE TOWN04 (El loop infinito)
    world = client.get_world()
    if "Town04" not in world.get_map().name:
        print("[INFO] Cargando Town04...")
        world = client.load_world("Town04")
    
    steer_smoothing = deque(maxlen=3)
    win_name = "Inferencia PhD - Loop Town04"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        
        # PUNTO DE SPAW EN EL LOOP (Autopista)
        # El 38 o 45 suelen dejarte justo al inicio de la curva del loop
        spawn_point = world.get_map().get_spawn_points()[38] 
        vehicle = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), spawn_point)

        image_queue = Queue()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        # Ajustamos cámara un poco más adelante para evitar ver el capó
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.6, z=1.6)), attach_to=vehicle)
        camera.listen(image_queue.put)

        print("[INFO] Tesla spawneado en el Loop de Town04.")

        while True:
            world.tick()
            try:
                data = image_queue.get(timeout=1.0)
                img_tensor, display_frame = preprocess(data)
                
                v = vehicle.get_velocity()
                speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                speed_tensor = torch.tensor([[speed_kmh / 50.0]], dtype=torch.float32).to(DEVICE)

                with torch.no_grad():
                    p_steer, p_thr = model(img_tensor, speed_tensor)

                # Suavizado y aplicación
                steer_smoothing.append(p_steer.item())
                final_steer = sum(steer_smoothing) / len(steer_smoothing)
                
                # Control de velocidad crucero (Máx 20 km/h para pruebas de carril)
                thr_val = p_thr.item()
                if speed_kmh > 20.0:
                    thr_val = 0.0
                
                control = carla.VehicleControl(steer=float(final_steer), throttle=float(thr_val))
                vehicle.apply_control(control)

                cv2.imshow(win_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            except Empty: continue

    finally:
        world.apply_settings(original_settings)
        if 'camera' in locals(): camera.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()