import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from queue import Queue, Empty
import carla

# ---------- Parámetros de Inferencia ----------
HOST = "127.0.0.1"
PORT = 2000
# Asegúrate de que esta ruta apunte a tu mejor modelo guardado hoy
MODEL_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/model_training/pilotnet_20260423.pth" 

IMG_W, IMG_H = 640, 360
CAM_FPS = 10
FIXED_DT = 1.0 / CAM_FPS

# --- Configuración de Manejo ---
STEER_SMOOTHING_WINDOW = 3
SPEED_LIMIT_KMH = 12.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {device}")

# --- 1. ARQUITECTURA EXACTA DEL ENTRENAMIENTO ---
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
        # 1152 es el tamaño de salida de las convs con entrada 160x320
        self.decision = nn.Sequential(
            nn.Linear(27456 + 1, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU()
        )
        self.out_steer = nn.Linear(10, 1)
        self.out_throttle = nn.Linear(10, 1)

    def forward(self, img, speed):
        x = self.features(img)
        x = torch.cat((x, speed), dim=1) # Inyección de velocidad
        x = self.decision(x)
        return self.out_steer(x), self.out_throttle(x)

# --- 2. FUNCIONES DE APOYO ---
def calculate_physics_sens(speed_ms):
    """Tu fórmula de sensibilidad ajustada para el Tesla Model 3."""
    if speed_ms < 0.5: return 0.7
    L, Gr, base_ratio = 2.88, 0.095, 11.5
    K_stab = -0.0015 # Valor simplificado de estabilidad
    numerator = speed_ms / L
    denominator = 1 + K_stab * (speed_ms**2)
    ratio_i = (1.0 / Gr) * (numerator / denominator)
    sensitivity = (base_ratio / max(1.0, ratio_i)) * 0.50 
    return max(0.1, min(0.9, sensitivity))

def to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()

def apply_half_mask(frame):
    h = frame.shape[0]
    frame[0:h//2, :] = 0
    return frame

def get_speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

# --- 3. BUCLE PRINCIPAL ---
def main():
    # Cargar Modelo
    model = ConditionalPilotNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[INFO] Modelo PilotNet Condicional cargado con éxito.")

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Asegurar que estamos en Town04 para la autopista
    if "Town04" not in world.get_map().name:
        print("[INFO] Cargando Town04...")
        world = client.load_world("Town04")

    original_settings = world.get_settings()
    steer_buffer = deque(maxlen=STEER_SMOOTHING_WINDOW)

    try:
        # Modo Síncrono
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)

        # Spawn Tesla Ego
        bp_lib = world.get_blueprint_library()
        spawn_point = world.get_map().get_spawn_points()[40]
        vehicle = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), spawn_point)

        # Cámara RGB
        image_queue = Queue()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=0.8, z=1.3)), attach_to=vehicle)
        camera.listen(image_queue.put)

        print("[INFO] Iniciando Piloto Automático IA...")

        while True:
            world.tick()
            try:
                img_data = image_queue.get(timeout=2.0)
            except Empty: continue

            # Preprocesamiento idéntico al entrenamiento
            frame = to_bgr(img_data)
            frame_masked = apply_half_mask(frame)
            frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (320, 160))
            
            # Aseguramos que la imagen sea float32
            img_tensor = torch.from_numpy(frame_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

            current_speed = get_speed_kmh(vehicle)
            # FORZAMOS el tensor de velocidad a ser FLOAT32 explícitamente
            speed_tensor = torch.tensor([[current_speed / 50.0]], dtype=torch.float32).to(device)

            # --- INFERENCIA ---
            with torch.no_grad():
                pred_steer, pred_throttle = model(img_tensor.float(), speed_tensor.float())
            
            # 1. Volante Directo e Instantáneo (Sin Buffer)
            STEER_GAIN = 1.0
            final_steer = pred_steer.item() * STEER_GAIN

            # 2. Post-procesamiento de Acelerador y Frenos
            raw_throttle = pred_throttle.item()
            final_throttle = raw_throttle
            final_brake = 0.0

            if raw_throttle < 0.12:
                final_throttle = 0.0
                final_brake = 0.4  

            if current_speed > SPEED_LIMIT_KMH:
                final_throttle = 0.0 
                final_brake = 0.5

            # Aplicar Control
            control = carla.VehicleControl()
            control.steer = float(np.clip(final_steer, -1.0, 1.0))
            control.throttle = float(np.clip(final_throttle, 0.0, 0.7))
            control.brake = float(final_brake)
            
            vehicle.apply_control(control)

            # Visualización
            status = f"Vel: {current_speed:.1f} km/h | Steer: {final_steer:.2f} | Thr: {final_throttle:.2f}"
            cv2.putText(frame_masked, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Tesla AI - Inferencia PhD", frame_masked)
            print(f"[CAJA NEGRA] Vel: {current_speed:05.1f} km/h | Volante (Steer): {final_steer:>6.3f} | Acelerador (Thr): {final_throttle:>5.3f}")
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("[INFO] Limpiando simulación...")
        world.apply_settings(original_settings)
        if 'camera' in locals(): camera.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()