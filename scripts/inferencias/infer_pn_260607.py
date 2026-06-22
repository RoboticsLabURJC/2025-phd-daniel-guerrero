import carla
import cv2
import numpy as np
import torch
import torch.nn as nn
import queue
import random

# --- CONFIGURACIÓN ---
MODEL_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/model_training/pilotnet_model.pth"
TOWN_NAME = "Town04_Opt"  # El circuito grande con autopista periférica
THROTTLE_CONSTANTE = 0.40 # Aumentamos un poco la velocidad para la autopista

# Mismo dispositivo usado en entrenamiento
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. ARQUITECTURA PILOTNET (Idéntica al entrenamiento) ---
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
            nn.Linear(64 * 1 * 18, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# --- 2. PIPELINE DE PROCESAMIENTO DE IMAGEN ---
image_queue = queue.Queue()

def process_image(image):
    raw_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    raw_data = np.reshape(raw_data, (image.height, image.width, 4))
    image_rgb = raw_data[:, :, :3] 
    image_queue.put(image_rgb)

def preprocess_for_model(image_rgb):
    # Recorte y redimensionamiento exacto al entrenamiento
    img_cropped = image_rgb[250:500, :, :] 
    img_resized = cv2.resize(img_cropped, (200, 66), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized / 255.0 
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(DEVICE)

# --- 3. BUCLE DE INFERENCIA EN CARLA ---
def main():
    print(f"[INFO] Cargando modelo PilotNet desde {MODEL_PATH}...")
    model = PilotNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo. Verifica la ruta: {e}")
        return
        
    model.eval()

    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    vehicle = None
    camera = None

    try:
        print(f"[INFO] Cargando mapa {TOWN_NAME} (Circuito Periférico)...")
        world = client.load_world(TOWN_NAME)
        
        # Modo síncrono a 20 FPS
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20.0
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        
        # 1. Spawn del Vehículo EGO
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        
        # Town04 tiene los puntos de la autopista generalmente mezclados, elegimos uno aleatorio
        # Si apareces en el pueblo y quieres la autopista, simplemente reinicia el script
        spawn_point = random.choice(spawn_points) 
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("[INFO] Tesla Model 3 en línea y cediendo el control a la IA.")

        # 2. Spawn de la Cámara (POSICIÓN EXACTA)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05') 
        
        camera_transform = carla.Transform(carla.Location(x=0.4, y=-0.3, z=1.3))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(lambda image: process_image(image))

        print("[INFO] Inferencia iniciada. Presiona 'q' en la ventana de OpenCV para salir.")

        while True:
            world.tick() 
            
            if not image_queue.empty():
                image_rgb = image_queue.get()
                
                # --- INFERENCIA ---
                with torch.no_grad():
                    tensor_input = preprocess_for_model(image_rgb)
                    predicted_steering = model(tensor_input).item()
                
                # --- APLICAR CONTROL ---
                control = carla.VehicleControl()
                control.steer = predicted_steering
                control.throttle = THROTTLE_CONSTANTE
                control.brake = 0.0
                vehicle.apply_control(control)

                # --- VISUALIZACIÓN OPENCV ---
                display_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                bar_x_center = 400
                bar_y = 550
                bar_width = int(predicted_steering * 200) 
                
                cv2.rectangle(display_img, (200, bar_y-10), (600, bar_y+10), (50, 50, 50), -1) 
                cv2.line(display_img, (bar_x_center, bar_y-15), (bar_x_center, bar_y+15), (255, 255, 255), 2) 
                
                if predicted_steering > 0: 
                    cv2.rectangle(display_img, (bar_x_center, bar_y-10), (bar_x_center + bar_width, bar_y+10), (255, 0, 0), -1)
                else: 
                    cv2.rectangle(display_img, (bar_x_center + bar_width, bar_y-10), (bar_x_center, bar_y+10), (0, 0, 255), -1)

                cv2.putText(display_img, f"Steering: {predicted_steering:.3f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Inferencia PilotNet", display_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"\n[ERROR] Ocurrió un fallo: {e}")
    finally:
        print("\n[INFO] Limpiando simulador y cerrando...")
        cv2.destroyAllWindows()
        
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()

if __name__ == '__main__':
    main()