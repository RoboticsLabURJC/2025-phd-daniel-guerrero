import carla
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import queue

# ==========================================
# 1. ARQUITECTURA PILOTNET (Debe ser idéntica al entrenamiento)
# ==========================================
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

# ==========================================
# 2. PREPROCESAMIENTO IDÉNTICO AL ENTRENAMIENTO
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_for_inference(image_bgr):
    """Recorta el cielo, redimensiona y convierte a tensor."""
    # 1. Recorte espacial (Crop Top 300px) de imagen 800x600 original
    cropped = image_bgr[300:600, 0:800]
    
    # 2. Redimensión al formato PilotNet (200x66)
    resized = cv2.resize(cropped, (200, 66))
    
    # 3. Conversión BGR (OpenCV) a RGB (PyTorch)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 4. Transformación a Tensor Normalizado
    tensor = transform(image_rgb).unsqueeze(0) # Añadir dimensión de batch
    return tensor, resized

# ==========================================
# 3. INFERENCIA EN CARLA
# ==========================================
def main():
    # Cargar modelo entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PilotNet().to(device)
    
    # NOTA: Asegúrate de tener tu archivo .pth en la misma ruta o actualiza el path
    weights_path = "pilotnet_balanced_weights.pth" 
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval() # Modo inferencia (apaga Dropouts/BatchNorms si los hubiera)
        print(f"[INFO] Modelo {weights_path} cargado correctamente en {device}.")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        return

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Cargar Town04 (Circuito periférico)
    world = client.load_world('Town04_Opt')
    print("[INFO] Entorno Town04 cargado.")
    
    # Configurar clima despejado
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    
    # Instanciar vehículo en un punto de spawn válido de la autopista
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0] # Modifica el índice si quieres empezar en otra curva
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("[INFO] Tesla Model 3 instanciado.")

    # Configurar Cámara RGB
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_bp.set_attribute('sensor_tick', '0.05') # 20 FPS para inferencia suave
    
    # Posición idéntica a la fase de recolección de datos
    camera_transform = carla.Transform(carla.Location(x=2.0, y=0.0, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Cola de imágenes asíncrona
    image_queue = queue.Queue()
    camera.listen(lambda image: image_queue.put(image))

    print("[INFO] Iniciando Piloto Automático... Presiona 'q' en la ventana para salir.")
    
    # Velocidad de crucero constante (Throttle)
    CRUISE_THROTTLE = 0.35 

    try:
        while True:
            if not image_queue.empty():
                image = image_queue.get()
                
                # Extraer datos de la cámara de CARLA
                raw_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                raw_data = np.reshape(raw_data, (image.height, image.width, 4))
                image_bgr_full = raw_data[:, :, :3]
                
                # Preprocesamiento y paso por la Red Neuronal
                tensor, image_cropped_hud = preprocess_for_inference(image_bgr_full)
                tensor = tensor.to(device)
                
                with torch.no_grad():
                    predicted_steer = model(tensor).item()
                
                # Aplicar control al vehículo
                control = carla.VehicleControl()
                control.throttle = CRUISE_THROTTLE
                control.steer = float(predicted_steer)
                # Limitar el giro a rangos físicos lógicos para evitar comportamientos erráticos
                control.steer = max(-1.0, min(1.0, control.steer)) 
                vehicle.apply_control(control)

                # ==========================================
                # DASHBOARD TELEMETRÍA (OpenCV)
                # ==========================================
                hud_display = image_bgr_full.copy()
                
                # Barra indicadora de volante
                cv2.rectangle(hud_display, (200, 500), (600, 530), (50, 50, 50), -1)
                cv2.line(hud_display, (400, 500), (400, 530), (255, 255, 255), 2)
                
                # Calcular posición de la barra indicadora
                steer_pos = int(400 + (predicted_steer * 200))
                cv2.circle(hud_display, (steer_pos, 515), 10, (0, 255, 0), -1)
                
                cv2.putText(hud_display, f"Steering: {predicted_steer:.3f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(hud_display, f"Throttle: {CRUISE_THROTTLE}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Mostrar la vista recortada (lo que ve la red) en la esquina superior derecha
                hud_display[20:86, 580:780] = image_cropped_hud

                cv2.imshow("PilotNet Inference - Town04", hud_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Apagando piloto automático...")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Ejecución interrumpida por el usuario.")
    finally:
        print("[INFO] Limpiando actores y cerrando...")
        cv2.destroyAllWindows()
        if 'camera' in locals() and camera.is_alive:
            camera.stop()
            camera.destroy()
        if 'vehicle' in locals() and vehicle.is_alive:
            vehicle.destroy()

if __name__ == '__main__':
    main()