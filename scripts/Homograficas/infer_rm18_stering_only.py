import carla
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import math

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO
# ==========================================
class DualInputModel(nn.Module):
    def __init__(self):
        super(DualInputModel, self).__init__()
        self.cnn = models.resnet18(weights=None)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity() 
        self.fusion_fc1 = nn.Linear(num_features + 1, 128)
        self.fusion_fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, image, throttle_val):
        visual_features = self.cnn(image) 
        combined = torch.cat((visual_features, throttle_val), dim=1) 
        x = F.relu(self.fusion_fc1(combined))
        x = F.relu(self.fusion_fc2(x))
        return self.output_layer(x)

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. CONFIGURACIÓN BEV Y SIMULADOR
# ==========================================
H_MATRIX = np.array([
    [-1.38322388e-01, -8.80401482e-01,  4.42338059e+02],
    [-4.51221359e-02, -2.59903503e+00,  1.07695709e+03],
    [-5.42655982e-05, -2.67570238e-03,  1.00000000e+00]
])
BEV_WIDTH, BEV_HEIGHT = 600, 800

current_image = None

def process_camera_data(image):
    global current_image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    current_image = array[:, :, :3]

# ==========================================
# 3. BUCLE PRINCIPAL DE INFERENCIA
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de inferencia: {device}")

    # 1. Cargar el modelo entrenado
    model_path = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/Homograficas/modelo_conduccion_autonoma.pth"
    model = DualInputModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("[INFO] Modelo cargado exitosamente.")

    # 2. Conectar a CARLA y Cargar Town04
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0) # Aumentado para dar tiempo a cargar el mapa
    
    print("[INFO] Cargando mapa Town04 (Circuito)...")
    world = client.load_world('Town04') # <--- LÍNEA NUEVA PARA CAMBIAR EL MAPA
    blueprint_library = world.get_blueprint_library()

    # 3. Spawn del Vehículo (Tesla Model 3)
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print("[INFO] Vehículo instanciado.")

    # 4. Configurar y anclar la cámara
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '110')
    
    camera_transform = carla.Transform(carla.Location(x=2.0, y=0.0, z=1.6), carla.Rotation(pitch=-15.0))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(lambda image: process_camera_data(image))

    CONSTANT_THROTTLE = 0.4 
    print("[INFO] Autopilot activado. Presiona Ctrl+C para detener.")

    try:
        while True:
            if current_image is not None:
                img_copy = current_image.copy()
                img_copy[:450, :] = 0 
                
                bev_img = cv2.warpPerspective(img_copy, H_MATRIX, (BEV_WIDTH, BEV_HEIGHT))
                bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
                
                img_tensor = image_transform(bev_rgb).unsqueeze(0).to(device) 
                throttle_tensor = torch.tensor([[CONSTANT_THROTTLE]], dtype=torch.float32).to(device)

                with torch.no_grad():
                    steering_pred = model(img_tensor, throttle_tensor).item()

                steering_pred = max(-1.0, min(1.0, steering_pred))

                control = carla.VehicleControl(
                    throttle=CONSTANT_THROTTLE,
                    steer=steering_pred,
                    brake=0.0
                )
                vehicle.apply_control(control)

                cv2.imshow("Vision del Coche (BEV)", bev_img)
                if cv2.waitKey(1) == ord('q'):
                    break

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[INFO] Deteniendo inferencia...")
    finally:
        if 'camera' in locals() and camera is not None:
            camera.stop()
            camera.destroy()
        if 'vehicle' in locals() and vehicle is not None:
            vehicle.destroy()
        cv2.destroyAllWindows()
        print("[INFO] Actores destruidos. Fin de simulación.")

if __name__ == '__main__':
    main()