import carla
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO (RESNET-18 - IMAGE ONLY)
# ==========================================
class ResNetSteeringOnly(nn.Module):
    def __init__(self):
        super(ResNetSteeringOnly, self).__init__()
        self.cnn = models.resnet18(weights=None)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity() 
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, image):
        x = self.cnn(image) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output_layer(x)

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. CONFIGURACIÓN BEV BIFOCAL Y SIMULADOR
# ==========================================
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
BEV_WIDTH, BEV_HEIGHT = 600, 800

def build_bifocal_bev_maps(img_w=IMG_WIDTH, img_h=IMG_HEIGHT, bev_w=BEV_WIDTH, bev_h=BEV_HEIGHT):
    split_y = 0.45 * bev_h
    p_left_seam = [0.39 * img_w, 0.385 * img_h]
    p_right_seam = [0.61 * img_w, 0.385 * img_h]
    p_left_bot = [-0.25 * img_w, 0.98 * img_h]
    p_right_bot = [1.25 * img_w, 0.98 * img_h]

    src_bot = np.float32([p_left_bot, p_right_bot, p_right_seam, p_left_seam])
    dst_bot = np.float32([[0, bev_h], [bev_w, bev_h], [bev_w, split_y], [0, split_y]])
    H_bot_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_bot, dst_bot))

    src_top = np.float32([p_left_seam, p_right_seam, [0.76 * img_w, 0.24 * img_h], [0.24 * img_w, 0.24 * img_h]])
    dst_top = np.float32([[0, split_y], [bev_w, split_y], [bev_w, 0], [0, 0]])
    H_top_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_top, dst_top))

    y_grid, x_grid = np.indices((bev_h, bev_w), dtype=np.float32)
    coords = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1).reshape(-1, 3)

    c_bot = coords @ H_bot_inv.T
    u_bot = (c_bot[:, 0] / c_bot[:, 2]).reshape(bev_h, bev_w)
    v_bot = (c_bot[:, 1] / c_bot[:, 2]).reshape(bev_h, bev_w)

    c_top = coords @ H_top_inv.T
    u_top = (c_top[:, 0] / c_top[:, 2]).reshape(bev_h, bev_w)
    v_top = (c_top[:, 1] / c_top[:, 2]).reshape(bev_h, bev_w)

    alpha_blend = 1.0 / (1.0 + np.exp(-(y_grid - split_y) / 22.0))
    map_x = (alpha_blend * u_bot + (1.0 - alpha_blend) * u_top).astype(np.float32)
    map_y = (alpha_blend * v_bot + (1.0 - alpha_blend) * v_top).astype(np.float32)
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)

BEV_MAP1, BEV_MAP2 = build_bifocal_bev_maps()

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
    # Dispositivo fijado en CPU para procesadores/gráficas AMD sin ROCm
    device = torch.device("cpu")
    print(f"[INFO] Dispositivo de inferencia: {device}")

    # 1. Cargar el modelo entrenado
    # IMPORTANTE: Asegúrate de que apunte a tu archivo ResNet-18
    model_path = "/home/daniel/code/2025-phd-daniel-guerrero/models/bev/modelo_resnet18_steering_only.pth" 
    
    model = ResNetSteeringOnly()
    # Desactivamos el bloqueo estricto de PyTorch 2.6+
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    print("[INFO] Modelo ResNet-18 (Image Only) cargado exitosamente.")

    # 2. Conectar a CARLA y Cargar Town04
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    print("[INFO] Cargando mapa Town04 (Circuito)...")
    world = client.load_world('Town04')
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
                
                bev_img = cv2.remap(img_copy, BEV_MAP1, BEV_MAP2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
                
                img_tensor = image_transform(bev_rgb).unsqueeze(0).to(device) 

                # Inferencia Visual Pura
                with torch.no_grad():
                    steering_pred = model(img_tensor).item()

                steering_pred = max(-1.0, min(1.0, steering_pred))

                # Control Físico
                control = carla.VehicleControl(
                    throttle=CONSTANT_THROTTLE,
                    steer=steering_pred,
                    brake=0.0
                )
                vehicle.apply_control(control)

                cv2.imshow("Vision del Coche (BEV) - ResNet-18", bev_img)
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