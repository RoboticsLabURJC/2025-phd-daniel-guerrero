#!/usr/bin/env python3
"""Script de inferencia en CARLA (Town04) calibrado EXACTAMENTE con la cámara de Bench2Drive:

  - Resolución: 1600x900
  - FOV: 70.0°
  - Posición (cam2ego): x=0.8, y=0.0, z=1.6, pitch=0.0, yaw=0.0, roll=0.0
  - Preprocesamiento idéntico a train_bench2drive_frontal.py (crop_top=0.35 ->
  Resize(240, 320))
  - Suavizado EMA en dirección y control de velocidad para máxima estabilidad en
  el circuito Town04.
"""

import os
import queue
import carla
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ==========================================
# 1. ARQUITECTURA RESNET-18 (IDÉNTICA AL ENTRENAMIENTO)
# ==========================================
class ResNet18SteeringOnly(nn.Module):

  def __init__(self):
    super().__init__()
    self.cnn = models.resnet18(weights=None)
    num_features = self.cnn.fc.in_features
    self.cnn.fc = nn.Identity()

    self.fc1 = nn.Linear(num_features, 128)
    self.fc2 = nn.Linear(128, 64)
    self.output_layer = nn.Linear(64, 1)

  def forward(self, x):
    x = self.cnn(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.output_layer(x)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_frame(image_bgr, crop_top_ratio=0.35):
  h = image_bgr.shape[0]
  y0 = int(h * crop_top_ratio)
  cropped_bgr = image_bgr[y0:, :, :]
  image_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
  tensor = transform(image_rgb).unsqueeze(0)
  return tensor, cropped_bgr


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  lf_path = "/home/daniel/code/2025-phd-daniel-guerrero/models/frontal/modelo_resnet18_bench2drive_lanefollow.pth"
  old_path = "/home/daniel/code/2025-phd-daniel-guerrero/models/frontal/modelo_resnet18_bench2drive_frontal.pth"
  weights_path = lf_path if os.path.exists(lf_path) else old_path

  model = ResNet18SteeringOnly().to(device)
  model.load_state_dict(
      torch.load(weights_path, map_location=device, weights_only=True)
  )
  model.eval()
  print(f"[INFO] Modelo cargado en {device}: {weights_path}")

  client = carla.Client("localhost", 2000)
  client.set_timeout(20.0)

  world = client.load_world("Town04")
  world.set_weather(carla.WeatherParameters.ClearNoon)
  original_settings = world.get_settings()
  print("[INFO] Mapa Town04 cargado.")

  settings = world.get_settings()
  settings.synchronous_mode = True
  settings.fixed_delta_seconds = 0.05  # 20 FPS síncronos
  world.apply_settings(settings)

  bp_lib = world.get_blueprint_library()
  vehicle_bp = bp_lib.find("vehicle.tesla.model3")
  vehicle_bp.set_attribute("role_name", "ego_vehicle")

  spawn_points = world.get_map().get_spawn_points()
  spawn_idx = 40 if len(spawn_points) > 40 else 0
  vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_idx])
  print(f"[INFO] Vehículo instanciado en Town04 Highway Loop (spawn_points[{spawn_idx}]).")

  # Calibración EXACTA de CAM_FRONT en Bench2Drive (extraída de anno/*.json.gz)
  camera_bp = bp_lib.find("sensor.camera.rgb")
  camera_bp.set_attribute("image_size_x", "1600")
  camera_bp.set_attribute("image_size_y", "900")
  camera_bp.set_attribute("fov", "70.0")

  camera_transform = carla.Transform(
      carla.Location(x=0.8, y=0.0, z=1.6),
      carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
  )
  camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

  image_queue = queue.Queue()
  camera.listen(image_queue.put)

  TARGET_SPEED_KMH = 18.0
  KP_THROTTLE = 0.20
  KP_BRAKE = 0.50
  # El modelo lanefollow ya está entrenado con ángulos continuos reales (escala 1:1)
  STEER_GAIN = 1.0 if "lanefollow" in weights_path else 0.30
  STEER_DAMPING = 0.08
  EMA_ALPHA = 0.60
  MAX_STEER_HIGHWAY = 0.25
  prev_raw_steer = 0.0
  prev_steer = 0.0

  print(
      "[INFO] Inferencia síncrona iniciada en el Loop de Town04. Presiona 'q'"
      " en la ventana para salir."
  )

  try:
    while True:
      world.tick()
      try:
        while image_queue.qsize() > 1:
          image_queue.get_nowait()
        image = image_queue.get(timeout=2.0)
      except queue.Empty:
        continue

      raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
          (image.height, image.width, 4)
      )
      frame_bgr = raw[:, :, :3]

      tensor, cropped_bgr = preprocess_frame(frame_bgr, crop_top_ratio=0.35)
      tensor = tensor.to(device, non_blocking=True)

      with torch.no_grad():
        raw_steer = float(model(tensor).item())

      # Control PD + EMA sobre la predicción para evitar oscilación (zigzag)
      scaled_steer = (STEER_GAIN * raw_steer) - (
          STEER_DAMPING * (raw_steer - prev_raw_steer)
      )
      prev_raw_steer = raw_steer

      smooth_steer = EMA_ALPHA * scaled_steer + (1.0 - EMA_ALPHA) * prev_steer
      smooth_steer = max(
          -MAX_STEER_HIGHWAY, min(MAX_STEER_HIGHWAY, smooth_steer)
      )
      prev_steer = smooth_steer

      vel = vehicle.get_velocity()
      speed_kmh = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
      speed_error = TARGET_SPEED_KMH - speed_kmh
      throttle = min(0.8, float(speed_error * KP_THROTTLE)) if speed_error > 0 else 0.0
      brake = min(0.5, float(-speed_error * KP_BRAKE)) if speed_error <= 0 else 0.0

      control = carla.VehicleControl(
          throttle=throttle, steer=smooth_steer, brake=brake
      )
      vehicle.apply_control(control)

      hud = cv2.resize(frame_bgr, (960, 540))
      cv2.putText(
          hud,
          f"Steer: {smooth_steer:+.3f} (Raw: {raw_steer:+.2f}) | Speed: {speed_kmh:.1f} km/h",
          (20, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.8,
          (0, 255, 0),
          2,
      )
      cv2.imshow("Bench2Drive ResNet18 - Town04", hud)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
  except KeyboardInterrupt:
    pass
  finally:
    world.apply_settings(original_settings)
    cv2.destroyAllWindows()
    if camera.is_alive:
      camera.stop()
      camera.destroy()
    if vehicle.is_alive:
      vehicle.destroy()


if __name__ == "__main__":
  main()
