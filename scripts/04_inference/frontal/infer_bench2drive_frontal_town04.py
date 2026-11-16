#!/usr/bin/env python3
"""Script de inferencia en CARLA (Town04) calibrado con Bench2Drive (Lincoln MKZ 2020):

  - Soporta visualización remota en vivo desde el navegador del celular (Termux
  / Tailscale / Wi-Fi)
    mediante un servidor MJPEG ligero integrado en el puerto 8080:
      * Tailscale: http://100.110.46.77:8080
      * Red Local: http://192.168.1.200:8080
  - No requiere ventana X11 (funciona 100% por SSH desde Termux).
  - Muestra telemetría en vivo en la consola y en el stream de video (velocidad,
  volante,
    distancia recorrida y desviación al centro del carril en metros).
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import queue
import threading
import time
import carla
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

LATEST_JPEG = None
FRAME_LOCK = threading.Lock()


class StreamHandler(BaseHTTPRequestHandler):

  def log_message(self, format, *args):
    pass  # Silenciar logs HTTP para no ensuciar la consola de Termux

  def do_GET(self):
    if self.path in ("/", "/index.html"):
      html = (
          b"<!DOCTYPE html><html><head><meta charset='utf-8'>"
          b"<meta name='viewport' content='width=device-width,"
          b" initial-scale=1'><title>CARLA Live Inference - Town04</title>"
          b"<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}"
          b"img{width:100%;max-width:960px;height:auto;border-bottom:2px"
          b" solid #0f0}h3{margin:8px 0;font-size:16px;color:#0f0}</style>"
          b"</head><body><h3>CARLA Town04 - Lincoln MKZ 2020 (Live"
          b" Stream)</h3><img src='/stream.mjpg' /></body></html>"
      )
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(html)))
      self.end_headers()
      self.wfile.write(html)
    elif self.path.startswith("/stream.mjpg"):
      self.send_response(200)
      self.send_header("Age", "0")
      self.send_header("Cache-Control", "no-cache, private")
      self.send_header("Pragma", "no-cache")
      self.send_header(
          "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
      )
      self.end_headers()
      try:
        while True:
          with FRAME_LOCK:
            jpg = LATEST_JPEG
          if jpg is not None:
            self.wfile.write(b"--FRAME\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            self.wfile.write(b"\r\n")
          time.sleep(0.05)
      except Exception:
        pass
    else:
      self.send_error(404)


def start_web_streamer(port=8080):
  server = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)
  t = threading.Thread(target=server.serve_forever, daemon=True)
  t.start()
  return server


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
  global LATEST_JPEG
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

  # Iniciar servidor de streaming MJPEG para ver desde el navegador en el celular
  port = 8080
  start_web_streamer(port=port)
  print("=" * 68)
  print(" [STREAM EN VIVO DISPONIBLE PARA TU CELULAR / NAVEGADOR]")
  print(f"   -> Por Tailscale : http://100.110.46.77:{port}")
  print(f"   -> Por Wi-Fi LAN : http://192.168.1.200:{port}")
  print("=" * 68)

  client = carla.Client("localhost", 2000)
  client.set_timeout(20.0)

  world = client.load_world("Town04")
  world.set_weather(carla.WeatherParameters.ClearNoon)
  carla_map = world.get_map()
  original_settings = world.get_settings()
  print("[INFO] Mapa Town04 cargado.")

  settings = world.get_settings()
  settings.synchronous_mode = True
  settings.fixed_delta_seconds = 0.05  # 20 FPS síncronos
  world.apply_settings(settings)

  bp_lib = world.get_blueprint_library()
  lincoln_bps = bp_lib.filter("vehicle.lincoln.mkz_2020")
  if not lincoln_bps:
    lincoln_bps = bp_lib.filter("vehicle.lincoln.mkz*")
  vehicle_bp = (
      lincoln_bps[0] if lincoln_bps else bp_lib.find("vehicle.tesla.model3")
  )
  vehicle_bp.set_attribute("role_name", "ego_vehicle")

  spawn_points = carla_map.get_spawn_points()
  spawn_idx = 40 if len(spawn_points) > 40 else 0
  vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_idx])
  print(
      f"[INFO] Vehículo '{vehicle_bp.id}' instanciado en Town04 Highway Loop"
      f" (spawn_points[{spawn_idx}])."
  )

  # Sensor de colisión para reportar en consola si choca
  col_bp = bp_lib.find("sensor.other.collision")
  col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
  collision_count = 0

  def on_collision(_event):
    nonlocal collision_count
    collision_count += 1

  col_sensor.listen(on_collision)

  # Calibración EXACTA de CAM_FRONT en Bench2Drive
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

  TARGET_SPEED_KMH = 20.0
  KP_THROTTLE = 0.20
  KP_BRAKE = 0.50
  STEER_GAIN = 1.0
  EMA_ALPHA = 0.75
  prev_steer = 0.0

  total_dist = 0.0
  prev_loc = vehicle.get_location()
  step = 0
  has_display = bool(os.environ.get("DISPLAY"))

  print("[INFO] Inferencia iniciada. Presiona Ctrl+C en Termux para detener.")

  try:
    while True:
      world.tick()
      step += 1
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

      tensor, _ = preprocess_frame(frame_bgr, crop_top_ratio=0.35)
      tensor = tensor.to(device, non_blocking=True)

      with torch.no_grad():
        raw_steer = float(model(tensor).item())

      scaled_steer = STEER_GAIN * raw_steer
      smooth_steer = EMA_ALPHA * scaled_steer + (1.0 - EMA_ALPHA) * prev_steer
      smooth_steer = max(-1.0, min(1.0, smooth_steer))
      prev_steer = smooth_steer

      vel = vehicle.get_velocity()
      speed_kmh = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
      speed_error = TARGET_SPEED_KMH - speed_kmh
      throttle = (
          min(0.8, float(speed_error * KP_THROTTLE)) if speed_error > 0 else 0.0
      )
      brake = (
          min(0.5, float(-speed_error * KP_BRAKE)) if speed_error <= 0 else 0.0
      )

      control = carla.VehicleControl(
          throttle=throttle, steer=smooth_steer, brake=brake
      )
      vehicle.apply_control(control)

      # Métricas en vivo: distancia recorrida y desviación al centro del carril
      cur_loc = vehicle.get_location()
      total_dist += cur_loc.distance(prev_loc)
      prev_loc = cur_loc
      wp = carla_map.get_waypoint(
          cur_loc, project_to_road=True, lane_type=carla.LaneType.Driving
      )
      lane_offset = cur_loc.distance(wp.transform.location) if wp else 0.0

      # Construir HUD para el stream web (800x450 para fluidez en móvil)
      hud = cv2.resize(frame_bgr, (800, 450))
      cv2.rectangle(hud, (0, 0), (800, 52), (0, 0, 0), -1)
      color = (0, 255, 0) if collision_count == 0 else (0, 0, 255)
      cv2.putText(
          hud,
          f"Steer: {smooth_steer:+.3f} | Vel: {speed_kmh:.1f} km/h | Dist:"
          f" {total_dist:.1f}m | Offset: {lane_offset:.2f}m | Col:"
          f" {collision_count}",
          (12, 34),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.62,
          color,
          2,
      )
      # Barra visual de volante abajo al centro
      cv2.rectangle(hud, (250, 415), (550, 435), (50, 50, 50), -1)
      cv2.line(hud, (400, 412), (400, 438), (255, 255, 255), 2)
      bar_x = int(400 + np.clip(smooth_steer, -1.0, 1.0) * 140)
      cv2.circle(hud, (bar_x, 425), 8, color, -1)

      ok, enc = cv2.imencode(".jpg", hud, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
      if ok:
        with FRAME_LOCK:
          LATEST_JPEG = enc.tobytes()

      if step % 10 == 0:
        print(
            f"[Paso {step:04d}] Dist: {total_dist:6.1f}m | Vel:"
            f" {speed_kmh:4.1f} km/h | Steer: {smooth_steer:+.3f} | Offset"
            f" carril: {lane_offset:.2f}m | Colisiones: {collision_count}",
            flush=True,
        )

      if has_display:
        cv2.imshow("Bench2Drive Lincoln MKZ - Town04", hud)
        if cv2.waitKey(1) & 0xFF == ord("q"):
          break
  except KeyboardInterrupt:
    print("\n[INFO] Inferencia detenida por el usuario.")
  finally:
    world.apply_settings(original_settings)
    if has_display:
      cv2.destroyAllWindows()
    if col_sensor.is_alive:
      col_sensor.stop()
      col_sensor.destroy()
    if camera.is_alive:
      camera.stop()
      camera.destroy()
    if vehicle.is_alive:
      vehicle.destroy()


if __name__ == "__main__":
  main()
