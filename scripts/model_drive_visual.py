# model_drive.py
import argparse
import carla
import torch
import torch.serialization
import cv2
import numpy as np
import time
import threading
from collections import deque

from model import CarlaPilotNet  # Asegúrate de que model.py está en el mismo directorio

# Permitir que PyTorch cargue la clase desde el .pth
torch.serialization.add_safe_globals([CarlaPilotNet])

# ---------------- Utilidades HUD/controles ----------------
def clamp(x, a, b):
    return max(a, min(b, x))

def ema(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def draw_bar(img, x, y, w, h, value, label, color=(80, 220, 80)):
    v = float(clamp(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
    fill_h = int(h * v)
    y0 = y + h - fill_h; y1 = y + h
    cv2.rectangle(img, (x + 2, y0 + 2), (x + w - 2, y1 - 2), color, -1)
    cv2.putText(img, f"{label}: {v:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

def draw_center_steer_bar(img, cx, cy, w, h, steer_val, color=(80, 220, 80), label="steer"):
    s = float(clamp(steer_val, -1.0, 1.0))
    x0 = int(cx - w // 2); x1 = int(cx + w // 2)
    y0 = int(cy - h // 2); y1 = int(cy + h // 2)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 2)
    cv2.line(img, (cx, y0), (cx, y1), (180, 180, 180), 1)
    dx = int((w // 2) * s)
    cv2.rectangle(img, (cx + dx - 6, y0 + 2), (cx + dx + 6, y1 - 2), color, -1)
    cv2.putText(img, f"{label}: {s:+.3f}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

def draw_steer_arrow(img, center, length, steer_val, color=(80, 220, 80), thickness=6):
    s = float(clamp(steer_val, -1.0, 1.0))
    max_deg = 30.0
    angle_deg = s * max_deg
    angle_rad = np.deg2rad(angle_deg - 90)
    x0, y0 = center
    x1 = int(x0 + length * np.cos(angle_rad)); y1 = int(y0 + length * np.sin(angle_rad))
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.25)
    cv2.putText(img, f"{angle_deg:+.1f}°", (x0 - 30, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

# ---------------- Args ----------------
parser = argparse.ArgumentParser(description="Model Drive con vista de cámara y anti-stall")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=2000)
# Resolución ALTA para ver; el modelo siempre verá 200x66 internamente
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fov", type=float, default=90.0)
parser.add_argument("--fps_sensor", type=int, default=20)
parser.add_argument("--autopilot", action="store_true", help="Iniciar con autopilot ON")
parser.add_argument("--kickstart", type=float, default=0.15, help="Throttle mínimo si está parado (0 desactiva)")
parser.add_argument("--kick_speed", type=float, default=1.0, help="Umbral km/h para considerar 'parado'")
parser.add_argument("--steer_ema", type=float, default=0.2, help="Suavizado exponencial del steer [0..1]")
args = parser.parse_args()

# ---------------- Modelo ----------------
model = torch.load("carla_model.pth", map_location="cpu", weights_only=False)
model.eval()

# ---------------- CARLA ----------------
client = carla.Client(args.host, args.port)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

veh_bps = bp_lib.filter("vehicle.tesla.model3")
vehicle_bp = veh_bps[0] if len(veh_bps) else bp_lib.filter("vehicle.*")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    for sp in world.get_map().get_spawn_points():
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle: break
if vehicle is None:
    raise RuntimeError("No se pudo spawnear el vehículo.")

vehicle.set_autopilot(args.autopilot)

# Cámara RGB de ALTA resolución para visualización
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(args.width))
cam_bp.set_attribute("image_size_y", str(args.height))
cam_bp.set_attribute("fov", str(args.fov))
cam_bp.set_attribute("sensor_tick", str(1.0 / max(1, args.fps_sensor)))
cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

# ---------------- Estados compartidos ----------------
last_bgr = None
last_pred = (0.0, 0.0, 0.0)
lock = threading.Lock()
hud_h = 140
scale_display = 1  # ya estamos en 960x540; 1x es suficiente
steer_state = 0.0  # para EMA
use_kickstart = args.kickstart > 0.0

# ---------------- Preprocesado ----------------
def preprocess_for_model(bgr_frame):
    # Redimensiona a 200x66 para el modelo y convierte a RGB
    resized = cv2.resize(bgr_frame, (200, 66), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))  # (3,66,200)
    return torch.from_numpy(chw).unsqueeze(0)  # (1,3,66,200)

# ---------------- Control ----------------
def compute_control_from_preds(preds, spd_kmh):
    global steer_state
    s, t, b = preds[0].tolist()

    # Clamps
    s = float(clamp(s, -1.0, 1.0))
    t = float(clamp(t, 0.0, 1.0))
    b = float(clamp(b, 0.0, 1.0))

    # Suavizado de steer
    steer_state = ema(steer_state, s, args.steer_ema)
    s = steer_state

    # Evitar freno y acel a la vez
    if t > 0.05:
        b = 0.0

    # Anti-stall / Kickstart
    if use_kickstart and spd_kmh < args.kick_speed:
        t = max(t, args.kickstart)
        b = 0.0

    return s, t, b

def apply_control(s, t, b):
    vehicle.apply_control(carla.VehicleControl(steer=float(s), throttle=float(t), brake=float(b)))

# ---------------- Callback cámara ----------------
def camera_callback(image):
    global last_bgr, last_pred
    # BGRA -> BGR
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]

    if not args.autopilot:
        # Inferencia del modelo con frame redimensionado a 200x66
        inp = preprocess_for_model(bgr)
        with torch.no_grad():
            preds = model(inp)
        spd = speed_kmh(vehicle)
        s, t, b = compute_control_from_preds(preds, spd)
        apply_control(s, t, b)
        with lock:
            last_pred = (s, t, b)

    with lock:
        last_bgr = bgr

camera.listen(camera_callback)

# ---------------- Bucle de visualización ----------------
print("🚗 Conducción en curso. Ventana: 'Model Drive'. (Q/ESC salir, P pausa, A autopilot, K kickstart)")
paused = False
writer = None

try:
    while True:
        with lock:
            frame = None if last_bgr is None else last_bgr.copy()
            s, t, b = last_pred

        if frame is not None:
            disp = cv2.resize(frame, None, fx=scale_display, fy=scale_display, interpolation=cv2.INTER_AREA)
            h, w = disp.shape[:2]
            canvas = np.zeros((h + hud_h, w, 3), dtype=np.uint8)
            canvas[:h, :w] = disp

            # HUD
            pad = 20
            bar_w = 28
            bar_h = 100
            draw_bar(canvas, pad, h + pad, bar_w, bar_h, t, "throttle", (80, 220, 80))
            draw_bar(canvas, pad + bar_w + 12, h + pad, bar_w, bar_h, b, "brake", (60, 60, 230))
            cx = w // 2
            draw_center_steer_bar(canvas, cx, h + 30, 360, 22, s, (80, 220, 80), "steer")
            draw_steer_arrow(canvas, (cx, h + 90), 70, s, (80, 220, 80), thickness=6)

            spd = speed_kmh(vehicle)
            status = f"Speed: {spd:5.1f} km/h   Autopilot: {'ON' if args.autopilot else 'OFF'}   Kickstart: {'ON' if use_kickstart else 'OFF'}"
            cv2.putText(canvas, status, (pad, h + hud_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

            cv2.imshow("Model Drive", canvas)

        key = cv2.waitKey(1 if not paused else 100) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('a'):
            args.autopilot = not args.autopilot
            vehicle.set_autopilot(args.autopilot)
        elif key == ord('k'):
            use_kickstart = not use_kickstart

        time.sleep(0.002)

finally:
    print("🛑 Deteniendo.")
    try:
        if writer: writer.release()
    except: pass
    try:
        camera.stop()
    except: pass
    try:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    except: pass
    try:
        vehicle.destroy()
    except: pass
    cv2.destroyAllWindows()
