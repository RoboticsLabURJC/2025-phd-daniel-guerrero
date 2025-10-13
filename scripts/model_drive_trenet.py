#!/usr/bin/env python3
import argparse
import time
import threading
from collections import deque

import carla
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.serialization

# ====== MODELO TRENNet (igual que en train_trenet_canny.py) ======
class CarlaTreNet(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_size: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 66, 200)
            cnn_out = self.cnn(dummy).shape[1]
        self.lstm = nn.LSTM(input_size=cnn_out, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 3)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)            # (B*T, F)
        feats = feats.view(B, T, -1)   # (B, T, F)
        _, (h_n, _) = self.lstm(feats)
        h_last = h_n[-1]
        out = self.fc(h_last)
        return out

# Registrar la clase para poder cargar pickles guardados con torch.save(model)
torch.serialization.add_safe_globals([CarlaTreNet])

# ====== Utils HUD/controles ======
def clamp(x, a, b): return max(a, min(b, x))
def ema(prev, new, alpha): return alpha * new + (1 - alpha) * prev

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

# ====== Args ======
parser = argparse.ArgumentParser(description="Model Drive (TreNet: secuencias + Canny + PiP + limitador)")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=2000)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fov", type=float, default=90.0)
parser.add_argument("--fps_sensor", type=int, default=20)
parser.add_argument("--autopilot", action="store_true")
parser.add_argument("--steer_ema", type=float, default=0.2)
parser.add_argument("--kickstart", type=float, default=0.10)
parser.add_argument("--kick_speed", type=float, default=1.0)

# Modelo / secuencia
parser.add_argument("--model_path", type=str, default="carla_trenet_canny.pth")
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--hidden_size", type=int, default=128)  # solo si cargas state_dict

# Canny
parser.add_argument("--use_canny", action="store_true")
parser.add_argument("--canny_low", type=int, default=100)
parser.add_argument("--canny_high", type=int, default=200)
parser.add_argument("--canny_blur", type=int, default=5)
parser.add_argument("--canny_sigma", type=float, default=1.0)
parser.add_argument("--show_canny", action="store_true")

# Velocidad
parser.add_argument("--target_speed_kmh", type=float, default=12.0)
parser.add_argument("--throttle_cap", type=float, default=0.25)

args = parser.parse_args()

# ====== Carga del modelo (soporta pickle completo o state_dict) ======
def load_trenet(model_path):
    try:
        m = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(m, dict) and "state_dict" in m:
            model = CarlaTreNet(in_channels=1, hidden_size=m.get("hidden_size", args.hidden_size))
            model.load_state_dict(m["state_dict"])
        elif isinstance(m, nn.Module):
            model = m
        else:
            # asumir state_dict plano
            model = CarlaTreNet(in_channels=1, hidden_size=args.hidden_size)
            model.load_state_dict(m)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"No pude cargar el modelo desde {model_path}: {e}")

model = load_trenet(args.model_path)

# ====== CARLA ======
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

cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(args.width))
cam_bp.set_attribute("image_size_y", str(args.height))
cam_bp.set_attribute("fov", str(args.fov))
cam_bp.set_attribute("sensor_tick", str(1.0 / max(1, args.fps_sensor)))
cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

# ====== Estados ======
lock = threading.Lock()
last_bgr = None
last_pred = (0.0, 0.0, 0.0)
last_canny_preview = None
hud_h = 160
steer_state = 0.0
use_kickstart = args.kickstart > 0.0

# Buffer temporal (T, 1, 66, 200) de últimos frames
seq_buf = deque(maxlen=args.seq_len)

# ====== Preprocesado ======
def preprocess_frame_to_canny1(bgr_frame):
    """Devuelve: (1, 66, 200) float32 [0,1] y preview uint8 HxW."""
    resized = cv2.resize(bgr_frame, (200, 66), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if args.use_canny:
        if args.canny_blur and args.canny_blur > 0:
            k = args.canny_blur if args.canny_blur % 2 == 1 else args.canny_blur + 1
            gray = cv2.GaussianBlur(gray, (k, k), args.canny_sigma)
        edges_u8 = cv2.Canny(gray, args.canny_low, args.canny_high)
        edges = edges_u8.astype(np.float32) / 255.0
        return edges[None, :, :], edges_u8
    else:
        # Sin Canny: luma normalizada
        luma = gray.astype(np.float32) / 255.0
        return luma[None, :, :], None

# ====== Control ======
def compute_control_from_preds(preds, spd_kmh):
    global steer_state
    s, t, b = preds[0].tolist()
    s = float(clamp(s, -1.0, 1.0))
    t = float(clamp(t, 0.0, 1.0))
    b = float(clamp(b, 0.0, 1.0))

    # Suavizado de dirección
    steer_state = ema(steer_state, s, args.steer_ema)
    s = steer_state

    # Throttle cap para ir suave
    t = min(t, max(0.0, min(1.0, args.throttle_cap)))

    # Limitador simple por velocidad
    if args.target_speed_kmh > 0.0:
        target = args.target_speed_kmh
        margin = 1.0
        if spd_kmh > target + 2.0:
            t = 0.0
            b = max(b, min(0.4, 0.1 + 0.02 * (spd_kmh - target)))
        elif spd_kmh > target + margin:
            t = 0.0
        elif spd_kmh > target:
            t = min(t, 0.05)

    if t > 0.05:
        b = 0.0

    if use_kickstart and spd_kmh < args.kick_speed:
        t = max(t, args.kickstart)
        b = 0.0

    return s, t, b

def apply_control(s, t, b):
    vehicle.apply_control(carla.VehicleControl(steer=float(s), throttle=float(t), brake=float(b)))

# ====== Callback cámara ======
def camera_callback(image):
    global last_bgr, last_pred, last_canny_preview
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]

    # Preprocesar y actualizar buffer
    x1, canny_prev = preprocess_frame_to_canny1(bgr)  # (1,66,200)
    with lock:
        seq_buf.append(x1)  # cada x1 es (1,66,200)
        last_canny_preview = canny_prev
        last_bgr = bgr

    if args.autopilot:
        return

    # Solo inferimos cuando el buffer está lleno
    if len(seq_buf) == args.seq_len:
        with lock:
            seq_np = np.stack(list(seq_buf), axis=0).astype(np.float32)  # (T,1,66,200)
        inp = torch.from_numpy(seq_np).unsqueeze(0)  # (1,T,1,66,200)

        with torch.no_grad():
            preds = model(inp)
        spd = speed_kmh(vehicle)
        s, t, b = compute_control_from_preds(preds, spd)
        apply_control(s, t, b)
        with lock:
            last_pred = (s, t, b)

camera.listen(camera_callback)

# ====== Bucle de visualización ======
print("🚗 TreNet drive. Ventana: 'Model Drive (TreNet)'. (Q/ESC salir, P pausa, A autopilot, K kickstart)")
paused = False

try:
    while True:
        with lock:
            frame = None if last_bgr is None else last_bgr.copy()
            s, t, b = last_pred
            canny_prev = last_canny_preview
            filled = len(seq_buf)

        if frame is not None:
            disp = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
            h, w = disp.shape[:2]
            hud_h = 170
            canvas = np.zeros((h + hud_h, w, 3), dtype=np.uint8)
            canvas[:h, :w] = disp

            pad = 20
            bar_w = 28
            bar_h = 110
            draw_bar(canvas, pad, h + pad, bar_w, bar_h, t, "throttle", (80, 220, 80))
            draw_bar(canvas, pad + bar_w + 12, h + pad, bar_w, bar_h, b, "brake", (60, 60, 230))
            cx = w // 2
            draw_center_steer_bar(canvas, cx, h + 30, 360, 22, s, (80, 220, 80), "steer")
            draw_steer_arrow(canvas, (cx, h + 100), 70, s, (80, 220, 80), thickness=6)

            spd = speed_kmh(vehicle)
            status = (
                f"Speed: {spd:5.1f} km/h   "
                f"Autopilot: {'ON' if args.autopilot else 'OFF'}   "
                f"Kickstart: {'ON' if use_kickstart else 'OFF'}   "
                f"SeqBuf: {filled}/{args.seq_len}   "
                f"Canny: {'ON' if args.use_canny else 'OFF'}"
            )
            cv2.putText(canvas, status, (pad, h + hud_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

            # PiP canny del último frame
            if args.show_canny and canny_prev is not None:
                pip = canny_prev
                pip_w = max(200, int(w * 0.25))
                pip_h = int(pip.shape[0] * (pip_w / pip.shape[1]))
                pip_bgr = cv2.cvtColor(cv2.resize(pip, (pip_w, pip_h)), cv2.COLOR_GRAY2BGR)
                x0, y0 = w - pip_w - 20, 20
                x1, y1 = x0 + pip_w, y0 + pip_h
                canvas[y0:y1, x0:x1] = pip_bgr
                cv2.rectangle(canvas, (x0 - 2, y0 - 22), (x1 + 2, y1 + 2), (180, 180, 180), 2)
                cv2.putText(canvas, "CANNY (ultimo frame)", (x0, y0 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

            cv2.imshow("Model Drive (TreNet)", canvas)

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
    try: camera.stop()
    except: pass
    try: vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    except: pass
    try: vehicle.destroy()
    except: pass
    cv2.destroyAllWindows()
