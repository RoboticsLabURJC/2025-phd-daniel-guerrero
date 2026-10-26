# model_drive_mobilenet.py
import argparse
import carla
import torch
import torch.nn as nn
import cv2
import numpy as np
import time
import threading
from PIL import Image
from torchvision import models, transforms

# ---------------- Utilidades HUD/controles ----------------
def clamp(x, a, b):
    return max(a, min(b, x))

def ema(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def draw_bar(img, x, y, w, h, value, label, color=(80, 220, 80)):
    v = float(clamp(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
    fill_h = int(h * v)
    y0 = y + h - fill_h
    y1 = y + h
    cv2.rectangle(img, (x + 2, y0 + 2), (x + w - 2, y1 - 2), color, -1)
    cv2.putText(img, f"{label}: {v:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1, cv2.LINE_AA)

def draw_center_steer_bar(img, cx, cy, w, h, steer_val, color=(80, 220, 80), label="steer"):
    s = float(clamp(steer_val, -1.0, 1.0))
    x0 = int(cx - w // 2); x1 = int(cx + w // 2)
    y0 = int(cy - h // 2); y1 = int(cy + h // 2)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 2)
    cv2.line(img, (cx, y0), (cx, y1), (180, 180, 180), 1)
    dx = int((w // 2) * s)
    cv2.rectangle(img, (cx + dx - 6, y0 + 2), (cx + dx + 6, y1 - 2), color, -1)
    cv2.putText(img, f"{label}: {s:+.3f}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1, cv2.LINE_AA)

def draw_steer_arrow(img, center, length, steer_val, color=(80, 220, 80), thickness=6):
    s = float(clamp(steer_val, -1.0, 1.0))
    max_deg = 30.0
    angle_deg = s * max_deg
    angle_rad = np.deg2rad(angle_deg - 90)
    x0, y0 = center
    x1 = int(x0 + length * np.cos(angle_rad))
    y1 = int(y0 + length * np.sin(angle_rad))
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.25)
    cv2.putText(img, f"{angle_deg:+.1f}°", (x0 - 30, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

# ---------------- Modelo (MobileNetV2 como en tu entrenamiento) ----------------
class CarlaMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base.last_channel, 3)  # steer, throttle, brake
        )

    def forward(self, x):
        return self.base(x)  # salidas crudas (B,3)

# ---------------- Args ----------------
parser = argparse.ArgumentParser(description="Model Drive (MobileNetV2) con HUD + anti-stall")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=2000)

# Resolución para visualización (la red SIEMPRE ve 224x224 internamente)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--fov", type=float, default=90.0)
parser.add_argument("--fps_sensor", type=int, default=20)

parser.add_argument("--autopilot", action="store_true", help="Iniciar con autopilot ON")
parser.add_argument("--kickstart", type=float, default=0.15, help="Throttle mínimo si está parado (0 desactiva)")
parser.add_argument("--kick_speed", type=float, default=1.0, help="Umbral km/h para considerar 'parado'")
parser.add_argument("--steer_ema", type=float, default=0.2, help="Suavizado exponencial del steer [0..1]")

parser.add_argument("--weights", type=str, default="carla_mobilenet_balance.pth",
                    help="Ruta al .pth (state_dict) del MobileNet entrenado")

# Modo de salida:
# - sigmoid01: steer=tanh, throttle/brake=sigmoid -> [0,1] (como en tu primer script)
# - tanh11: steer/throttle/brake=tanh -> [-1,1] y luego mapea throttle/brake a [0,1]
parser.add_argument("--output_mode", type=str, default="sigmoid01", choices=["sigmoid01", "tanh11"],
                    help="Cómo convertir las salidas del modelo a controles.")
args = parser.parse_args()

# ---------------- Device + Modelo ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = CarlaMobileNet().to(device)
state = torch.load(args.weights, map_location=device)
model.load_state_dict(state)
model.eval()

# ---------------- Preprocesado (igual a entrenamiento MobileNet) ----------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_for_model(bgr_frame):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0)  # (1,3,224,224)
    return x

# ---------------- Control ----------------
steer_state = 0.0
use_kickstart = args.kickstart > 0.0

def postprocess_outputs(raw_out):
    """
    raw_out: tensor shape (1,3) crudo.
    Devuelve (steer, throttle, brake) con throttle/brake en [0,1].
    """
    if args.output_mode == "sigmoid01":
        s = torch.tanh(raw_out[:, 0])
        t = torch.sigmoid(raw_out[:, 1])
        b = torch.sigmoid(raw_out[:, 2])
    else:  # tanh11
        s = torch.tanh(raw_out[:, 0])
        t = torch.tanh(raw_out[:, 1])
        b = torch.tanh(raw_out[:, 2])
        # map [-1,1] -> [0,1] para CARLA throttle/brake
        t = (t + 1.0) * 0.5
        b = (b + 1.0) * 0.5

    steer = float(s.item())
    throttle = float(t.item())
    brake = float(b.item())

    # clamps finales
    steer = float(clamp(steer, -1.0, 1.0))
    throttle = float(clamp(throttle, 0.0, 1.0))
    brake = float(clamp(brake, 0.0, 1.0))
    return steer, throttle, brake

def compute_control_from_preds(preds, spd_kmh):
    global steer_state, use_kickstart

    s, t, b = postprocess_outputs(preds)

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

def apply_control(vehicle, s, t, b):
    vehicle.apply_control(carla.VehicleControl(steer=float(s), throttle=float(t), brake=float(b)))

# ---------------- CARLA ----------------
client = carla.Client(args.host, args.port)
client.set_timeout(10.0)
world = client.load_world("Town01")
bp_lib = world.get_blueprint_library()

veh_bps = bp_lib.filter("vehicle.tesla.model3")
vehicle_bp = veh_bps[0] if len(veh_bps) else bp_lib.filter("vehicle.*")[0]

spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise RuntimeError("No hay spawn points en el mapa actual.")
vehicle = None
for sp in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle:
        break
if vehicle is None:
    raise RuntimeError("No se pudo spawnear el vehículo.")

vehicle.set_autopilot(args.autopilot)

# Cámara RGB alta resolución para visualización
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(args.width))
cam_bp.set_attribute("image_size_y", str(args.height))
cam_bp.set_attribute("fov", str(args.fov))
cam_bp.set_attribute("sensor_tick", str(1.0 / max(1, args.fps_sensor)))
cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

# ---------------- Estados compartidos ----------------
last_bgr = None
last_pred = (0.0, 0.0, 0.0)
lock = threading.Lock()
hud_h = 140
scale_display = 1

# ---------------- Callback cámara ----------------
def camera_callback(image):
    global last_bgr, last_pred

    # BGRA -> BGR
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]

    if not args.autopilot:
        inp = preprocess_for_model(bgr).to(device)
        with torch.no_grad():
            preds = model(inp)

        spd = speed_kmh(vehicle)
        s, t, b = compute_control_from_preds(preds, spd)
        apply_control(vehicle, s, t, b)

        with lock:
            last_pred = (s, t, b)

    with lock:
        last_bgr = bgr

camera.listen(camera_callback)

# ---------------- Bucle de visualización ----------------
print("🚗 Conducción en curso. Ventana: 'Model Drive (MobileNet)'.")
print("   Controles: Q/ESC salir | P pausa | A autopilot | K kickstart")
paused = False

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
            status = f"Speed: {spd:5.1f} km/h   Autopilot: {'ON' if args.autopilot else 'OFF'}   Kickstart: {'ON' if use_kickstart else 'OFF'}   Mode: {args.output_mode}"
            cv2.putText(canvas, status, (pad, h + hud_h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (230, 230, 230), 1, cv2.LINE_AA)

            cv2.imshow("Model Drive (MobileNet)", canvas)

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
        camera.stop()
    except:
        pass
    try:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    except:
        pass
    try:
        camera.destroy()
    except:
        pass
    try:
        vehicle.destroy()
    except:
        pass
    cv2.destroyAllWindows()
