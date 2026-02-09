# drive_mobilenet_cropped_town10hd_opt_autopilot.py
import argparse
import threading
import time

import carla
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def clamp(x, a, b):
    return max(a, min(b, x))

def ema(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5


class CarlaMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base.last_channel, 3)
        )

    def forward(self, x):
        return self.base(x)


# ---------------- Args ----------------
parser = argparse.ArgumentParser("Drive MobileNet (cropped) on Town10HD_Opt + Autopilot toggle")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=2000)

parser.add_argument("--weights", type=str, required=True, help="Ruta al .pth entrenado (cropped)")

parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=920)
parser.add_argument("--fov", type=float, default=90.0)
parser.add_argument("--fps_sensor", type=int, default=20)

parser.add_argument("--crop_top", type=int, default=300)
parser.add_argument("--crop_bottom", type=int, default=80)

parser.add_argument("--steer_ema", type=float, default=0.2)
parser.add_argument("--kickstart", type=float, default=0.15)   # 0 desactiva
parser.add_argument("--kick_speed", type=float, default=1.0)

parser.add_argument("--curve_throttle_cap", type=float, default=0.12)
parser.add_argument("--curve_steer_thr", type=float, default=0.35)

parser.add_argument("--autopilot", action="store_true", help="Iniciar con autopilot ON")
args = parser.parse_args()


# ---------------- Device + load ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = CarlaMobileNet().to(device)
state = torch.load(args.weights, map_location=device)
model.load_state_dict(state)
model.eval()


# ---------------- Transform (igual que entrenamiento) ----------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_for_model(bgr_frame: np.ndarray) -> torch.Tensor:
    h, w = bgr_frame.shape[:2]
    top = int(clamp(args.crop_top, 0, h - 2))
    bottom = int(clamp(args.crop_bottom, 0, h - top - 1))

    cropped = bgr_frame[top:h-bottom, :, :]
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0)
    return x


# ---------------- Control logic ----------------
steer_state = 0.0
use_kickstart = args.kickstart > 0.0

def postprocess(raw_out: torch.Tensor):
    s = torch.tanh(raw_out[:, 0])
    t = torch.sigmoid(raw_out[:, 1])
    b = torch.sigmoid(raw_out[:, 2])
    return float(s.item()), float(t.item()), float(b.item())

def compute_control(raw_out: torch.Tensor, spd_kmh: float):
    global steer_state, use_kickstart

    s, t, b = postprocess(raw_out)

    steer_state = ema(steer_state, s, args.steer_ema)
    s = steer_state

    if t > 0.05:
        b = 0.0

    if abs(s) > args.curve_steer_thr:
        t = min(t, args.curve_throttle_cap)
        b = 0.0

    if use_kickstart and spd_kmh < args.kick_speed:
        t = max(t, args.kickstart)
        b = 0.0

    s = float(clamp(s, -1.0, 1.0))
    t = float(clamp(t, 0.0, 1.0))
    b = float(clamp(b, 0.0, 1.0))
    return s, t, b


# ---------------- CARLA setup ----------------
client = carla.Client(args.host, args.port)
client.set_timeout(60.0)

print("[INFO] Cargando Town10HD_Opt ...")
world = client.load_world("Town10HD_Opt")

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

# Autopilot inicial
autopilot_on = bool(args.autopilot)
vehicle.set_autopilot(autopilot_on)

# Cámara
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", str(args.width))
cam_bp.set_attribute("image_size_y", str(args.height))
cam_bp.set_attribute("fov", str(args.fov))
cam_bp.set_attribute("sensor_tick", str(1.0 / max(1, args.fps_sensor)))

cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

last_bgr = None
last_pred = (0.0, 0.0, 0.0)
lock = threading.Lock()


def camera_callback(image: carla.Image):
    global last_bgr, last_pred, autopilot_on

    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3]

    # Solo aplica el modelo si autopilot está OFF
    if not autopilot_on:
        inp = preprocess_for_model(bgr).to(device)
        with torch.no_grad():
            raw = model(inp)

        spd = speed_kmh(vehicle)
        s, t, b = compute_control(raw, spd)
        vehicle.apply_control(carla.VehicleControl(steer=s, throttle=t, brake=b))

        with lock:
            last_pred = (s, t, b)

    with lock:
        last_bgr = bgr


camera.listen(camera_callback)

print("🚗 Driving on Town10HD_Opt")
print("Controles: Q/ESC salir | A toggle autopilot")

try:
    while True:
        with lock:
            frame = None if last_bgr is None else last_bgr.copy()
            s, t, b = last_pred
            ap = autopilot_on

        if frame is not None:
            overlay = frame.copy()
            txt1 = f"Autopilot: {'ON' if ap else 'OFF'}"
            txt2 = f"steer={s:+.3f}  thr={t:.3f}  brk={b:.3f}"
            cv2.putText(overlay, txt1, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(overlay, txt2, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("Drive (MobileNet Cropped)", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('a'):
            autopilot_on = not autopilot_on
            vehicle.set_autopilot(autopilot_on)
            # reset steer EMA para evitar salto cuando vuelves al modelo
            if not autopilot_on:
                steer_state = 0.0

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
