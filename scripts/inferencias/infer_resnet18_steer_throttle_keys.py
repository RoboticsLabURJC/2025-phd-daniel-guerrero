# infer_resnet18_steer_throttle_keys.py
import time
from queue import Queue, Empty

import cv2
import numpy as np
import pygame
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import carla

# ---------------- Config ----------------
HOST = "127.0.0.1"
PORT = 2000

TOWN = "Town10HD_Opt"   # cambia si quieres
CAM_FPS = 20
FIXED_DT = 1.0 / CAM_FPS

IMG_W, IMG_H = 640, 360

# Modelo entrenado (ajusta la ruta)
MODEL_PATH = "resnet18_steer_best.pt"

# Throttle fijo inicial (ajustable en ejecución)
THROTTLE_INIT = 0.30
THROTTLE_STEP = 0.02
THROTTLE_MIN = 0.00
THROTTLE_MAX = 0.80

# Suavizado de steer aplicado (para que no vibre)
STEER_EMA_ALPHA = 0.35

# Ajuste manual (trim) de steer en vivo (flechas izq/der)
STEER_TRIM_INIT = 0.0
STEER_TRIM_STEP = 0.02
STEER_TRIM_MIN = -0.40
STEER_TRIM_MAX = 0.40

# ----------------------------------------

def to_bgr(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]  # BGR

def build_road_mask(h: int, w: int):
    # Mismo trapecio que usamos al generar dataset
    pts = np.array([[
        (int(0.10 * w), int(0.98 * h)),
        (int(0.90 * w), int(0.98 * h)),
        (int(0.62 * w), int(0.55 * h)),
        (int(0.38 * w), int(0.55 * h)),
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask

def apply_mask(frame_bgr, mask_u8):
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_u8)

def ema(prev: float, cur: float, alpha: float) -> float:
    return float(alpha * cur + (1.0 - alpha) * prev)

def get_image_for_snapshot(image_queue: Queue, target_frame: int, timeout=2.0):
    t0 = time.time()
    last = None
    while True:
        remaining = max(0.01, timeout - (time.time() - t0))
        try:
            img = image_queue.get(timeout=remaining)
            last = img
            if img.frame == target_frame:
                return img
            if img.frame < target_frame:
                continue
            if img.frame > target_frame:
                return img
        except Empty:
            return last

class ResNet18Steer(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=None)  # pesos vienen del checkpoint
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.backbone = m  # <- debe llamarse backbone (como en el checkpoint)

    def forward(self, x):
        return torch.tanh(self.backbone(x))

def main():
    # Torch device (ROCm suele aparecer como "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # Cargar modelo
    model = ResNet18Steer().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("[INFO] Modelo cargado:", MODEL_PATH)

    # Transform igual al entrenamiento
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # CARLA
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    # Cargar mapa si hace falta
    current_map = client.get_world().get_map().name
    if TOWN not in current_map:
        print(f"[INFO] Cargando mapa {TOWN}...")
        world = client.load_world(TOWN)
    else:
        world = client.get_world()

    original_settings = world.get_settings()

    vehicle = None
    camera = None
    image_queue = Queue()

    # Pygame para teclas
    pygame.init()
    pygame.display.set_mode((360, 90))  # ventana mínima para capturar teclado
    pygame.display.set_caption("Controls (focus here)")

    try:
        # Sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        # Vehículo
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "ego")

        for sp in world.get_map().get_spawn_points():
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle:
                break
        if vehicle is None:
            raise RuntimeError("No se pudo spawnear el vehículo.")

        # Cámara
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", str(FIXED_DT))

        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)

        cv2.namedWindow("CARLA", cv2.WINDOW_AUTOSIZE)

        # Máscara
        mask = build_road_mask(IMG_H, IMG_W)

        throttle = THROTTLE_INIT
        steer_smoothed = 0.0
        steer_trim = STEER_TRIM_INIT
        steer_cmd = 0.0

        print("[INFO] Controles (haz click en la ventana 'Controls'):")
        print("  Q: salir")
        print("  Flecha ARRIBA / + : subir throttle")
        print("  Flecha ABAJO / -  : bajar throttle")
        print("  Flecha IZQ/DER     : steer trim -/+ (acomodar en carril)")
        print("  0                  : reset trim")
        print("  ESPACIO            : freno completo (mientras mantienes)")
        print("  H                  : mostrar/ocultar vista enmascarada (solo visual)")

        show_masked_view = False

        # Tick inicial
        world.tick()

        running = True
        last_h_toggle = 0.0

        while running:
            world.tick()
            snap = world.get_snapshot()
            wf = snap.frame

            img = get_image_for_snapshot(image_queue, wf, timeout=2.0)
            if img is None:
                continue

            frame = to_bgr(img)

            # --- Teclas (pygame) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_q]:
                running = False

            # Ajuste de throttle en vivo
            if keys[pygame.K_UP] or keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                throttle = min(THROTTLE_MAX, throttle + THROTTLE_STEP)
            if keys[pygame.K_DOWN] or keys[pygame.K_MINUS]:
                throttle = max(THROTTLE_MIN, throttle - THROTTLE_STEP)

            # Ajuste manual de steer (trim)
            if keys[pygame.K_LEFT]:
                steer_trim = max(STEER_TRIM_MIN, steer_trim - STEER_TRIM_STEP)
            if keys[pygame.K_RIGHT]:
                steer_trim = min(STEER_TRIM_MAX, steer_trim + STEER_TRIM_STEP)
            if keys[pygame.K_0]:
                steer_trim = 0.0

            # Toggle vista (anti-rebote sin congelar todo)
            if keys[pygame.K_h]:
                now = time.time()
                if now - last_h_toggle > 0.25:
                    show_masked_view = not show_masked_view
                    last_h_toggle = now

            # --- Inferencia ---
            frame_masked = apply_mask(frame, mask)
            rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = tf(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                steer_pred = model(x).item()

            steer_smoothed = ema(steer_smoothed, steer_pred, STEER_EMA_ALPHA)
            steer_smoothed = float(np.clip(steer_smoothed, -1.0, 1.0))

            # Comando final con trim
            steer_cmd = float(np.clip(steer_smoothed + steer_trim, -1.0, 1.0))

            # --- Control ---
            control = carla.VehicleControl()
            control.steer = steer_cmd
            control.throttle = float(np.clip(throttle, 0.0, 1.0))
            control.brake = 1.0 if keys[pygame.K_SPACE] else 0.0
            control.reverse = False
            vehicle.apply_control(control)

            # --- Visualización ---
            view = frame_masked if show_masked_view else frame
            overlay = view.copy()
            cv2.putText(
                overlay,
                f"thr={throttle:.2f} | steer={steer_smoothed:+.3f} | trim={steer_trim:+.3f} | cmd={steer_cmd:+.3f} | view={'masked' if show_masked_view else 'orig'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("CARLA", overlay)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    finally:
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass
        try:
            if camera is not None:
                camera.destroy()
        except Exception:
            pass
        try:
            if vehicle is not None:
                vehicle.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        pygame.quit()

if __name__ == "__main__":
    main()
