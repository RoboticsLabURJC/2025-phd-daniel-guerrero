import os
import time
import csv
from datetime import datetime
from queue import Queue, Empty

import cv2
import numpy as np
import pygame
import carla

# ---------- Parámetros ----------
HOST = "127.0.0.1"
PORT = 2000

IMG_W, IMG_H = 640, 360
CAM_FPS = 20
FIXED_DT = 1.0 / CAM_FPS

AXIS_STEER = 0
AXIS_THROTTLE = 1
AXIS_BRAKE = 2

DEADZONE_STEER = 0.02
DEADZONE_PEDAL = 0.01

EMA_ALPHA_STEER = 0.35
EMA_ALPHA_THROTTLE = 0.25
EMA_ALPHA_BRAKE = 0.25

NOISE_STD_STEER = 0.05
NOISE_STD_THROTTLE = 0.01
NOISE_STD_BRAKE = 0.00
NOISE_ENABLED_DEFAULT = False

SAVE_JPEG_QUALITY = 90

KEEP_PROB_SMALL = 0.10
KEEP_PROB_MED   = 0.50
KEEP_PROB_LARGE = 1.00
STEER_SMALL_TH = 0.05
STEER_LARGE_TH = 0.20

THROTTLE_SCALE = 1.0
# -------------------------------

def to_bgr(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]  # BGR

def pedal_inverted_to_01(v: float, deadzone: float = 0.0) -> float:
    v = float(np.clip(v, -1.0, 1.0))
    if v > 1.0 - deadzone:
        v = 1.0
    return (1.0 - v) / 2.0

def axis_with_deadzone(v: float, dz: float) -> float:
    v = float(np.clip(v, -1.0, 1.0))
    return 0.0 if abs(v) < dz else v

def ema(prev: float, cur: float, alpha: float) -> float:
    return float(alpha * cur + (1.0 - alpha) * prev)

def save_image(out_dir, step, frame_bgr):
    img_rel = f"rgb/{step:06d}.jpg"
    cv2.imwrite(
        os.path.join(out_dir, img_rel),
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(SAVE_JPEG_QUALITY)],
    )
    return img_rel

def build_road_mask(h: int, w: int):
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

def should_keep_frame(steer: float) -> bool:
    a = abs(steer)
    if a < STEER_SMALL_TH:
        p = KEEP_PROB_SMALL
    elif a < STEER_LARGE_TH:
        p = KEEP_PROB_MED
    else:
        p = KEEP_PROB_LARGE
    return (np.random.rand() < p)

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

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    DESIRED_MAP = "Town10HD_Opt"
    current_map = client.get_world().get_map().name
    if DESIRED_MAP not in current_map:
        print(f"[INFO] Cargando mapa {DESIRED_MAP}...")
        world = client.load_world(DESIRED_MAP)
    else:
        print(f"[INFO] Ya estás en {DESIRED_MAP}")
        world = client.get_world()

    original_settings = world.get_settings()

    vehicle = None
    camera = None
    image_queue = Queue()

    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick: {joystick.get_name()} | Ejes: {joystick.get_numaxes()} | Botones: {joystick.get_numbuttons()}")
    else:
        print("[INFO] No hay joystick/volante. Usa teclado: W/S/A/D, SPACE (freno), R (reversa), Q (salir)")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("dagger_runs", run_id)
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)

    log_path = os.path.join(out_dir, "labels.csv")
    print(f"[INFO] Grabando dataset en: {out_dir}")
    print("[INFO] Teclas: Q salir | R reversa | G toggle grabación | N toggle ruido")

    step = 0
    record_enabled = True
    noise_enabled = bool(NOISE_ENABLED_DEFAULT)

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "image",
            "world_frame", "sensor_frame", "sim_time_s",
            "expert_steer", "expert_throttle", "expert_brake", "expert_reverse",
            "applied_steer", "applied_throttle", "applied_brake", "applied_reverse",
            "noise_enabled",
            "noise_std_steer", "noise_std_throttle", "noise_std_brake",
            "mask_roi",
        ])

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "ego")

        spawn_points = world.get_map().get_spawn_points() or [carla.Transform()]
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle:
                break
        if vehicle is None:
            raise RuntimeError("No se pudo spawnear el vehículo.")

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", str(FIXED_DT))

        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)

        cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)

        # Máscara (pero solo se usa al guardar)
        mask = build_road_mask(IMG_H, IMG_W)

        steer = throttle = brake = 0.0
        reverse = False
        hand_brake = False
        control = carla.VehicleControl()

        steer_f = throttle_f = brake_f = 0.0

        world.tick()

        running = True
        last_toggle_time = 0.0

        while running:
            world.tick()
            snap = world.get_snapshot()
            world_frame = snap.frame
            sim_time_s = snap.timestamp.elapsed_seconds

            img = get_image_for_snapshot(image_queue, world_frame, timeout=2.0)

            frame = None
            if img is not None:
                frame = to_bgr(img)

                # MOSTRAR ORIGINAL (sin máscara)
                overlay = frame.copy()
                status = [
                    f"record={record_enabled}",
                    f"noise={noise_enabled}",
                    f"wf={world_frame}",
                    f"sf={img.frame}",
                ]
                cv2.putText(overlay, " | ".join(status), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("CAM", overlay)
                cv2.waitKey(1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            now = time.time()

            if keys[pygame.K_g] and (now - last_toggle_time) > 0.25:
                record_enabled = not record_enabled
                print(f"[INFO] record_enabled -> {record_enabled}")
                last_toggle_time = now

            if keys[pygame.K_n] and (now - last_toggle_time) > 0.25:
                noise_enabled = not noise_enabled
                print(f"[INFO] noise_enabled -> {noise_enabled}")
                last_toggle_time = now

            if keys[pygame.K_q]:
                running = False

            # Entrada humano
            if joystick is not None:
                steer = axis_with_deadzone(joystick.get_axis(AXIS_STEER), DEADZONE_STEER)
                throttle = float(np.clip(pedal_inverted_to_01(joystick.get_axis(AXIS_THROTTLE), DEADZONE_PEDAL), 0.0, 1.0))
                brake = float(np.clip(pedal_inverted_to_01(joystick.get_axis(AXIS_BRAKE), DEADZONE_PEDAL), 0.0, 1.0))

                if keys[pygame.K_r]:
                    reverse = True
                if (not keys[pygame.K_r]) and reverse and throttle == 0.0:
                    reverse = False
            else:
                # fallback teclado mínimo
                if keys[pygame.K_a]:
                    steer -= 2.5 * FIXED_DT
                elif keys[pygame.K_d]:
                    steer += 2.5 * FIXED_DT
                else:
                    steer *= 0.9

                if keys[pygame.K_w]:
                    throttle = min(1.0, throttle + 0.02); brake = 0.0
                elif keys[pygame.K_s]:
                    brake = min(1.0, brake + 0.05); throttle = max(0.0, throttle - 0.02)
                else:
                    throttle *= 0.98
                    brake *= 0.9

                if keys[pygame.K_SPACE]:
                    brake = 1.0

                if keys[pygame.K_r]:
                    reverse = True
                if (not keys[pygame.K_r]) and reverse and throttle == 0.0:
                    reverse = False

            steer = float(np.clip(steer, -1.0, 1.0))
            throttle = float(np.clip(throttle * THROTTLE_SCALE, 0.0, 1.0))
            brake = float(np.clip(brake, 0.0, 1.0))

            steer_f = ema(steer_f, steer, EMA_ALPHA_STEER)
            throttle_f = ema(throttle_f, throttle, EMA_ALPHA_THROTTLE)
            brake_f = ema(brake_f, brake, EMA_ALPHA_BRAKE)

            expert = {
                "steer": float(np.clip(steer_f, -1.0, 1.0)),
                "throttle": float(np.clip(throttle_f, 0.0, 1.0)),
                "brake": float(np.clip(brake_f, 0.0, 1.0)),
                "reverse": bool(reverse),
            }

            applied = dict(expert)
            if noise_enabled:
                applied["steer"] = float(np.clip(applied["steer"] + np.random.normal(0.0, NOISE_STD_STEER), -1.0, 1.0))
                applied["throttle"] = float(np.clip(applied["throttle"] + np.random.normal(0.0, NOISE_STD_THROTTLE), 0.0, 1.0))
                applied["brake"] = float(np.clip(applied["brake"] + np.random.normal(0.0, NOISE_STD_BRAKE), 0.0, 1.0))

            control.steer = applied["steer"]
            control.throttle = applied["throttle"]
            control.brake = applied["brake"]
            control.hand_brake = hand_brake
            control.reverse = applied["reverse"]
            vehicle.apply_control(control)

            # Guardar SOLO dataset enmascarado (sin cielo)
            if record_enabled and (frame is not None):
                if should_keep_frame(expert["steer"]):
                    frame_masked = apply_mask(frame, mask)  # AQUÍ se aplica la máscara
                    img_rel = save_image(out_dir, step, frame_masked)
                    with open(log_path, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([
                            step, img_rel,
                            int(world_frame), int(img.frame), float(sim_time_s),
                            expert["steer"], expert["throttle"], expert["brake"], int(expert["reverse"]),
                            applied["steer"], applied["throttle"], applied["brake"], int(applied["reverse"]),
                            int(noise_enabled),
                            float(NOISE_STD_STEER), float(NOISE_STD_THROTTLE), float(NOISE_STD_BRAKE),
                            1
                        ])
                    step += 1

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
