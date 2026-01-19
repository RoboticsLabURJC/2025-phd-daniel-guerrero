# control_carla_dagger_continuous_noise_csv.py
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

IMG_W, IMG_H = 1280, 920
CAM_FPS = 20

STEER_RATE = 2.5
THROTTLE_STEP = 0.02
BRAKE_STEP = 0.05

# --- Mapeo volante/pedales ---
AXIS_STEER = 0
AXIS_THROTTLE = 1
AXIS_BRAKE = 2

DEADZONE_STEER = 0.02
DEADZONE_PEDAL = 0.01

# --- DAgger-style: continuous action noise injection + expert labeling ---
# (Caso A: ruido pequeño en cada tick; etiqueta = experto)
NOISE_STD_STEER = 0.05        # 0.02–0.10 típico
NOISE_STD_THROTTLE = 0.01     # opcional, pequeño
NOISE_STD_BRAKE = 0.00        # normalmente 0
NOISE_ENABLED_DEFAULT = True  # arranca con ruido activo

SAVE_JPEG_QUALITY = 90
# -------------------------------


def to_bgr(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]  # BGR


def pedal_inverted_to_01(v: float, deadzone: float = 0.0) -> float:
    """
    Convierte un valor de pedal en rango [1..-1] a [0..1]
    1   -> 0.0 (no pisado)
    -1  -> 1.0 (a fondo)
    """
    v = float(np.clip(v, -1.0, 1.0))
    if v > 1.0 - deadzone:
        v = 1.0
    return (1.0 - v) / 2.0


def axis_with_deadzone(v: float, dz: float) -> float:
    v = float(np.clip(v, -1.0, 1.0))
    return 0.0 if abs(v) < dz else v


def save_image(out_dir, step, frame_bgr):
    img_rel = f"rgb/{step:06d}.jpg"
    cv2.imwrite(
        os.path.join(out_dir, img_rel),
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(SAVE_JPEG_QUALITY)],
    )
    return img_rel


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(5.0)

    DESIRED_MAP = "Town01"
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

    # Inicializa pygame
    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(
            f"[INFO] Joystick: {joystick.get_name()} | Ejes: {joystick.get_numaxes()} | Botones: {joystick.get_numbuttons()}"
        )
    else:
        print("[INFO] No hay joystick/volante. Usa teclado: W/S/A/D, SPACE (freno), R (reversa), Q (salir)")

    # --- Carpeta de salida ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("dagger_runs", run_id)
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)

    # CSV
    log_path = os.path.join(out_dir, "labels.csv")
    print(f"[INFO] Guardando dataset (continuous noise) en: {out_dir}")

    # Estado
    step = 0
    record_enabled = True
    noise_enabled = bool(NOISE_ENABLED_DEFAULT)

    # Escribir header CSV
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step",
            "image",
            "expert_steer", "expert_throttle", "expert_brake", "expert_reverse",
            "applied_steer", "applied_throttle", "applied_brake", "applied_reverse",
            "mode",
            "noise_enabled",
            "noise_std_steer",
            "noise_std_throttle",
            "noise_std_brake",
        ])

    try:
        # ----- Modo síncrono -----
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / CAM_FPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        # ----- Vehículo -----
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "ego")

        spawn_points = world.get_map().get_spawn_points() or [carla.Transform()]
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle:
                break
        if vehicle is None:
            raise RuntimeError("No se pudo spawnear el vehículo (mapa sin puntos libres).")

        # ----- Cámara RGB -----
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", str(1.0 / CAM_FPS))

        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)

        cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)

        # Variables de control
        steer = 0.0
        throttle = 0.0
        brake = 0.0
        reverse = False
        hand_brake = False
        control = carla.VehicleControl()

        # Primer tick
        world.tick()

        print("[INFO] Teclas: Q salir | R reversa | G toggle grabación | N toggle ruido")

        running = True
        while running:
            # Tick sim (sincrónico)
            world.tick()

            # Leer la imagen más reciente
            img = None
            while True:
                try:
                    img = image_queue.get_nowait()
                except Empty:
                    break

            frame = None
            if img is not None:
                frame = to_bgr(img)

                overlay = frame.copy()
                status = [
                    f"record={record_enabled}",
                    f"noise={noise_enabled}",
                    f"std_steer={NOISE_STD_STEER}",
                ]
                cv2.putText(
                    overlay, " | ".join(status),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2
                )
                cv2.imshow("CAM", overlay)
                cv2.waitKey(1)

            # Eventos pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # toggle grabación (anti-rebote simple)
            if keys[pygame.K_g]:
                record_enabled = not record_enabled
                print(f"[INFO] record_enabled -> {record_enabled}")
                time.sleep(0.2)

            # toggle ruido (anti-rebote simple)
            if keys[pygame.K_n]:
                noise_enabled = not noise_enabled
                print(f"[INFO] noise_enabled -> {noise_enabled}")
                time.sleep(0.2)

            # Cerrar con Q
            if keys[pygame.K_q]:
                running = False

            # --- Entrada humano (experto) ---
            if joystick is None:
                # Dirección
                if keys[pygame.K_a]:
                    steer -= STEER_RATE * settings.fixed_delta_seconds
                elif keys[pygame.K_d]:
                    steer += STEER_RATE * settings.fixed_delta_seconds
                else:
                    steer *= 0.9

                # Acelerador / freno
                if keys[pygame.K_w]:
                    throttle = min(1.0, throttle + THROTTLE_STEP)
                    brake = 0.0
                elif keys[pygame.K_s]:
                    brake = min(1.0, brake + BRAKE_STEP)
                    throttle = max(0.0, throttle - THROTTLE_STEP)
                else:
                    throttle *= 0.98
                    brake *= 0.9

                if keys[pygame.K_SPACE]:
                    brake = 1.0

                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

            else:
                # volante
                try:
                    steer_val = joystick.get_axis(AXIS_STEER)
                except Exception:
                    steer_val = 0.0
                steer = axis_with_deadzone(steer_val, DEADZONE_STEER)

                # acelerador
                try:
                    accel_val = joystick.get_axis(AXIS_THROTTLE)
                except Exception:
                    accel_val = 1.0
                throttle = float(np.clip(pedal_inverted_to_01(accel_val, DEADZONE_PEDAL), 0.0, 1.0))

                # freno
                try:
                    brake_val = joystick.get_axis(AXIS_BRAKE)
                except Exception:
                    brake_val = 1.0
                brake = float(np.clip(pedal_inverted_to_01(brake_val, DEADZONE_PEDAL), 0.0, 1.0))

                # reversa con R
                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

            # Limitar entrada experto
            steer = float(np.clip(steer, -1.0, 1.0))
            throttle = float(np.clip(throttle, 0.0, 1.0)) / 4.0  # tu escala original
            brake = float(np.clip(brake, 0.0, 1.0))

            # Etiqueta (lo correcto)
            expert = {
                "steer": steer,
                "throttle": throttle,
                "brake": brake,
                "reverse": bool(reverse),
            }

            # ---------------------------------------------------------
            # CASO A: Continuous noise injection (acción aplicada = expert + ruido)
            # ---------------------------------------------------------
            applied = dict(expert)
            if noise_enabled:
                applied["steer"] = float(np.clip(
                    applied["steer"] + np.random.normal(0.0, NOISE_STD_STEER),
                    -1.0, 1.0
                ))
                applied["throttle"] = float(np.clip(
                    applied["throttle"] + np.random.normal(0.0, NOISE_STD_THROTTLE),
                    0.0, 1.0
                ))
                applied["brake"] = float(np.clip(
                    applied["brake"] + np.random.normal(0.0, NOISE_STD_BRAKE),
                    0.0, 1.0
                ))

            # ---------------------------------------------------------
            # Aplicar control a CARLA
            # ---------------------------------------------------------
            control.steer = float(np.clip(applied["steer"], -1.0, 1.0))
            control.throttle = float(np.clip(applied["throttle"], 0.0, 1.0))
            control.brake = float(np.clip(applied["brake"], 0.0, 1.0))
            control.hand_brake = hand_brake
            control.reverse = bool(applied["reverse"])
            vehicle.apply_control(control)

            # ---------------------------------------------------------
            # Guardar (imagen + CSV) con etiqueta = expert
            # ---------------------------------------------------------
            if record_enabled and (frame is not None):
                img_rel = save_image(out_dir, step, frame)

                mode = "dagger_continuous_noise"
                with open(log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        step,
                        img_rel,

                        expert["steer"],
                        expert["throttle"],
                        expert["brake"],
                        int(expert["reverse"]),

                        applied["steer"],
                        applied["throttle"],
                        applied["brake"],
                        int(applied["reverse"]),

                        mode,
                        int(noise_enabled),
                        NOISE_STD_STEER,
                        NOISE_STD_THROTTLE,
                        NOISE_STD_BRAKE,
                    ])
                step += 1

        cv2.destroyAllWindows()

    finally:
        # Limpieza
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
