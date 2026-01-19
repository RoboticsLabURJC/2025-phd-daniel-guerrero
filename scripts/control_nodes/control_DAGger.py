# control_carla_dagger_recovery_only.py
import os
import json
import time
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

# --- DAgger-style "recovery-only" logging + noise ---
NOISE_INTERVAL_SEC = 20.0      # cada cuánto meter ruido
NOISE_TICKS = 20                # cuántos ticks dura el empujón (NO se graba)
STEER_NOISE_MAG = 0.25         # magnitud del ruido en steer (0.15-0.35 típico)
RECOVERY_SEC = 3.0             # cuánto grabar tras el ruido (solo recuperación)

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


def save_sample(out_dir, step, frame_bgr, expert, applied, meta):
    img_rel = f"rgb/{step:06d}.jpg"
    cv2.imwrite(
        os.path.join(out_dir, img_rel),
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(SAVE_JPEG_QUALITY)],
    )
    row = {
        "step": step,
        "image": img_rel,
        "expert": expert,     # etiqueta (lo que tú hiciste)
        "applied": applied,   # lo que se aplicó (puede incluir ruido)
        "meta": meta,
    }
    return row


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
        print(f"[INFO] Joystick: {joystick.get_name()} | Ejes: {joystick.get_numaxes()} | Botones: {joystick.get_numbuttons()}")
    else:
        print("[INFO] No hay joystick/volante. Usa teclado: W/S/A/D, SPACE (freno), R (reversa), Q (salir)")

    # --- Carpeta de salida ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("dagger_runs", run_id)
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)
    log_path = os.path.join(out_dir, "labels.jsonl")
    print(f"[INFO] Guardando recovery-only dataset en: {out_dir}")

    # Estado DAgger-style
    step = 0
    record_enabled = True

    # Control de “ruido” y “recovery”
    noise_ticks_left = 0
    recovery_ticks_left = 0

    # Temporizador de ruido cada 20s (en sim time)
    sim_since_last_noise = 0.0

    def trigger_noise_and_recovery():
        nonlocal noise_ticks_left, recovery_ticks_left, sim_since_last_noise
        noise_ticks_left = int(NOISE_TICKS)
        recovery_ticks_left = int(round(RECOVERY_SEC * CAM_FPS))
        sim_since_last_noise = 0.0
        print(f"[NOISE] Trigger -> NOISE_TICKS={noise_ticks_left}, RECOVERY_TICKS={recovery_ticks_left}")

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

        print("[INFO] Teclas: Q salir | R reversa | G toggle grabación | (Ruido automático cada 20s)")
        running = True

        while running:
            # Tick sim (sincrónico)
            world.tick()

            # sim time para programar ruido
            sim_since_last_noise += float(settings.fixed_delta_seconds)

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
                # overlay estado
                overlay = frame.copy()
                status = []
                status.append(f"record={record_enabled}")
                status.append(f"noise_ticks_left={noise_ticks_left}")
                status.append(f"recovery_ticks_left={recovery_ticks_left}")
                status.append(f"t_since_noise={sim_since_last_noise:0.1f}s")
                text = " | ".join(status)
                cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("CAM", overlay)
                cv2.waitKey(1)

            # Eventos pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # toggle grabación
            if keys[pygame.K_g]:
                # anti-rebote sencillo
                record_enabled = not record_enabled
                print(f"[INFO] record_enabled -> {record_enabled}")
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

            expert = {
                "steer": steer,
                "throttle": throttle,
                "brake": brake,
                "reverse": bool(reverse),
            }

            # ---------------------------------------------------------
            # 1) Disparar ruido automáticamente cada NOISE_INTERVAL_SEC
            #    Solo si NO estamos ya en ruido/recovery
            # ---------------------------------------------------------
            if noise_ticks_left == 0 and recovery_ticks_left == 0 and sim_since_last_noise >= NOISE_INTERVAL_SEC:
                trigger_noise_and_recovery()

            # ---------------------------------------------------------
            # 2) Construir control aplicado (se aplica al coche)
            #    Base = experto, pero con ruido en steer durante NOISE
            # ---------------------------------------------------------
            applied = dict(expert)
            used_noise = False

            if noise_ticks_left > 0:
                # Ruido breve (NO grabar estos ticks)
                sign = 1.0 if np.random.randn() >= 0 else -1.0
                applied["steer"] = float(np.clip(applied["steer"] + sign * STEER_NOISE_MAG, -1.0, 1.0))
                used_noise = True
                noise_ticks_left -= 1

                # cuando termina NOISE, ya estamos listos para RECOVERY (ticks siguientes)
                # recovery_ticks_left ya estaba seteado

            elif recovery_ticks_left > 0:
                # Estamos en recuperación (SI se graba)
                recovery_ticks_left -= 1

            # ---------------------------------------------------------
            # 3) Aplicar control a CARLA
            # ---------------------------------------------------------
            control.steer = float(np.clip(applied["steer"], -1.0, 1.0))
            control.throttle = float(np.clip(applied["throttle"], 0.0, 1.0))
            control.brake = float(np.clip(applied["brake"], 0.0, 1.0))
            control.hand_brake = hand_brake
            control.reverse = bool(applied["reverse"])
            vehicle.apply_control(control)

            # ---------------------------------------------------------
            # 4) Guardar SOLO RECOVERY (y nunca NOISE)
            # ---------------------------------------------------------
            in_recovery = (recovery_ticks_left > 0) and (noise_ticks_left == 0)
            in_noise = (noise_ticks_left > 0) or used_noise

            if record_enabled and (frame is not None) and in_recovery and (not in_noise):
                meta = {
                    "mode": "recovery_only",
                    "noise_interval_sec": NOISE_INTERVAL_SEC,
                    "noise_mag": STEER_NOISE_MAG,
                    "noise_ticks": NOISE_TICKS,
                    "recovery_sec": RECOVERY_SEC,
                    "sim_time_since_last_noise": sim_since_last_noise,
                }
                row = save_sample(
                    out_dir, step, frame,
                    expert=expert,
                    applied=applied,
                    meta=meta
                )
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                step += 1

        cv2.destroyAllWindows()

    finally:
        # Limpieza
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        if camera is not None:
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        pygame.quit()


if __name__ == "__main__":
    main()
