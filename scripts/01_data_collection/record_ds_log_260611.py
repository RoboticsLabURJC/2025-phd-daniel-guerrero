import os
import time
import random
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

THROTTLE_SCALE = 1.0

# --- Parámetros DAgger ---
DAGGER_INTERVAL = 10.0  # Segundos entre cada perturbación
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

    DESIRED_MAP = "Town02"
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
    collision_sensor = None
    image_queue = Queue()

    has_collided = False
    def collision_handler(event):
        nonlocal has_collided
        has_collided = True

    # --- INICIALIZACIÓN DE PYGAME Y VENTANA ---
    pygame.init()
    display = pygame.display.set_mode((IMG_W, IMG_H))
    pygame.display.set_caption("CARLA Data Collection + DAgger")
    
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick: {joystick.get_name()} | Ejes: {joystick.get_numaxes()}")
    else:
        print("[INFO] No hay joystick/volante. Usa teclado: W/S/A/D, SPACE (freno), R (reversa), Q (salir)")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(f"recording_{run_id}.log")
    
    print(f"[INFO] Grabando telemetría en archivo LOG: {log_path}")
    print("[INFO] Teclas: Q salir | R reversa | N toggle ruido")

    noise_enabled = bool(NOISE_ENABLED_DEFAULT)

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

        # Sensor de Cámara
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", str(FIXED_DT))
        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)

        # Sensor de Colisiones
        col_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(collision_handler)

        mask = build_road_mask(IMG_H, IMG_W)

        steer = throttle = brake = 0.0
        reverse = False
        hand_brake = False
        control = carla.VehicleControl()

        steer_f = throttle_f = brake_f = 0.0

        client.start_recorder(log_path, True)

        world.tick()
        running = True
        last_toggle_time = 0.0
        
        # Reloj interno de simulación para DAgger
        sim_start_time = world.get_snapshot().timestamp.elapsed_seconds
        last_dagger_time = sim_start_time

        while running:
            world.tick()
            snap = world.get_snapshot()
            world_frame = snap.frame
            current_sim_time = snap.timestamp.elapsed_seconds

            # --- SISTEMA DE RESCATE ---
            if has_collided or vehicle.get_location().z < -2.0:
                print("[ALERTA] Choque o salida del mapa. Reubicando vehículo...")
                new_sp = random.choice(spawn_points)
                wp = world.get_map().get_waypoint(new_sp.location)
                new_tf = wp.transform
                new_tf.location.z += 1.0 # Altura de seguridad
                vehicle.set_transform(new_tf)
                
                # Reseteamos controles y banderas
                steer = throttle = brake = 0.0
                steer_f = throttle_f = brake_f = 0.0
                has_collided = False
                last_dagger_time = current_sim_time
                continue # Saltamos este frame para que se asiente

            # --- PERTURBACIÓN DAGGER ---
            if current_sim_time - last_dagger_time > DAGGER_INTERVAL:
                tf = vehicle.get_transform()
                right_vec = tf.get_right_vector()
                
                # Desplazamiento aleatorio: 1.0m a 1.5m
                dist = random.uniform(1.0, 1.5)
                # Dirección aleatoria: 1 (derecha) o -1 (izquierda)
                direction = random.choice([-1, 1])
                
                tf.location.x += right_vec.x * dist * direction
                tf.location.y += right_vec.y * dist * direction
                tf.location.z += 0.2 # Ligera elevación
                
                # Ángulo aleatorio: 5 a 15 grados
                angle_offset = random.uniform(5.0, 15.0) * direction
                tf.rotation.yaw += angle_offset
                
                vehicle.set_transform(tf)
                dir_str = "Derecha" if direction == 1 else "Izquierda"
                print(f"[DAgger] Salto a la {dir_str}: {dist:.2f}m | Giro: {angle_offset:.1f}°")
                
                last_dagger_time = current_sim_time

            # --- PROCESAMIENTO DE IMAGEN Y HUD ---
            img = get_image_for_snapshot(image_queue, world_frame, timeout=2.0)

            if img is not None:
                frame = to_bgr(img)
                overlay = frame.copy()
                status = [
                    f"REC: ON",
                    f"DAgger: {DAGGER_INTERVAL - (current_sim_time - last_dagger_time):.1f}s",
                    f"wf={world_frame}",
                ]
                cv2.putText(overlay, " | ".join(status), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                surface = pygame.surfarray.make_surface(overlay_rgb.swapaxes(0, 1))
                
                display.blit(surface, (0, 0))
                pygame.display.flip()

            # --- CONTROLES MANUALES (Intactos) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            now = time.time()

            if keys[pygame.K_n] and (now - last_toggle_time) > 0.25:
                noise_enabled = not noise_enabled
                print(f"[INFO] noise_enabled -> {noise_enabled}")
                last_toggle_time = now

            if keys[pygame.K_q]:
                running = False

            if joystick is not None:
                steer = axis_with_deadzone(joystick.get_axis(AXIS_STEER), DEADZONE_STEER)
                throttle = float(np.clip(pedal_inverted_to_01(joystick.get_axis(AXIS_THROTTLE), DEADZONE_PEDAL), 0.0, 1.0))
                brake = float(np.clip(pedal_inverted_to_01(joystick.get_axis(AXIS_BRAKE), DEADZONE_PEDAL), 0.0, 1.0))

                if keys[pygame.K_r]:
                    reverse = True
                if (not keys[pygame.K_r]) and reverse and throttle == 0.0:
                    reverse = False
            else:
                if keys[pygame.K_a]: steer -= 2.5 * FIXED_DT
                elif keys[pygame.K_d]: steer += 2.5 * FIXED_DT
                else: steer *= 0.9

                if keys[pygame.K_w]:
                    throttle = min(1.0, throttle + 0.02); brake = 0.0
                elif keys[pygame.K_s]:
                    brake = min(1.0, brake + 0.05); throttle = max(0.0, throttle - 0.02)
                else:
                    throttle *= 0.98
                    brake *= 0.9

                if keys[pygame.K_SPACE]: brake = 1.0

                if keys[pygame.K_r]: reverse = True
                if (not keys[pygame.K_r]) and reverse and throttle == 0.0: reverse = False

            steer = float(np.clip(steer, -1.0, 1.0))
            throttle = float(np.clip(throttle * THROTTLE_SCALE, 0.0, 1.0))
            brake = float(np.clip(brake, 0.0, 1.0))

            steer_f = ema(steer_f, steer, EMA_ALPHA_STEER)
            throttle_f = ema(throttle_f, throttle, EMA_ALPHA_THROTTLE)
            brake_f = ema(brake_f, brake, EMA_ALPHA_BRAKE)

            applied_steer = steer_f
            applied_throttle = throttle_f
            applied_brake = brake_f

            if noise_enabled:
                applied_steer = float(np.clip(applied_steer + np.random.normal(0.0, NOISE_STD_STEER), -1.0, 1.0))
                applied_throttle = float(np.clip(applied_throttle + np.random.normal(0.0, NOISE_STD_THROTTLE), 0.0, 1.0))
                applied_brake = float(np.clip(applied_brake + np.random.normal(0.0, NOISE_STD_BRAKE), 0.0, 1.0))

            control.steer = applied_steer
            control.throttle = applied_throttle
            control.brake = applied_brake
            control.hand_brake = hand_brake
            control.reverse = reverse
            
            vehicle.apply_control(control)

    finally:
        print(f"[INFO] Deteniendo grabación. Archivo guardado en: {log_path}")
        client.stop_recorder()
        
        try:
            if collision_sensor is not None:
                collision_sensor.destroy()
        except Exception:
            pass
        try:
            if camera is not None:
                camera.stop()
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