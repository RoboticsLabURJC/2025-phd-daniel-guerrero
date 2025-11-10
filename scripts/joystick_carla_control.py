# control_carla_simple_axes.py
import time
from queue import Queue, Empty

import cv2
import numpy as np
import pygame

import carla

# ---------- Parámetros ----------
HOST = "127.0.0.1"
PORT = 2000

IMG_W, IMG_H = 1280, 920   # Baja resolución = más FPS
CAM_FPS = 20              # Sincroniza sim + sensor a este FPS

STEER_RATE = 2.5          # Velocidad de cambio de dirección con teclado
THROTTLE_STEP = 0.02
BRAKE_STEP = 0.05

# --- Mapeo de tu control ---
AXIS_STEER = 1    # volante: -1 (izq) ... +1 (der)
AXIS_THROTTLE = 5 # acelerador:  +1 (sin acelerar) ... -1 (a fondo)
AXIS_BRAKE = 4    # freno:       +1 (sin frenar)  ... -1 (a fondo)

DEADZONE_STEER = 0.02  # deadzone para volante
DEADZONE_PEDAL = 0.01  # deadzone para pedales
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
    # Deadzone cerca de 1 (sin pisar)
    if v > 1.0 - deadzone:
        v = 1.0
    # Mapeo lineal invertido
    return (1.0 - v) / 2.0

def axis_with_deadzone(v: float, dz: float) -> float:
    v = float(np.clip(v, -1.0, 1.0))
    return 0.0 if abs(v) < dz else v

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(5.0)

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
        vehicle = None
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

        # Ventana OpenCV
        cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)

        # Variables de control
        steer = 0.0
        throttle = 0.0
        brake = 0.0
        reverse = False
        hand_brake = False
        control = carla.VehicleControl()

        # Primer tick para estabilizar
        world.tick()

        running = True
        while running:
            # Avanza simulación (sincrónico)
            world.tick()

            # Leer imagen más reciente disponible
            img = None
            while True:
                try:
                    img = image_queue.get_nowait()
                except Empty:
                    break
            if img is not None:
                frame = to_bgr(img)
                cv2.imshow("CAM", frame)
                cv2.waitKey(1)

            # Eventos pygame (cierre ventana, etc.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # --- Teclado (fallback) ---
            if joystick is None:
                # Dirección
                if keys[pygame.K_a]:
                    steer -= STEER_RATE * settings.fixed_delta_seconds
                elif keys[pygame.K_d]:
                    steer += STEER_RATE * settings.fixed_delta_seconds
                else:
                    steer *= 0.9  # autocentrado

                # Acelerador / freno
                if keys[pygame.K_w]:
                    throttle = min(1.0, throttle + THROTTLE_STEP); brake = 0.0
                elif keys[pygame.K_s]:
                    brake = min(1.0, brake + BRAKE_STEP); throttle = max(0.0, throttle - THROTTLE_STEP)
                else:
                    throttle *= 0.98
                    brake *= 0.9

                if keys[pygame.K_SPACE]:
                    brake = 1.0
                if keys[pygame.K_q]:
                    running = False
                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

            # --- Volante/joystick (tu mapeo) ---
            else:
                # Eje 0: volante -1..1
                try:
                    steer_val = -joystick.get_axis(AXIS_STEER)
                except Exception:
                    steer_val = 0.0
                steer = axis_with_deadzone(steer_val, DEADZONE_STEER)

                # Eje 1 (acelerador): 1..-1  -> 0..1
                try:
                    accel_val = joystick.get_axis(AXIS_THROTTLE)
                except Exception:
                    accel_val = 1.0  # sin acelerar
                throttle = float(np.clip(pedal_inverted_to_01(accel_val, DEADZONE_PEDAL), 0.0, 1.0))

                # Eje 2 (freno): 1..-1 -> 0..1
                try:
                    brake_val = joystick.get_axis(AXIS_BRAKE)
                except Exception:
                    brake_val = 1.0  # sin frenar
                brake = float(np.clip(pedal_inverted_to_01(brake_val, DEADZONE_PEDAL), 0.0, 1.0))

                # Reversa (opcional): sigue usando tecla R para alternar
                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

                # Cerrar con Q
                if keys[pygame.K_q]:
                    running = False

            # Limitar
            steer = float(np.clip(steer, 1.0, -1.0))
            throttle = float(np.clip(throttle, 0.0, 1.0))
            brake = float(np.clip(brake, 0.0, 1.0))

            # Aplicar control
            control.steer = steer
            control.throttle = throttle
            control.brake = brake
            control.hand_brake = hand_brake
            control.reverse = reverse
            vehicle.apply_control(control)

        cv2.destroyAllWindows()

    finally:
        # Limpieza segura
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        if camera is not None:
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()

        # Restaurar ajustes del mundo
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        pygame.quit()

if __name__ == "__main__":
    main()