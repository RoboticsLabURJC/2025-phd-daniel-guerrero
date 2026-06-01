import os
import time
from datetime import datetime
from queue import Queue, Empty

import cv2
import numpy as np
import pygame
import carla

# ---------- Parámetros de Configuración ----------
HOST = "127.0.0.1"
PORT = 2000

# CAMBIO 1: Define aquí el mapa (Town) que deseas cargar por defecto
# Mapas comunes: Town01, Town02, Town03 (urbano), Town04 (autopista), Town07 (rural)
TARGET_TOWN = "Town02_Opt" 

IMG_W, IMG_H = 640, 360
CAM_FPS = 20
FIXED_DT = 1.0 / CAM_FPS

# CONFIGURACIÓN DE EJES JOYSTICK
AXIS_STEER = 0      # Eje horizontal
AXIS_THROTTLE = 4   # Eje acelerador (-1 a 1)
BUTTON_BRAKE = 0    # Botón para frenar (X/A)

# PARÁMETROS DE DINÁMICA Y SUAVEZ
DEADZONE_STEER = 0.05
DEADZONE_PEDAL = 0.02

# EMA_ALPHA: Menos de 0.15 hace que el volante se sienta "pesado" y fluido
EMA_ALPHA_STEER = 0.12    
EMA_ALPHA_THROTTLE = 0.20
EMA_ALPHA_BRAKE = 0.25

# AJUSTE DE GIRO (Aquí controlamos la intensidad)
STEER_EXPONENT = 3   # Suavidad en el centro
STEER_LIMITER = 0.4  # LIMITADOR: 0.4 significa que el volante solo girará hasta el 40%
THROTTLE_SCALE = 0.8 # Limita la aceleración máxima

# CONFIGURACIÓN DAGGER / PERTURBACIÓN
PERTURB_INTERVAL = 10.0 
PERTURB_DISTANCE = 1.3  
MIN_SPEED_PERTURB = 5.0 
# ------------------------------------------------

def to_bgr(image: carla.Image):
    """Convierte la imagen de CARLA a BGR y crea una copia para poder editarla."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()

def joystick_axis_to_01(v: float) -> float:
    return float(np.clip((v + 1.0) / 2.0, 0.0, 1.0))

def axis_with_deadzone(v: float, dz: float) -> float:
    v = float(np.clip(v, -1.0, 1.0))
    return 0.0 if abs(v) < dz else v

def ema(prev: float, cur: float, alpha: float) -> float:
    return float(alpha * cur + (1.0 - alpha) * prev)

def apply_perturbation(vehicle, distance):
    """Desplaza el vehículo lateralmente para forzar recuperación."""
    transform = vehicle.get_transform()
    location = transform.location
    side_vector = transform.get_right_vector()
    direction = 1 if np.random.random() > 0.5 else -1
    
    location.x += side_vector.x * distance * direction
    location.y += side_vector.y * distance * direction
    location.z += 0.15 
    
    vehicle.set_transform(carla.Transform(location, transform.rotation))
    print(f"\n[!!!] SALTO: {distance}m a la {'derecha' if direction > 0 else 'izquierda'}")

def get_image_for_snapshot(image_queue: Queue, target_frame: int, timeout=2.0):
    t0 = time.time()
    last = None
    while True:
        remaining = max(0.01, timeout - (time.time() - t0))
        try:
            img = image_queue.get(timeout=remaining)
            last = img
            if img.frame >= target_frame: return img
        except Empty: return last

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    
    # --- CAMBIO 1: LOGICA DE CAMBIO DE TOWN ---
    world = client.get_world()
    current_map = world.get_map().name.split('/')[-1]
    
    if current_map != TARGET_TOWN:
        print(f"[INFO] Cambiando de {current_map} a {TARGET_TOWN}...")
        world = client.load_world(TARGET_TOWN)
    else:
        print(f"[INFO] Permaneciendo en {TARGET_TOWN}")

    # Configuración de modo sincrónico
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    pygame.init()
    display = pygame.display.set_mode((IMG_W, IMG_H))
    pygame.joystick.init()
    
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"[INFO] Joystick: {joystick.get_name()}")

    bp_lib = world.get_blueprint_library()
    vehicle = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), 
                                world.get_map().get_spawn_points()[0])
    
    # --- CAMBIO 2: CONFIGURACIÓN CÁMARA EN TERCERA PERSONA ---
    # Colocamos la cámara detrás (x = -5.5), arriba (z = 2.5) y apuntando ligeramente hacia abajo (pitch = -15)
    camera_transform = carla.Transform(
        carla.Location(x=-5.5, z=2.5), 
        carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)
    )
    camera = world.spawn_actor(bp_lib.find("sensor.camera.rgb"), camera_transform, attach_to=vehicle)
    
    image_queue = Queue()
    camera.listen(image_queue.put)

    steer_f = throttle_f = brake_f = 0.0
    last_perturb_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.abspath(f"dagger_log_{run_id}.log")
    
    client.start_recorder(log_path, True)
    print(f"[OK] Grabando en: {log_path}")

    running = True
    try:
        while running:
            current_time = time.time()
            world.tick()
            snap = world.get_snapshot()
            
            # --- LÓGICA DE PERTURBACIÓN ---
            v_vec = vehicle.get_velocity()
            speed_kmh = 3.6 * np.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            
            if (current_time - last_perturb_time) >= PERTURB_INTERVAL:
                if speed_kmh > MIN_SPEED_PERTURB:
                    apply_perturbation(vehicle, PERTURB_DISTANCE)
                    last_perturb_time = current_time
                else:
                    last_perturb_time = current_time - (PERTURB_INTERVAL - 2.0)

            # --- PROCESAMIENTO DE IMAGEN ---
            img = get_image_for_snapshot(image_queue, snap.frame)
            if img is not None:
                frame = to_bgr(img)
                cv2.putText(frame, f"Speed: {int(speed_kmh)} km/h", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Next Jump: {int(PERTURB_INTERVAL - (current_time - last_perturb_time))}s", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
                display.blit(surface, (0, 0))
                pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]: running = False

            if joystick:
                raw_steer = joystick.get_axis(AXIS_STEER)
                steer_val = axis_with_deadzone(raw_steer, DEADZONE_STEER)
                
                steer = ((abs(steer_val) ** STEER_EXPONENT) * (1.0 if steer_val > 0 else -1.0)) * STEER_LIMITER
                throttle = joystick_axis_to_01(joystick.get_axis(AXIS_THROTTLE)) * THROTTLE_SCALE
                brake = 1.0 if (joystick.get_button(BUTTON_BRAKE) or keys[pygame.K_SPACE]) else 0.0
            else:
                steer = -0.5 if keys[pygame.K_a] else 0.5 if keys[pygame.K_d] else 0.0
                throttle = 0.5 if keys[pygame.K_w] else 0.0
                brake = 1.0 if keys[pygame.K_s] else 0.0

            # Suavizado EMA
            steer_f = ema(steer_f, steer, EMA_ALPHA_STEER)
            throttle_f = ema(throttle_f, throttle, EMA_ALPHA_THROTTLE)
            brake_f = ema(brake_f, brake, EMA_ALPHA_BRAKE)

            vehicle.apply_control(carla.VehicleControl(
                steer=steer_f, throttle=throttle_f, brake=brake_f, reverse=keys[pygame.K_r]
            ))

    finally:
        client.stop_recorder()
        world.apply_settings(original_settings)
        if 'camera' in locals(): camera.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()