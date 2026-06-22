import os
import time
import csv
import glob
import math  # <-- Nueva librería para calcular la velocidad
from datetime import datetime
from queue import Queue, Empty

import cv2
import numpy as np
import carla

# ---------- Parámetros ----------
HOST = "127.0.0.1"
PORT = 2000

# Lee toda la carpeta de logs crudos
LOGS_DIR = os.path.abspath("/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_crudos") 

# Generación dinámica de la carpeta con la fecha y hora actual
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"dataset_extraido_{TIMESTAMP}"

IMG_W, IMG_H = 640, 360
CAM_FPS = 10  # <-- Reducido a 10 FPS (1 foto cada 100ms)
FIXED_DT = 1.0 / CAM_FPS
# -------------------------------

def to_bgr(image: carla.Image):
    """Convierte la imagen raw de CARLA a formato BGR de OpenCV."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()

def apply_half_mask(frame_bgr):
    """
    Enmascara la mitad superior de la imagen (el cielo) poniéndola en negro.
    Mantiene intacta la mitad inferior (la carretera).
    """
    h = frame_bgr.shape[0]
    frame_bgr[0:h//2, :] = 0
    return frame_bgr

def main():
    # 1. Buscar todos los archivos .log
    log_files = glob.glob(os.path.join(LOGS_DIR, "*.log"))
    if not log_files:
        print(f"[ERROR] No se encontraron archivos .log en la carpeta: {LOGS_DIR}")
        return

    print(f"[INFO] Se encontraron {len(log_files)} logs. Creando dataset masivo a 10 FPS (100ms)...")

    # Crear carpetas de salida con la fecha actual
    os.makedirs(os.path.join(OUT_DIR, "rgb"), exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "driving_log.csv")

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)
    
    # Contador global para que las imágenes de múltiples logs no se sobrescriban
    global_step = 0

    # Abrimos el CSV una sola vez para volcar todos los logs ahí
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # <-- Agregamos "speed_kmh" a la cabecera
        writer.writerow(["frame", "image_path", "steer", "throttle", "brake", "speed_kmh", "source_log"])

        # 2. Bucle principal para iterar sobre la carpeta
        for log_file in log_files:
            log_name = os.path.basename(log_file)
            
            # Obtener información del log
            info = client.show_recorder_file_info(log_file, False)
            duration_str = [line for line in info.split('\n') if line.startswith('Duration:')]
            total_duration = float(duration_str[0].split()[1]) if duration_str else 0.0
            
            print(f"\n[INFO] --- Cargando Replay: {log_name} ---")
            print(f"[INFO] Duración del log: {total_duration} segundos.")

            # Iniciar Replay
            client.replay_file(log_file, 0.0, 0.0, 0)
            world = client.get_world()

            # Configurar modo síncrono
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DT
            world.apply_settings(settings)

            # Esperar un par de ticks
            world.tick()
            world.tick()

            # Buscar el vehículo Ego
            vehicle = None
            for actor in world.get_actors().filter('vehicle.*'):
                if actor.attributes.get('role_name') == 'ego':
                    vehicle = actor
                    break
                    
            if vehicle is None:
                print(f"[ERROR] No se encontró el vehículo 'ego' en {log_name}. Saltando...")
                settings.synchronous_mode = False
                world.apply_settings(settings)
                continue

            print("[INFO] Vehículo Ego encontrado. Acoplando cámara...")

            # Spawnea la cámara
            bp_lib = world.get_blueprint_library()
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(IMG_W))
            cam_bp.set_attribute("image_size_y", str(IMG_H))
            cam_bp.set_attribute("fov", "90")
            cam_bp.set_attribute("sensor_tick", str(FIXED_DT))

            cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
            camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
            
            image_queue = Queue()
            camera.listen(image_queue.put)
            
            elapsed_time = 0.0
            local_step = 0

            try:
                # 3. Bucle de extracción exacto
                while elapsed_time < total_duration:
                    world.tick()
                    elapsed_time += FIXED_DT

                    try:
                        img = image_queue.get(timeout=2.0)
                    except Empty:
                        print("[ADVERTENCIA] No se recibió imagen en este tick.")
                        continue

                    # Procesar imagen
                    frame = to_bgr(img)
                    frame_masked = apply_half_mask(frame)

                    # Mostrar progreso visual
                    cv2.imshow("Extrayendo Dataset Masivo", frame_masked)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Extracción cancelada por el usuario.")
                        return # Sale de todo el script

                    # Guardar imagen usando el contador GLOBAL
                    img_filename = f"rgb/{global_step:06d}.jpg"
                    img_path = os.path.join(OUT_DIR, img_filename)
                    cv2.imwrite(img_path, frame_masked, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    # Leer controles y calcular la velocidad en km/h
                    control = vehicle.get_control()
                    vel = vehicle.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) # <-- Cálculo de velocidad

                    # Guardar telemetría
                    writer.writerow([
                        global_step, 
                        img_filename, 
                        round(control.steer, 5), 
                        round(control.throttle, 5), 
                        round(control.brake, 5),
                        round(speed_kmh, 2), # <-- Nueva columna en el CSV
                        log_name
                    ])
                    
                    global_step += 1
                    local_step += 1
                    
                    if local_step % 50 == 0:
                        print(f"-> Extraídos {local_step} frames ({elapsed_time:.1f} / {total_duration:.1f} segs) | Total global: {global_step} | Vel: {speed_kmh:.1f} km/h")

                print(f"[INFO] Log completado. Frames extraídos de este log: {local_step}")

            finally:
                # Limpiar sensores para poder cargar el siguiente log limpiamente
                settings.synchronous_mode = False
                world.apply_settings(settings)
                
                if camera is not None:
                    camera.stop()
                    camera.destroy()

    print(f"\n[ÉXITO] Procesamiento masivo completado. Total de frames globales a 10 FPS: {global_step}")
    print(f"[INFO] Dataset guardado en: {OUT_DIR}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()