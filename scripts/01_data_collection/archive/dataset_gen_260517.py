import carla
import time
import math
import os
import csv
import glob
from queue import Queue, Empty
from datetime import datetime

import cv2
import numpy as np

# ---------------- Config ----------------
HOST = "127.0.0.1"
PORT = 2000

LOGS_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_dagger"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f"dataset_replay_{TIMESTAMP}"
IMG_DIR = os.path.join(SAVE_PATH, "images")
CSV_PATH = os.path.join(SAVE_PATH, "driving_log.csv")
os.makedirs(IMG_DIR, exist_ok=True)

IMG_W, IMG_H = 800, 600
FOV = 90
FPS = 10         
DT = 1.0 / FPS
# ---------------------------------------

def bgr_from_carla_image(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]

def pick_vehicle(world: carla.World):
    vehicles = world.get_actors().filter("vehicle.*")
    if len(vehicles) == 0:
        return None

    for name in ("ego", "hero"):
        for v in vehicles:
            if v.attributes.get("role_name", "") == name:
                return v

    return vehicles[0]

def main():
    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "*.log")))
    if not log_files:
        print(f"[ERROR] No se encontraron archivos .log en {LOGS_DIR}")
        return

    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)

    world = client.get_world()
    original_settings = world.get_settings()

    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "filename", "frame", "timestamp_s",
        "steer", "throttle", "brake", "speed_kmh", 
        "v_forward_mps", "yaw_rate_rps", "source_log"
    ])
    csv_file.flush()

    cv2.namedWindow("REPLAY CAM", cv2.WINDOW_AUTOSIZE)
    
    frame_id = 0
    abort = False

    try:
        for log_file in log_files:
            if abort:
                break
                
            log_name = os.path.basename(log_file)
            print(f"\n[INFO] --- Procesando log: {log_name} ---")

            info = client.show_recorder_file_info(log_file, False)
            duration_str = [line for line in info.split('\n') if line.startswith('Duration:')]
            total_duration = float(duration_str[0].split()[1]) if duration_str else 0.0

            map_str = [line for line in info.split('\n') if line.startswith('Map:')]
            target_map = map_str[0].split()[1] if map_str else None

            if target_map:
                current_map = world.get_map().name
                if target_map.split('/')[-1].lower() not in current_map.lower():
                    print(f"[INFO] El log requiere {target_map}. Cambiando mapa (esto tomará unos segundos)...")
                    
                    settings = world.get_settings()
                    settings.synchronous_mode = False
                    world.apply_settings(settings)
                    
                    client.load_world(target_map)
                    time.sleep(4.0) 
                    world = client.get_world() 

            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = DT
            settings.no_rendering_mode = False
            world.apply_settings(settings)

            client.replay_file(log_file, 0.0, 0.0, 0)

            for _ in range(20):
                world.tick()

            ego = pick_vehicle(world)
            if ego is None:
                print(f"[ERROR] No apareció ningún vehículo en {log_name}. Saltando log...")
                client.stop_replayer(True)
                
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
                continue

            print(f"[OK] Vehículo listo. Extrayendo {total_duration:.1f} segundos a 10 FPS...")

            bp_lib = world.get_blueprint_library()
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(IMG_W))
            cam_bp.set_attribute("image_size_y", str(IMG_H))
            cam_bp.set_attribute("fov", str(FOV))
            cam_bp.set_attribute("sensor_tick", str(DT))

            camera = world.spawn_actor(
                cam_bp,
                carla.Transform(carla.Location(x=0.8, z=1.3)),
                attach_to=ego
            )

            image_q = Queue()
            camera.listen(image_q.put)

            elapsed_time = 0.0

            while elapsed_time < total_duration:
                world.tick()
                elapsed_time += DT

                # --- LA CORRECCIÓN CLAVE ---
                try:
                    # Espera bloqueante: Obligamos a Python a esperar la foto de este tick
                    img = image_q.get(timeout=2.0)
                except Empty:
                    # Si de verdad tarda más de 2 segundos, lo registramos pero no colapsamos
                    continue

                # Vaciamos la cola por si llegaron frames atrasados, quedándonos siempre con el último
                while not image_q.empty():
                    img = image_q.get_nowait()
                # ---------------------------

                bgr = bgr_from_carla_image(img)

                cv2.imshow("REPLAY CAM", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Cancelado por el usuario.")
                    abort = True
                    break

                fname = f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(os.path.join(IMG_DIR, fname), bgr)

                control = ego.get_control()
                vel = ego.get_velocity()
                ang = ego.get_angular_velocity()
                tf = ego.get_transform()
                fwd = tf.get_forward_vector()

                speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) 
                v_fwd = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
                yaw_rate = ang.z

                writer.writerow([
                    fname,
                    int(img.frame),
                    round(float(img.timestamp), 3),
                    round(control.steer, 5), 
                    round(control.throttle, 5), 
                    round(control.brake, 5),
                    round(speed_kmh, 2),
                    round(v_fwd, 4),
                    round(yaw_rate, 5),
                    log_name
                ])

                if frame_id % 20 == 0:
                    csv_file.flush()

                frame_id += 1
            
            camera.stop()
            camera.destroy()
            client.stop_replayer(True)
            
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
            time.sleep(1.0) 

    finally:
        print("\n[INFO] Finalizando y limpiando...")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
        except Exception:
            pass

        try:
            original_settings.synchronous_mode = False
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("[OK] Dataset generado exitosamente en orden temporal:")
        print(" - Directorio Raíz:", SAVE_PATH)
        print(" - Imágenes:", IMG_DIR)
        print(" - Archivo CSV:", CSV_PATH)

if __name__ == "__main__":
    main()