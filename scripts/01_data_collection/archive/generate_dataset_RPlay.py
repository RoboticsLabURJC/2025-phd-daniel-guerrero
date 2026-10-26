import os
import time
import csv
from queue import Queue, Empty

import cv2
import numpy as np
import carla

# ---------- Parámetros ----------
HOST = "127.0.0.1"
PORT = 2000

LOG_FILE_PATH = os.path.abspath("/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_crudos/recording_20260328_200646.log") 
OUT_DIR = "dataset_extraido"

IMG_W, IMG_H = 640, 360
CAM_FPS = 20
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
    # Reemplazamos los píxeles desde la altura 0 hasta la mitad (h//2) por negro (0)
    frame_bgr[0:h//2, :] = 0
    return frame_bgr

def main():
    if not os.path.exists(LOG_FILE_PATH):
        print(f"[ERROR] No se encontró el archivo log: {LOG_FILE_PATH}")
        return

    # Crear carpetas de salida
    os.makedirs(os.path.join(OUT_DIR, "rgb"), exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "driving_log.csv")

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)
    
    # 1. Obtener información del log (para saber la duración total)
    info = client.show_recorder_file_info(LOG_FILE_PATH, False)
    duration_str = [line for line in info.split('\n') if line.startswith('Duration:')]
    total_duration = float(duration_str[0].split()[1]) if duration_str else 0.0
    
    print(f"[INFO] Cargando Replay: {LOG_FILE_PATH}")
    print(f"[INFO] Duración del log: {total_duration} segundos.")

    # 2. Iniciar Replay
    client.replay_file(LOG_FILE_PATH, 0.0, 0.0, 0)
    world = client.get_world()

    # Configurar modo síncrono para extraer cuadro por cuadro perfectamente
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    # Esperar un par de ticks a que los actores del replay aparezcan en el mundo
    world.tick()
    world.tick()

    # 3. Buscar el vehículo Ego que el Replay acaba de spawnear
    vehicle = None
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') == 'ego':
            vehicle = actor
            break
            
    if vehicle is None:
        print("[ERROR] No se encontró el vehículo 'ego' en el replay.")
        # Restaurar mundo y salir
        settings.synchronous_mode = False
        world.apply_settings(settings)
        return

    print("[INFO] Vehículo Ego encontrado. Acoplando cámara...")

    # 4. Spawnea la cámara en el vehículo re-creado
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

    print(f"[INFO] Iniciando extracción de datos a: {OUT_DIR}...")
    
    # Preparar el archivo CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "image_path", "steer", "throttle", "brake"])

        step = 0
        elapsed_time = 0.0

        try:
            # 5. Bucle principal de extracción
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
                cv2.imshow("Extrayendo Dataset (Mitad Inferior)", frame_masked)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Extracción cancelada por el usuario.")
                    break

                # Guardar imagen en disco
                img_filename = f"rgb/{step:06d}.jpg"
                img_path = os.path.join(OUT_DIR, img_filename)
                cv2.imwrite(img_path, frame_masked, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                # Leer controles del vehículo en ese instante exacto
                control = vehicle.get_control()

                # Guardar telemetría en CSV
                writer.writerow([
                    step, 
                    img_filename, 
                    round(control.steer, 5), 
                    round(control.throttle, 5), 
                    round(control.brake, 5)
                ])
                
                step += 1
                
                if step % 50 == 0:
                    print(f"-> Extraídos {step} frames ({elapsed_time:.1f} / {total_duration:.1f} segs)")

            print(f"[INFO] Extracción completada con éxito. Total frames: {step}")

        finally:
            print("[INFO] Limpiando sensores y restaurando simulador...")
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
            if camera is not None:
                camera.stop()
                camera.destroy()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()