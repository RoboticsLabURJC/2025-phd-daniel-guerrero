import carla
import time
import cv2
import numpy as np
import queue
import os
import csv
import re
import glob
from datetime import datetime

# Directorio donde están todos tus logs
LOGS_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/autopilot_dagger"

image_queue = queue.Queue()

def process_image(image):
    raw_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    raw_data = np.reshape(raw_data, (image.height, image.width, 4))
    image_rgb = raw_data[:, :, :3] 
    cam_location = image.transform.location
    image_queue.put((image.frame, image_rgb, cam_location))

def clear_queue(q):
    with q.mutex:
        q.queue.clear()

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "*.log")))
    if not log_files:
        print(f"[ERROR] No se encontraron archivos .log en {LOGS_DIR}")
        return
    
    print(f"[INFO] Se encontraron {len(log_files)} logs para procesar.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"dataset_masivo_{timestamp}"
    frames_dir = os.path.join(dataset_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    csv_path = os.path.join(dataset_dir, "driving_log.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_global', 'velocidad_kmh', 'throttle', 'steering', 'image_path', 'source_log'])
    
    print(f"[INFO] Dataset maestro creado en: {dataset_dir}/")

    imagenes_guardadas_global = 0
    frames_descartados_global = 0

    world = client.get_world()

    try:
        for log_idx, log_path in enumerate(log_files, 1):
            log_filename = os.path.basename(log_path)
            print(f"\n{'='*60}")
            print(f"[INFO] PROCESANDO LOG {log_idx}/{len(log_files)}: {log_filename}")
            print(f"{'='*60}")

            log_resumen = client.show_recorder_file_info(log_path, False)
            log_detalle = client.show_recorder_file_info(log_path, True)
            
            if "tesla.model3" not in log_detalle:
                print(f"[WARNING] Saltando {log_filename}: No contiene un Tesla Model 3.")
                continue

            match_dur = re.search(r'Duration:\s+([0-9.]+)', log_resumen)
            match_frm = re.search(r'Frames:\s+([0-9]+)', log_resumen)
            match_map = re.search(r'Map:\s+([^\n]+)', log_resumen)
            
            if not (match_dur and match_frm and match_map):
                print(f"[WARNING] Saltando {log_filename}: Datos corruptos o incompletos.")
                continue

            duracion_total = float(match_dur.group(1))
            frames_totales_grabados = int(match_frm.group(1))
            map_name_log = match_map.group(1).strip().split('/')[-1]

            # --- ESCUDO ANTI-CRASHEOS (Evita división por cero) ---
            if frames_totales_grabados <= 0 or duracion_total <= 0:
                print(f"[WARNING] Saltando {log_filename}: El log está vacío (0 segundos).")
                continue

            tiempo_por_frame = duracion_total / frames_totales_grabados 
            total_frames_esperados = int(duracion_total / 0.1)

            if total_frames_esperados <= 0:
                print(f"[WARNING] Saltando {log_filename}: El log es demasiado corto para extraer un solo frame útil.")
                continue
            # --------------------------------------------------------

            current_map = world.get_map().name.split('/')[-1]
            if current_map != map_name_log:
                print(f"[INFO] Cambio de mapa detectado: {current_map} -> {map_name_log}. Cargando mundo...")
                world = client.load_world(map_name_log)
                time.sleep(5) 
            else:
                print(f"[INFO] El servidor ya está en {map_name_log}. No es necesario recargar.")

            settings = world.get_settings()
            if settings.synchronous_mode:
                settings.synchronous_mode = False
                world.apply_settings(settings)

            client.stop_replayer(False) 
            clear_queue(image_queue)
            
            client.replay_file(log_path, 0, 0, 0)
            client.set_replayer_time_factor(20.0) 

            print(f"[INFO] Esperando vehículo en {map_name_log}...")
            ego_vehicle = None
            intentos = 0
            while ego_vehicle is None and intentos < 30:
                world = client.get_world() 
                vehicles = world.get_actors().filter('vehicle.*')
                for actor in vehicles:
                    if 'tesla.model3' in actor.type_id:
                        ego_vehicle = actor
                        break
                if not ego_vehicle and len(vehicles) > 0:
                    ego_vehicle = vehicles[0]
                if not ego_vehicle:
                    time.sleep(1)
                    intentos += 1

            if not ego_vehicle:
                print(f"[ERROR] Vehículo no encontrado en {log_filename}. Saltando al siguiente log.")
                continue

            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            camera_bp.set_attribute('sensor_tick', '0.1') 

            camera_transform = carla.Transform(carla.Location(x=0.4, y=-0.3, z=1.3))
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
            camera.listen(lambda image: process_image(image))

            print(f"[INFO] Extrayendo a x20 | Filtro: Descartar si Throttle==0 & Steering==0\n")
            
            prev_location = None
            prev_frame_id = None
            frames_procesados_local = 0
            descartados_local = 0

            while True:
                if not image_queue.empty():
                    frame_id, image_rgb, current_location = image_queue.get()
                    
                    velocidad_kmh = 0.0
                    if prev_location is not None and prev_frame_id is not None:
                        frames_avanzados = frame_id - prev_frame_id
                        if frames_avanzados > 0:
                            dt_simulado = frames_avanzados * tiempo_por_frame 
                            distancia = current_location.distance(prev_location)
                            velocidad_kmh = (distancia / dt_simulado) * 3.6
                    
                    prev_location = current_location
                    prev_frame_id = frame_id
                    
                    control = ego_vehicle.get_control()
                    
                    if abs(control.throttle) < 0.001 and abs(control.steer) < 0.001:
                        frames_descartados_global += 1
                        descartados_local += 1
                        frames_procesados_local += 1
                    else:
                        img_filename = f"{imagenes_guardadas_global:08d}.png" 
                        img_filepath = os.path.join(frames_dir, img_filename)
                        cv2.imwrite(img_filepath, image_rgb)
                        
                        rel_img_path = os.path.join("frames", img_filename)
                        csv_writer.writerow([
                            imagenes_guardadas_global, 
                            round(velocidad_kmh, 2), 
                            round(control.throttle, 3), 
                            round(control.steer, 3), 
                            rel_img_path,
                            log_filename 
                        ])
                        
                        imagenes_guardadas_global += 1
                        frames_procesados_local += 1

                    porcentaje = min((frames_procesados_local / total_frames_esperados) * 100, 100)
                    longitud_barra = 30
                    llenos = int((longitud_barra * frames_procesados_local) // total_frames_esperados)
                    barra = '█' * llenos + '-' * (longitud_barra - llenos)
                    
                    print(f"\rProgreso Log: |{barra}| {porcentaje:.1f}% | Guardados: {imagenes_guardadas_global} | Descartados (Log): {descartados_local}  ", end='', flush=True)

                    if frames_procesados_local >= total_frames_esperados:
                        print(f"\n[INFO] {log_filename} finalizado.")
                        break

            # Limpieza local CORREGIDA sin paréntesis en is_alive
            if 'camera' in locals() and camera.is_alive:
                camera.stop()
                camera.destroy()
            client.stop_replayer(False)

        print(f"\n{'='*60}")
        print(f"[ÉXITO TOTAL] Se procesaron {len(log_files)} logs.")
        print(f"Dataset Maestro generado con {imagenes_guardadas_global} imágenes útiles.")
        print(f"Frames basura descartados en total: {frames_descartados_global}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print(f"\n\n[INFO] Extracción interrumpida masivamente por el usuario.")
        print(f"Imágenes rescatadas hasta ahora: {imagenes_guardadas_global}")
    finally:
        print("[INFO] Cerrando puertos y guardando dataset de forma segura...")
        client.stop_replayer(False) 
        
        if 'csv_file' in locals() and not csv_file.closed:
            csv_file.close()
            
        # Limpieza global CORREGIDA sin paréntesis en is_alive
        if 'camera' in locals() and camera.is_alive:
            camera.stop()
            camera.destroy()
            
        print("[INFO] Pipeline maestro cerrado.")

if __name__ == '__main__':
    main()