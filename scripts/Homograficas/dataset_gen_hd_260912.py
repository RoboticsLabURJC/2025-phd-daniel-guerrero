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

LOGS_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_dagger_expert"
DATASET_DIR = "dataset_masivo_continuo" 

image_queue = queue.Queue()

def process_image(image):
    raw_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    raw_data = np.reshape(raw_data, (image.height, image.width, 4))
    image_bgr = raw_data[:, :, :3] 
    cam_location = image.transform.location
    image_queue.put((image.frame, image_bgr, cam_location, image.timestamp))

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
    
    frames_dir = os.path.join(DATASET_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    csv_path = os.path.join(DATASET_DIR, "driving_log.csv")

    logs_procesados = set()
    imagenes_guardadas_global = 0
    
    if os.path.exists(csv_path):
        print(f"[INFO] Dataset existente detectado. Reanudando progreso...")
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                imagenes_guardadas_global += 1
                if len(row) >= 6:
                    logs_procesados.add(row[5])
        
        csv_file = open(csv_path, mode='a', newline='')
        csv_writer = csv.writer(csv_file)
        print(f"[INFO] Saltando {len(logs_procesados)} logs ya completados.")
        print(f"[INFO] Imágenes previas rescatadas: {imagenes_guardadas_global}")
    else:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_global', 'velocidad_kmh', 'throttle', 'steering', 'image_path', 'source_log'])

    frames_descartados_global = 0

    try:
        for log_idx, log_path in enumerate(log_files, 1):
            log_filename = os.path.basename(log_path)
            
            if log_filename in logs_procesados:
                continue

            intentos_reconexion = 0
            exito_en_log = False

            while intentos_reconexion < 10 and not exito_en_log:
                try:
                    print(f"\n{'='*60}")
                    print(f"[INFO] PROCESANDO LOG {log_idx}/{len(log_files)}: {log_filename} (Intento {intentos_reconexion + 1})")
                    print(f"{'='*60}")
                    
                    world = client.get_world()

                    log_resumen = client.show_recorder_file_info(log_path, False)
                    log_detalle = client.show_recorder_file_info(log_path, True)
                    
                    if "tesla.model3" not in log_detalle:
                        print(f"[WARNING] Saltando {log_filename}: No contiene un Tesla Model 3.")
                        exito_en_log = True 
                        break

                    match_dur = re.search(r'Duration:\s+([0-9.]+)', log_resumen)
                    match_frm = re.search(r'Frames:\s+([0-9]+)', log_resumen)
                    match_map = re.search(r'Map:\s+([^\n]+)', log_resumen)
                    
                    if not (match_dur and match_frm and match_map):
                        print(f"[WARNING] Saltando {log_filename}: Datos corruptos.")
                        exito_en_log = True
                        break

                    duracion_total = float(match_dur.group(1))
                    frames_totales_grabados = int(match_frm.group(1))
                    map_name_log = match_map.group(1).strip().split('/')[-1]

                    if duracion_total <= 0:
                        print(f"[WARNING] Saltando {log_filename}: Log vacío.")
                        exito_en_log = True
                        break

                    tiempo_por_frame = duracion_total / frames_totales_grabados 
                    
                    current_map = world.get_map().name.split('/')[-1]
                    if current_map != map_name_log:
                        print(f"[INFO] Cargando mapa: {map_name_log}...")
                        world = client.load_world(map_name_log)
                        time.sleep(3) 

                    settings = world.get_settings()
                    if settings.synchronous_mode:
                        settings.synchronous_mode = False
                        world.apply_settings(settings)

                    client.stop_replayer(False) 
                    clear_queue(image_queue)
                    
                    client.replay_file(log_path, 0, 0, 0)
                    client.set_replayer_time_factor(1.0) 

                    print(f"[INFO] Esperando vehículo...")
                    ego_vehicle = None
                    intentos_vehiculo = 0
                    while ego_vehicle is None and intentos_vehiculo < 30:
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
                            intentos_vehiculo += 1

                    if not ego_vehicle:
                        print(f"[ERROR] Vehículo no encontrado. Saltando log.")
                        exito_en_log = True
                        break

                    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
                    camera_bp.set_attribute('image_size_x', '1920')
                    camera_bp.set_attribute('image_size_y', '1080')
                    camera_bp.set_attribute('fov', '110')
                    camera_bp.set_attribute('sensor_tick', '0.1') 

                    camera_transform = carla.Transform(
                        carla.Location(x=2.0, y=0.0, z=1.6), 
                        carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)
                    )
                    
                    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
                    camera.listen(lambda image: process_image(image))

                    prev_location = None
                    prev_frame_id = None
                    descartados_local = 0
                    start_sim_time = None 
                    
                    consecutive_timeouts = 0 

                    while True:
                        try:
                            frame_id, image_bgr, current_location, frame_timestamp = image_queue.get(timeout=0.1)
                            consecutive_timeouts = 0 
                            
                            if start_sim_time is None:
                                start_sim_time = frame_timestamp
                                
                            if (frame_timestamp - start_sim_time) >= duracion_total:
                                print(f"\n[INFO] {log_filename} finalizado con éxito.")
                                exito_en_log = True
                                logs_procesados.add(log_filename)
                                break

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
                            else:
                                img_filename = f"{imagenes_guardadas_global:08d}.png" 
                                img_filepath = os.path.join(frames_dir, img_filename)
                                cv2.imwrite(img_filepath, image_bgr)
                                
                                rel_img_path = os.path.join("frames", img_filename)
                                csv_writer.writerow([
                                    imagenes_guardadas_global, 
                                    round(velocidad_kmh, 2), 
                                    round(control.throttle, 3), 
                                    round(control.steer, 3), 
                                    rel_img_path,
                                    log_filename 
                                ])
                                csv_file.flush() 
                                
                                imagenes_guardadas_global += 1

                            porcentaje = min(((frame_timestamp - start_sim_time) / duracion_total) * 100, 100)
                            llenos = int((30 * porcentaje) // 100)
                            barra = '█' * llenos + '-' * (30 - llenos)
                            print(f"\rProgreso: |{barra}| {porcentaje:.1f}% | Total Guardado: {imagenes_guardadas_global} ", end='', flush=True)

                        except queue.Empty:
                            consecutive_timeouts += 1
                            if consecutive_timeouts > 50: 
                                print(f"\n[WARNING] Servidor congelado o log vacío (5s sin frames). Forzando cierre...")
                                break

                    if 'camera' in locals() and camera is not None and camera.is_alive:
                        try:
                            camera.stop()
                            camera.destroy()
                        except: pass
                    client.stop_replayer(False)

                except RuntimeError as e:
                    print(f"\n[CRÍTICO] Conexión perdida con CARLA (Segmentation Fault probable): {e}")
                    print(f"[INFO] Esperando 10 segundos para que el Watchdog levante el servidor...")
                    time.sleep(10)
                    client = carla.Client('localhost', 2000)
                    client.set_timeout(60.0)
                    intentos_reconexion += 1

            if not exito_en_log:
                print(f"\n[ERROR] Imposible procesar {log_filename} tras múltiples intentos. Saltando de forma definitiva.")
                logs_procesados.add(log_filename)

        print(f"\n{'='*60}")
        print(f"[ÉXITO TOTAL] Extracción finalizada.")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print(f"\n\n[INFO] Detenido por el usuario.")
    finally:
        if 'csv_file' in locals() and not csv_file.closed:
            csv_file.close()

if __name__ == '__main__':
    main()