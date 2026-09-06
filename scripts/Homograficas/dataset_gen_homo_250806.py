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

image_queue = queue.Queue()

def process_image(image, camera_name):
    raw_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    raw_data = np.reshape(raw_data, (image.height, image.width, 4))
    image_bgr = raw_data[:, :, :3] 
    cam_location = image.transform.location
    # Añadimos camera_name a la tupla
    image_queue.put((image.frame, camera_name, image_bgr, cam_location, image.timestamp))

def clear_queue(q):
    with q.mutex:
        q.queue.clear()

# --- FUNCIONES DE HOMOGRAFÍA (BEV) ---
def get_bev_matrices():
    """
    Define los puntos de origen (cámara) y destino (lienzo BEV) para calcular 
    las matrices de transformación de perspectiva.
    Tendrá que ajustarse empíricamente según la altura (z) y el pitch de las cámaras.
    """
    canvas_size = 800
    cam_w, cam_h = 800, 600
    
    # Puntos origen en la imagen de la cámara (Trapecio)
    # [Abajo-Izq, Abajo-Der, Arriba-Izq, Arriba-Der]
    src_pts = np.float32([
        [0, cam_h], [cam_w, cam_h], 
        [0, cam_h//2], [cam_w, cam_h//2]
    ])
    
    matrices = {}
    
    # Puntos destino en el lienzo BEV (Rectángulos)
    # Asumimos que el coche está en el centro del lienzo (400, 400)
    
    # Frontal (Ocupa la parte superior central del lienzo)
    dst_front = np.float32([[250, 400], [550, 400], [250, 0], [550, 0]])
    matrices['front'] = cv2.getPerspectiveTransform(src_pts, dst_front)
    
    # Trasera (Ocupa la parte inferior central)
    dst_rear = np.float32([[550, 400], [250, 400], [550, 800], [250, 800]])
    matrices['rear'] = cv2.getPerspectiveTransform(src_pts, dst_rear)
    
    # Izquierda (Ocupa la parte izquierda central)
    dst_left = np.float32([[400, 550], [400, 250], [0, 550], [0, 250]])
    matrices['left'] = cv2.getPerspectiveTransform(src_pts, dst_left)
    
    # Derecha (Ocupa la parte derecha central)
    dst_right = np.float32([[400, 250], [400, 550], [800, 250], [800, 550]])
    matrices['right'] = cv2.getPerspectiveTransform(src_pts, dst_right)
    
    return matrices, canvas_size

def stitch_bev(images_dict, matrices, canvas_size):
    """
    Fusiona las 4 imágenes en un solo lienzo BEV usando sus matrices.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    for cam_name, img in images_dict.items():
        if cam_name in matrices:
            # Proyectar la imagen
            warped = cv2.warpPerspective(img, matrices[cam_name], (canvas_size, canvas_size))
            # Combinar manteniendo los píxeles más brillantes (solapamiento simple)
            canvas = cv2.max(canvas, warped)
            
    # Dibujar un rectángulo que represente el Ego Vehicle (Tesla Model 3) en el centro
    cv2.rectangle(canvas, (360, 320), (440, 480), (0, 0, 255), -1) 
    return canvas

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "*.log")))
    if not log_files:
        print(f"[ERROR] No se encontraron archivos .log en {LOGS_DIR}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"dataset_bev_{timestamp}"
    frames_dir = os.path.join(dataset_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    csv_path = os.path.join(dataset_dir, "driving_log.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_global', 'velocidad_kmh', 'throttle', 'steering', 'image_path', 'source_log'])
    
    matrices_bev, bev_size = get_bev_matrices()
    imagenes_guardadas_global = 0
    frames_descartados_global = 0
    world = client.get_world()

    try:
        for log_idx, log_path in enumerate(log_files, 1):
            log_filename = os.path.basename(log_path)
            print(f"\n{'='*60}")
            print(f"[INFO] PROCESANDO LOG {log_idx}/{len(log_files)}: {log_filename}")

            log_resumen = client.show_recorder_file_info(log_path, False)
            match_dur = re.search(r'Duration:\s+([0-9.]+)', log_resumen)
            match_frm = re.search(r'Frames:\s+([0-9]+)', log_resumen)
            match_map = re.search(r'Map:\s+([^\n]+)', log_resumen)
            
            if not (match_dur and match_frm and match_map):
                continue

            duracion_total = float(match_dur.group(1))
            map_name_log = match_map.group(1).strip().split('/')[-1]
            tiempo_por_frame = duracion_total / int(match_frm.group(1))

            current_map = world.get_map().name.split('/')[-1]
            if current_map != map_name_log:
                world = client.load_world(map_name_log)
                time.sleep(5) 

            client.stop_replayer(False) 
            clear_queue(image_queue)
            client.replay_file(log_path, 0, 0, 0)
            client.set_replayer_time_factor(1.0) 

            ego_vehicle = None
            intentos = 0
            while ego_vehicle is None and intentos < 30:
                world = client.get_world() 
                for actor in world.get_actors().filter('vehicle.*'):
                    if 'tesla.model3' in actor.type_id:
                        ego_vehicle = actor
                        break
                if not ego_vehicle:
                    time.sleep(1)
                    intentos += 1

            if not ego_vehicle:
                continue

            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '110') # FOV ampliado para asegurar solapamiento
            camera_bp.set_attribute('sensor_tick', '0.1') 

            # Configuración del rig de 4 cámaras (Ajustadas para el techo/cofre de un Model 3)
            transforms = {
                'front': carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
                'rear':  carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(pitch=-15, yaw=180)),
                'left':  carla.Transform(carla.Location(y=-1.0, z=2.4), carla.Rotation(pitch=-15, yaw=-90)),
                'right': carla.Transform(carla.Location(y=1.0, z=2.4), carla.Rotation(pitch=-15, yaw=90))
            }

            active_cameras = []
            for cam_name, trans in transforms.items():
                cam = world.spawn_actor(camera_bp, trans, attach_to=ego_vehicle)
                cam.listen(lambda image, n=cam_name: process_image(image, n))
                active_cameras.append(cam)

            prev_location = None
            prev_frame_id = None
            start_sim_time = None 
            
            # Buffer para sincronizar frames asíncronos
            frame_buffer = {}

            while True:
                try:
                    frame_id, cam_name, image_bgr, current_location, frame_timestamp = image_queue.get(timeout=0.1)
                    
                    if start_sim_time is None:
                        start_sim_time = frame_timestamp
                        
                    if (frame_timestamp - start_sim_time) >= duracion_total:
                        break

                    # Almacenar en el buffer por ID de frame
                    if frame_id not in frame_buffer:
                        frame_buffer[frame_id] = {'images': {}, 'loc': current_location, 'ts': frame_timestamp}
                    
                    frame_buffer[frame_id]['images'][cam_name] = image_bgr

                    # Si tenemos las 4 imágenes de este frame exacto, procesamos el BEV
                    if len(frame_buffer[frame_id]['images']) == 4:
                        bev_image = stitch_bev(frame_buffer[frame_id]['images'], matrices_bev, bev_size)
                        
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
                        
                        cv2.imshow("BEV Combinado en Vivo", bev_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        if abs(control.throttle) < 0.001 and abs(control.steer) < 0.001:
                            frames_descartados_global += 1
                        else:
                            img_filename = f"{imagenes_guardadas_global:08d}.png" 
                            img_filepath = os.path.join(frames_dir, img_filename)
                            cv2.imwrite(img_filepath, bev_image)
                            
                            csv_writer.writerow([
                                imagenes_guardadas_global, round(velocidad_kmh, 2), 
                                round(control.throttle, 3), round(control.steer, 3), 
                                os.path.join("frames", img_filename), log_filename 
                            ])
                            imagenes_guardadas_global += 1

                        # Limpieza de memoria (borrar frames viejos incompletos para no saturar la RAM)
                        old_frames = [f for f in frame_buffer.keys() if f < frame_id - 5]
                        for f in old_frames:
                            del frame_buffer[f]

                except queue.Empty:
                    continue

            # Detener cámaras al cambiar de log
            for cam in active_cameras:
                if cam.is_alive:
                    cam.stop()
                    cam.destroy()
            active_cameras.clear()
            
            client.stop_replayer(False)

        print(f"\n[ÉXITO] Dataset BEV generado con {imagenes_guardadas_global} imágenes.")

    finally:
        cv2.destroyAllWindows()
        client.stop_replayer(False) 
        if 'csv_file' in locals() and not csv_file.closed:
            csv_file.close()

if __name__ == '__main__':
    main()