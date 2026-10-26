import os
import glob
import time
from queue import Queue, Empty

import cv2
import numpy as np
import carla
import h5py  # ¡La nueva librería estrella!

# ---------- Parámetros ----------
HOST = "127.0.0.1"
PORT = 2000

# Carpeta donde pondrás todos tus archivos .log
LOGS_DIR = "logs_crudos"  
# El super-archivo de salida
H5_OUT_PATH = "dataset_260329.h5" 

IMG_W, IMG_H = 640, 360
CAM_FPS = 10  # Bajamos a 10 FPS (100 ms) para eliminar redundancia
FIXED_DT = 1.0 / CAM_FPS
# -------------------------------

def to_bgr(image: carla.Image):
    """Convierte la imagen raw de CARLA a formato BGR de OpenCV."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()

def apply_half_mask(frame_bgr):
    """Enmascara la mitad superior de la imagen (el cielo)."""
    h = frame_bgr.shape[0]
    frame_bgr[0:h//2, :] = 0
    return frame_bgr

def main():
    # 1. Buscar todos los logs en la carpeta
    log_files = glob.glob(os.path.join(LOGS_DIR, "*.log"))
    if not log_files:
        print(f"[ERROR] No se encontraron archivos .log en la carpeta: {LOGS_DIR}")
        return

    print(f"[INFO] Se encontraron {len(log_files)} archivos .log para procesar.")

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)
    world = client.get_world()

    # 2. Preparar el archivo HDF5 masivo
    # Usamos 'w' para crearlo desde cero (sobrescribe si ya existe)
    with h5py.File(H5_OUT_PATH, 'w') as h5_file:
        
        # Creamos los datasets expandibles (maxshape=None permite que crezcan infinitamente)
        # Imágenes: (N, 360, 640, 3), formato uint8 (0-255) para ahorrar espacio
        img_ds = h5_file.create_dataset(
            "images", 
            shape=(0, IMG_H, IMG_W, 3), 
            maxshape=(None, IMG_H, IMG_W, 3), 
            dtype='uint8',
            compression="gzip",       # Ligera compresión para no llenar tu disco
            compression_opts=4
        )
        
        # Controles: (N, 3) -> [steer, throttle, brake], formato float32
        ctrl_ds = h5_file.create_dataset(
            "controls", 
            shape=(0, 3), 
            maxshape=(None, 3), 
            dtype='float32'
        )

        global_step = 0

        # 3. Bucle para procesar cada archivo .log
        for log_path in log_files:
            abs_log_path = os.path.abspath(log_path)
            
            # Obtener duración del log
            info = client.show_recorder_file_info(abs_log_path, False)
            duration_str = [line for line in info.split('\n') if line.startswith('Duration:')]
            total_duration = float(duration_str[0].split()[1]) if duration_str else 0.0
            
            print(f"\n[INFO] === Procesando: {os.path.basename(log_path)} ({total_duration} segs) ===")
            if total_duration < 2.0:
                print(f"[ADVERTENCIA] Log demasiado corto ({total_duration}s). Saltando al siguiente...")
                continue

            client.replay_file(abs_log_path, 0.0, 0.0, 0)
            
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DT
            world.apply_settings(settings)

            world.tick()
            world.tick()

            # Buscar el vehículo Ego
            vehicle = None
            for actor in world.get_actors().filter('vehicle.*'):
                if actor.attributes.get('role_name') == 'ego':
                    vehicle = actor
                    break
                    
            if vehicle is None:
                print(f"[ADVERTENCIA] No se encontró el vehículo 'ego' en {log_path}. Saltando al siguiente.")
                continue

            # Acoplar cámara
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
                # Bucle de extracción del log actual
                while elapsed_time < total_duration:
                    world.tick()
                    elapsed_time += FIXED_DT

                    try:
                        img = image_queue.get(timeout=2.0)
                    except Empty:
                        continue

                    # Procesar imagen
                    frame = to_bgr(img)
                    frame_masked = apply_half_mask(frame)

                    # --- MAGIA HDF5: Expandir y guardar ---
                    # Hacemos que el dataset crezca 1 fila más
                    img_ds.resize(global_step + 1, axis=0)
                    ctrl_ds.resize(global_step + 1, axis=0)
                    
                    # Guardamos los datos en esa nueva fila
                    img_ds[global_step] = frame_masked
                    
                    control = vehicle.get_control()
                    ctrl_ds[global_step] = [control.steer, control.throttle, control.brake]
                    
                    # Mostrar progreso
                    cv2.imshow("Generador Masivo HDF5", frame_masked)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Extracción cancelada por el usuario.")
                        return # Sale de todo el script
                        
                    local_step += 1
                    global_step += 1
                    
                    if local_step % 50 == 0:
                        print(f"  -> Archivo actual: {elapsed_time:.1f}/{total_duration:.1f}s | Total acumulado: {global_step} frames")

            finally:
                # Limpiar cámara antes de abrir el siguiente log
                if camera is not None:
                    camera.stop()
                    camera.destroy()

        print(f"\n[ÉXITO] Procesamiento por lotes finalizado.")
        print(f"[INFO] Dataset masivo creado: {H5_OUT_PATH}")
        print(f"[INFO] Total de frames limpios a 10 FPS: {global_step}")

    # Restaurar simulador
    settings.synchronous_mode = False
    world.apply_settings(settings)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()