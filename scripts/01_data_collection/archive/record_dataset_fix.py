# recorder_optimizado_sync.py
import carla
import os
import csv
import time
import numpy as np
import threading
from queue import Queue

import cv2  # para codificar JPG con imencode

# ------------------ Configuración ------------------
SAVE_PATH = "dataset"
IMG_DIR = os.path.join(SAVE_PATH, "images")
os.makedirs(IMG_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_PATH, "controls.csv")

HOST = "localhost"
PORT = 2000

CAM_FPS = 20                 # 20 FPS
IMG_W, IMG_H = 800, 600      # resolución
JPG_QUALITY = 80             # 0-100 (más alto = mejor calidad, más peso)

CONTROL_HZ = 20              # muestreo de controles (20 Hz)
CONTROL_DT = 1.0 / CONTROL_HZ

# Cola para escritura de imágenes
writer_q = Queue(maxsize=256)

# ------------------ Estado global ------------------
frame_id = 0
controls_dict = {}
last_control = None
stop_writer = False
stop_capture = False

# ------------------ Utilidades ---------------------
def image_to_bgr(image: carla.Image):
    """Convierte la imagen de CARLA (BGRA) a BGR uint8 numpy."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]  # BGR

def writer_loop(img_dir: str, jpg_quality: int):
    """Hilo escritor: toma frames de la cola y guarda JPG en disco."""
    while True:
        item = writer_q.get()
        if item is None:
            break
        filename, bgr = item
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        if ok:
            path = os.path.join(img_dir, filename)
            with open(path, "wb") as f:
                f.write(buf.tobytes())

# ------------------ Callbacks/Hilos ----------------
def save_image_cb(image: carla.Image, writer, img_dir: str):
    """Callback del sensor: encola imagen y escribe controles."""
    global frame_id, last_control

    bgr = image_to_bgr(image)
    filename = f"frame_{frame_id:05d}.jpg"

    # Encolar imagen
    try:
        writer_q.put((filename, bgr), block=False)
    except Exception:
        # Si la cola se llena, descartamos frame para no frenar la sim
        return

    # Buscar control del mismo frame; si no, usar el último control conocido
    control = controls_dict.get(image.frame, last_control)
    if control is not None:
        writer.writerow([filename, control.steer, control.throttle, control.brake])
        frame_id += 1

def capture_control_loop(vehicle: carla.Vehicle, world: carla.World):
    """Hilo que muestrea controles del vehículo a CONTROL_HZ."""
    global last_control, stop_capture
    while not stop_capture:
        try:
            control = vehicle.get_control()
            last_control = control
            snap = world.get_snapshot()
            controls_dict[snap.frame] = control
            time.sleep(CONTROL_DT)
        except Exception:
            break

# ------------------ Main ---------------------------
def main():
    global stop_writer, stop_capture

    # CSV listo
    csv_file = open(CSV_PATH, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "steer", "throttle", "brake"])

    # Conexión a CARLA
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Guardar settings originales
    original_settings = world.get_settings()

    try:
        # ----- Modo síncrono -----
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / CAM_FPS
        settings.no_rendering_mode = False   # pon True si quieres máxima velocidad sin render
        world.apply_settings(settings)

        # Vehículo
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        # Role name solo por claridad (no es obligatorio para la cámara)
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "ego_recorder")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            spawn_points = [carla.Transform()]

        vehicle = None
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle:
                break
        if vehicle is None:
            # Fallback duro si todos estaban ocupados
            vehicle = world.spawn_actor(vehicle_bp, carla.Transform())

        vehicle.set_autopilot(True)

        # Cámara RGB
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMG_W))
        camera_bp.set_attribute("image_size_y", str(IMG_H))
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", f"{1.0 / CAM_FPS:.4f}")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Lanzar hilo escritor
        tw = threading.Thread(target=writer_loop, args=(IMG_DIR, JPG_QUALITY), daemon=True)
        tw.start()

        # Lanzar hilo de captura de control
        tctrl = threading.Thread(target=capture_control_loop, args=(vehicle, world), daemon=True)
        tctrl.start()

        # Iniciar escucha de cámara
        camera.listen(lambda image: save_image_cb(image, writer, IMG_DIR))

        # Duración deseada
        dur_s = 60 * 20  # 20 minutos
        total_ticks = int(dur_s * CAM_FPS)

        print(f"[INFO] Grabando por ~{dur_s} s a {CAM_FPS} FPS (JPG q={JPG_QUALITY})…")

        # Bucle principal síncrono
        for _ in range(total_ticks):
            world.tick()

        print("[INFO] Tiempo de grabación alcanzado.")

    finally:
        print("[INFO] Terminando grabación...")

        # Parar hilos de control / cámara
        stop_capture = True
        try:
            camera.stop()
        except Exception:
            pass

        # Señal para cerrar el hilo escritor
        writer_q.put(None)
        try:
            tw.join(timeout=5)
        except Exception:
            pass

        # Destruir actores
        try:
            camera.destroy()
        except Exception:
            pass
        try:
            vehicle.destroy()
        except Exception:
            pass

        # Restaurar settings del mundo
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        # Cerrar CSV
        csv_file.close()

        print("[INFO] Listo. Imágenes en:", IMG_DIR)
        print("[INFO] Controles en:", CSV_PATH)

if __name__ == "__main__":
    main()
