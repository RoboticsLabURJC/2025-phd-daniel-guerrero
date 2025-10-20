# recorder_optimizado.py
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

CAM_FPS = 20                 # 20 FPS (sensor_tick = 0.05)
IMG_W, IMG_H = 800, 600      # resolución (ajusta si quieres)
JPG_QUALITY = 80             # 0-100 (más alto = mejor calidad, más peso)

CONTROL_HZ = 20              # muestreo de controles (20 Hz)
CONTROL_DT = 1.0 / CONTROL_HZ

# Cola para escritura de imágenes (evita bloquear el callback)
# Sube el maxsize si necesitas aguantar ráfagas más grandes
writer_q = Queue(maxsize=256)

# ------------------ Estado global ------------------
frame_id = 0
controls_dict = {}
last_control = None
stop_writer = False

# ------------------ Utilidades ---------------------
def image_to_bgr(image: carla.Image):
    """Convierte la imagen de CARLA (BGRA) a BGR uint8 numpy."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]  # BGR

def writer_loop(img_dir: str, jpg_quality: int):
    """Hilo escritor: toma frames de la cola y guarda JPG en disco."""
    global stop_writer
    while True:
        item = writer_q.get()
        if item is None:
            break
        filename, bgr = item
        # Codifica a JPG (rápido y con control de calidad)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        if ok:
            path = os.path.join(img_dir, filename)
            with open(path, "wb") as f:
                f.write(buf.tobytes())

# ------------------ Callbacks/Hilos ----------------
def save_image_cb(image: carla.Image, writer, img_dir: str):
    """Callback del sensor: encola imagen y escribe controles."""
    global frame_id, last_control

    # Convertir a BGR aquí (rápido) y encolar para IO en otro hilo
    bgr = image_to_bgr(image)
    filename = f"frame_{frame_id:05d}.jpg"

    # Encolar imagen para que la guarde el hilo escritor
    try:
        writer_q.put((filename, bgr), block=False)
    except Exception:
        # Si la cola se llena, puedes: (a) descartar frame, (b) bloquear.
        # Aquí descartamos para no frenar la sim:
        return

    # Buscar control del mismo frame; si no, usar el último control conocido
    control = controls_dict.get(image.frame, last_control)
    if control is not None:
        writer.writerow([filename, control.steer, control.throttle, control.brake])
        # Sólo avanzamos frame_id cuando tenemos control válido
        frame_id += 1

def capture_control_loop(vehicle: carla.Vehicle, world: carla.World):
    """Hilo que muestrea controles del vehículo a CONTROL_HZ y los asocia a frame del mundo."""
    global last_control
    while True:
        try:
            control = vehicle.get_control()
            last_control = control
            snap = world.get_snapshot()
            controls_dict[snap.frame] = control
            time.sleep(CONTROL_DT)
        except Exception:
            # Probablemente el mundo/vehículo ya fue destruido
            break

# ------------------ Main ---------------------------
def main():
    global stop_writer

    # CSV listo
    csv_file = open(CSV_PATH, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "steer", "throttle", "brake"])

    # Conexión a CARLA
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Vehículo
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        # Fallback si el primer punto está ocupado
        vehicle = world.spawn_actor(vehicle_bp, carla.Transform())

    vehicle.set_autopilot(True)

    # Cámara RGB (20 FPS)
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMG_W))
    camera_bp.set_attribute("image_size_y", str(IMG_H))
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", f"{1.0 / CAM_FPS:.4f}")  # LIMITE DE FPS

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Lanzar hilo escritor
    tw = threading.Thread(target=writer_loop, args=(IMG_DIR, JPG_QUALITY), daemon=True)
    tw.start()

    # Lanzar hilo de captura de control (20 Hz)
    tctrl = threading.Thread(target=capture_control_loop, args=(vehicle, world), daemon=True)
    tctrl.start()

    # Iniciar escucha de cámara
    camera.listen(lambda image: save_image_cb(image, writer, IMG_DIR))

    try:
        dur_s = 60 * 20  # 20 minutos
        print(f"[INFO] Grabando por {dur_s} s a {CAM_FPS} FPS (JPG q={JPG_QUALITY})…")
        time.sleep(dur_s)

    finally:
        print("[INFO] Terminando grabación...")
        try:
            camera.stop()
        except Exception:
            pass

        # Señal para cerrar el hilo escritor y esperar un poco
        writer_q.put(None)
        tw.join(timeout=5)

        # Destruir actores
        try:
            camera.destroy()
        except Exception:
            pass
        try:
            vehicle.destroy()
        except Exception:
            pass

        # Cerrar CSV
        csv_file.close()

        print("[INFO] Listo. Imágenes en:", IMG_DIR)
        print("[INFO] Controles en:", CSV_PATH)

if __name__ == "__main__":
    main()
