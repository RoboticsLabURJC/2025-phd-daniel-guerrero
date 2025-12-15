# recorder_from_ego.py
import carla
import os
import csv
import time
import threading
from queue import Queue

import cv2
import numpy as np

# ------------------ Configuración ------------------
SAVE_PATH   = "dataset_ego"
IMG_DIR     = os.path.join(SAVE_PATH, "images")
CSV_PATH    = os.path.join(SAVE_PATH, "controls.csv")

os.makedirs(IMG_DIR, exist_ok=True)

HOST = "localhost"
PORT = 2000

CAM_FPS     = 20            # Solo para sensor_tick (frecuencia de cámara)
IMG_W, IMG_H = 800, 600
JPG_QUALITY = 80            # Calidad JPG

CONTROL_HZ  = 20            # Frecuencia a la que muestreamos controles
CONTROL_DT  = 1.0 / CONTROL_HZ

writer_q = Queue(maxsize=256)

# ------------------ Estado global ------------------
frame_id      = 0
controls_dict = {}
last_control  = None
stop_capture  = False

# ------------------ Utilidades ---------------------
def image_to_bgr(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]

def writer_loop(img_dir: str, jpg_quality: int):
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

def save_image_cb(image: carla.Image, writer, img_dir: str):
    global frame_id, last_control

    bgr = image_to_bgr(image)
    filename = f"frame_{frame_id:05d}.jpg"

    # Encolamos imagen (no bloqueante)
    try:
        writer_q.put((filename, bgr), block=False)
    except Exception:
        # Si se llena la cola, descartamos frame
        return

    # Buscamos medición asociada a este frame del mundo
    control = controls_dict.get(image.frame, last_control)
    if control is not None:
        writer.writerow([filename, control[0], control[1]])
        frame_id += 1

def capture_control_loop(vehicle: carla.Vehicle, world: carla.World):
    """Lee estado del vehículo a CONTROL_HZ y lo asocia al frame del mundo."""
    global last_control, stop_capture
    while not stop_capture:
        try:
            # v (m/s) hacia adelante = dot(vel_world, forward_world)
            vel = vehicle.get_velocity()
            fwd = vehicle.get_transform().get_forward_vector()
            v_mps = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z

            # omega (rad/s) = yaw rate = angular_velocity.z
            ang = vehicle.get_angular_velocity()
            omega_rps = ang.z

            snap = world.get_snapshot()
            last_control = (v_mps, omega_rps)
            controls_dict[snap.frame] = (v_mps, omega_rps)

            time.sleep(CONTROL_DT)
        except Exception:
            break

# ------------------ Main ---------------------------
def main():
    global stop_capture

    # CSV
    csv_file = open(CSV_PATH, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "v_mps", "omega_rps"])

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # NO tocamos world.apply_settings()
    # El script del volante es el que debe tener el control del modo síncrono.

    # Buscamos el vehículo con role_name = "ego" (el de tu script de volante)
    ego_vehicle = None
    print("[INFO] Buscando vehículo con role_name='ego' ...")
    for _ in range(60):  # ~30 s máximo
        actors = world.get_actors().filter("vehicle.*")
        for v in actors:
            try:
                if v.attributes.get("role_name", "") == "ego":
                    ego_vehicle = v
                    break
            except Exception:
                continue
        if ego_vehicle is not None:
            break
        time.sleep(0.5)

    if ego_vehicle is None:
        print("[ERROR] No se encontró vehículo con role_name='ego'.")
        print("Asegúrate de correr primero tu script de control (control_carla_simple_axes.py).")
        csv_file.close()
        return

    print(f"[INFO] Encontrado vehículo ego: id={ego_vehicle.id}")

    # Cámara RGB adherida al ego
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", f"{1.0 / CAM_FPS:.4f}")

    cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego_vehicle)

    # Hilo escritor
    tw = threading.Thread(target=writer_loop, args=(IMG_DIR, JPG_QUALITY), daemon=True)
    tw.start()

    # Hilo de captura de estado (v, omega)
    tctrl = threading.Thread(target=capture_control_loop, args=(ego_vehicle, world), daemon=True)
    tctrl.start()

    # Callback de cámara
    camera.listen(lambda image: save_image_cb(image, writer, IMG_DIR))

    try:
        dur_s = 60 * 20  # 20 minutos, ajusta si quieres
        print(f"[INFO] Grabando ~{dur_s} s. Usa tu volante con el otro script.")
        time.sleep(dur_s)
    except KeyboardInterrupt:
        print("[INFO] Interrumpido por el usuario.")
    finally:
        print("[INFO] Terminando grabación...")

        stop_capture = True
        try:
            camera.stop()
        except Exception:
            pass

        writer_q.put(None)
        try:
            tw.join(timeout=5)
        except Exception:
            pass

        try:
            camera.destroy()
        except Exception:
            pass

        csv_file.close()

        print("[INFO] Imágenes en:", IMG_DIR)
        print("[INFO] Controles en:", CSV_PATH)

if __name__ == "__main__":
    main()
