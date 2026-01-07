import carla
import os
import csv
import time
import threading
from queue import Queue

import cv2
import numpy as np
import math

# ------------------ Configuración ------------------
SAVE_PATH   = "dataset_ego"
IMG_DIR     = os.path.join(SAVE_PATH, "images")
CSV_PATH    = os.path.join(SAVE_PATH, "dataset.csv")

os.makedirs(IMG_DIR, exist_ok=True)

HOST = "127.0.0.1"
PORT = 2000

# Cámara
CAM_FPS     = 20
IMG_W, IMG_H = 800, 600
JPG_QUALITY = 80

# Log/print
PRINT_HZ = 10  # cuántas veces por segundo imprime (solo monitor)
# ---------------------------------------------------

writer_q = Queue(maxsize=256)

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

def find_ego(world, timeout_s=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for v in world.get_actors().filter("vehicle.*"):
            if v.attributes.get("role_name", "") == "ego":
                return v
        time.sleep(0.2)
    return None

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # 1) Encontrar ego (creado por tu otro script)
    print("[INFO] Buscando vehículo con role_name='ego' ...")
    ego = find_ego(world, timeout_s=30.0)
    if ego is None:
        print("[ERROR] No se encontró vehículo 'ego'. Corre primero control_carla_simple_axes.py")
        return
    print(f"[OK] Ego encontrado: id={ego.id}")

    # 2) Abrir CSV
    csv_file = open(CSV_PATH, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "filename",
        "frame",
        "timestamp_s",
        "speed_mps",
        "speed_kmh",
        "v_forward_mps",
        "yaw_rate_rps",
        "throttle",
        "brake",
        "steer",
        "reverse"
    ])
    csv_file.flush()

    # 3) Spawn cámara pegada al ego (no tocamos world settings: tu script controla sync/tick)
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", "90")
    cam_bp.set_attribute("sensor_tick", f"{1.0 / CAM_FPS:.4f}")

    cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    # 4) Hilo escritor de imágenes
    tw = threading.Thread(target=writer_loop, args=(IMG_DIR, JPG_QUALITY), daemon=True)
    tw.start()

    # 5) Estado
    frame_id = 0
    last_print = 0.0
    print_dt = 1.0 / max(1, PRINT_HZ)

    def on_image(image: carla.Image):
        nonlocal frame_id, last_print

        # --- Guardar imagen ---
        bgr = image_to_bgr(image)
        filename = f"frame_{frame_id:06d}.jpg"
        try:
            writer_q.put((filename, bgr), block=False)
        except Exception:
            # si se llena la cola, descartamos frame (pero igual podrías registrar datos)
            return

        # --- Medición del ego (en el mismo instante del callback) ---
        tf = ego.get_transform()
        vel = ego.get_velocity()
        ang = ego.get_angular_velocity()
        ctl = ego.get_control()

        speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_kmh = speed_mps * 3.6

        fwd = tf.get_forward_vector()
        v_forward = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z

        yaw_rate = ang.z

        # Timestamp del sensor (segundos simulados)
        ts = float(image.timestamp)

        # --- Escribir CSV (SIEMPRE) ---
        writer.writerow([
            filename,
            int(image.frame),
            ts,
            float(speed_mps),
            float(speed_kmh),
            float(v_forward),
            float(yaw_rate),
            float(ctl.throttle),
            float(ctl.brake),
            float(ctl.steer),
            int(ctl.reverse)
        ])

        # flush cada N filas para no perder datos si se cierra abrupto
        if frame_id % 20 == 0:
            csv_file.flush()

        # --- Monitor en tiempo real (opcional) ---
        now = time.time()
        if now - last_print >= print_dt:
            last_print = now
            print(
                f"[frame {image.frame}] "
                f"speed={speed_mps:6.2f} m/s ({speed_kmh:6.1f} km/h) | "
                f"v_fwd={v_forward:6.2f} | yaw_rate={yaw_rate:7.3f} | "
                f"thr={ctl.throttle:4.2f} brk={ctl.brake:4.2f} str={ctl.steer:5.2f} rev={int(ctl.reverse)}"
            )

        frame_id += 1

    camera.listen(on_image)

    try:
        dur_s = 60 * 20  # 20 minutos (ajusta)
        print(f"[INFO] Grabando dataset por ~{dur_s} s. (Ctrl+C para parar)")
        time.sleep(dur_s)
    except KeyboardInterrupt:
        print("[INFO] Interrumpido por el usuario.")
    finally:
        print("[INFO] Cerrando...")

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

        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass

        print("[OK] Dataset listo:")
        print(" - Imágenes:", IMG_DIR)
        print(" - CSV:", CSV_PATH)

if __name__ == "__main__":
    main()
