import carla
import time
import math
import os
import csv
from queue import Queue, Empty

import cv2
import numpy as np

# ---------------- Config ----------------
HOST = "127.0.0.1"
PORT = 2000

LOG_FILE = "town01.log"   # asegúrate que este .log existe EN LA MISMA MÁQUINA del servidor CARLA

SAVE_PATH = "dataset_replay"
IMG_DIR = os.path.join(SAVE_PATH, "images")
CSV_PATH = os.path.join(SAVE_PATH, "dataset.csv")
os.makedirs(IMG_DIR, exist_ok=True)

IMG_W, IMG_H = 800, 600
FOV = 90
FPS = 20
DT = 1.0 / FPS
# ---------------------------------------

def bgr_from_carla_image(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]

def pick_vehicle(world: carla.World):
    """Intenta encontrar 'ego' / 'hero' del replay; si no, usa el primer vehículo."""
    vehicles = world.get_actors().filter("vehicle.*")
    if len(vehicles) == 0:
        return None

    # Prioridades típicas en logs
    for name in ("ego", "hero"):
        for v in vehicles:
            if v.attributes.get("role_name", "") == name:
                return v

    return vehicles[0]

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    original_settings = world.get_settings()

    camera = None
    csv_file = None

    try:
        # ✅ Fuerza SYNC y tick fijo (esto evita el freeze SIEMPRE)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        print("[INFO] Iniciando replay...")
        client.replay_file(LOG_FILE, 0.0, 0.0, 0)

        # Da unos ticks para que el replay “cargue” actores
        for _ in range(20):
            world.tick()

        ego = pick_vehicle(world)
        if ego is None:
            raise RuntimeError("No apareció ningún vehículo en el replay. ¿El .log tiene actores?")

        print(f"[OK] Vehículo para grabar: id={ego.id} role_name={ego.attributes.get('role_name','')}")

        # Sensor cámara
        bp_lib = world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(FOV))
        cam_bp.set_attribute("sensor_tick", str(DT))  # sincronizado al tick

        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=0.8, z=1.3)),
            attach_to=ego
        )

        image_q = Queue()
        camera.listen(image_q.put)

        # CSV
        csv_file = open(CSV_PATH, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "filename", "frame", "timestamp_s",
            "speed_mps", "speed_kmh", "v_forward_mps", "yaw_rate_rps"
        ])
        csv_file.flush()

        cv2.namedWindow("REPLAY CAM", cv2.WINDOW_AUTOSIZE)

        frame_id = 0
        print("[INFO] Grabando + mostrando. Presiona 'q' para salir.")

        while True:
            # ✅ Avanza replay (esto es lo que te faltaba cuando se quedaba detenido)
            world.tick()

            # Lee la imagen más reciente (vacía la cola)
            img = None
            while True:
                try:
                    img = image_q.get_nowait()
                except Empty:
                    break

            if img is None:
                # a veces el sensor tarda, sigue
                continue

            bgr = bgr_from_carla_image(img)

            # Mostrar ventana
            cv2.imshow("REPLAY CAM", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Guardar imagen
            fname = f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(os.path.join(IMG_DIR, fname), bgr)

            # Medidas del vehículo
            vel = ego.get_velocity()
            ang = ego.get_angular_velocity()
            tf = ego.get_transform()
            fwd = tf.get_forward_vector()

            speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            speed_kmh = speed_mps * 3.6
            v_fwd = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
            yaw_rate = ang.z

            writer.writerow([
                fname,
                int(img.frame),
                float(img.timestamp),
                float(speed_mps),
                float(speed_kmh),
                float(v_fwd),
                float(yaw_rate)
            ])

            if frame_id % 20 == 0:
                csv_file.flush()

            frame_id += 1

    finally:
        # Limpieza
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        try:
            if camera is not None:
                camera.destroy()
        except Exception:
            pass

        try:
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
        except Exception:
            pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("[OK] Dataset:")
        print(" - Imágenes:", IMG_DIR)
        print(" - CSV:", CSV_PATH)

if __name__ == "__main__":
    main()
