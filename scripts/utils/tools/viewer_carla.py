#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import cv2
import numpy as np
from collections import deque

# --- Importa CARLA (asegúrate de que el egg esté en PYTHONPATH si hace falta) ---
try:
    import carla
except ImportError as e:
    print("No se pudo importar 'carla'. Asegúrate de tener el paquete de Python de CARLA en el PYTHONPATH.")
    print("En Linux suele ser: PythonAPI/carla/dist/carla-<version>-py<ver>-linux-x86_64.egg")
    raise

# ------------------ Utilidades de dibujo ------------------
def clamp(x, a, b):
    return max(a, min(b, x))

def draw_bar(img, x, y, w, h, value, label, color=(80, 220, 80)):
    value = float(clamp(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
    fill_h = int(h * value)
    y0 = y + h - fill_h
    y1 = y + h
    cv2.rectangle(img, (x + 2, y0 + 2), (x + w - 2, y1 - 2), color, -1)
    cv2.putText(img, f"{label}: {value:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

def draw_center_steer_bar(img, cx, cy, w, h, steer_val, color=(80, 220, 80), label="steer"):
    steer_val = float(clamp(steer_val, -1.0, 1.0))
    x0 = int(cx - w // 2); x1 = int(cx + w // 2)
    y0 = int(cy - h // 2); y1 = int(cy + h // 2)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 2)
    cv2.line(img, (cx, y0), (cx, y1), (180, 180, 180), 1)
    dx = int((w // 2) * steer_val)
    cv2.rectangle(img, (cx + dx - 6, y0 + 2), (cx + dx + 6, y1 - 2), color, -1)
    cv2.putText(img, f"{label}: {steer_val:+.3f}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

def draw_steer_arrow(img, center, length, steer_val, color=(80, 220, 80), thickness=6):
    steer_val = float(clamp(steer_val, -1.0, 1.0))
    max_deg = 30.0
    angle_deg = steer_val * max_deg
    angle_rad = np.deg2rad(angle_deg - 90)
    x0, y0 = center
    x1 = int(x0 + length * np.cos(angle_rad))
    y1 = int(y0 + length * np.sin(angle_rad))
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.25)
    cv2.putText(img, f"{angle_deg:+.1f}°", (x0 - 30, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

def get_speed_kmh(vehicle):
    v = vehicle.get_velocity()
    # m/s -> km/h
    return 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)

# ------------------ Clase visor en vivo ------------------
class CarlaLiveViewer:
    def __init__(self, host, port, town, res, fov, fps_sensor, autopilot, save_path, sync):
        self.host = host
        self.port = port
        self.town = town
        self.width, self.height = res
        self.fov = fov
        self.fps_sensor = fps_sensor
        self.autopilot = autopilot
        self.save_path = save_path
        self.sync = sync

        self.client = None
        self.world = None
        self.original_settings = None
        self.traffic_manager = None
        self.vehicle = None
        self.camera = None
        self.spectator = None
        self.image_queue = deque(maxlen=5)
        self.writer = None
        self.hud_h = 140

    def connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        if self.town:
            self.client.load_world(self.town)
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()

        # Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        # Ajusta modo síncrono si se pide
        settings = self.world.get_settings()
        self.original_settings = settings

        if self.sync:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / max(1, self.fps_sensor)  # sincroniza con la cámara
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
        else:
            # Asegura asincrónico si no se quiere síncrono
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

    def spawn_vehicle_and_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        # Vehículo
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0] if blueprint_library.filter("vehicle.tesla.model3") else blueprint_library.filter("vehicle.*")[0]

        # Punto de spawn
        spawn_points = self.world.get_map().get_spawn_points()
        transform = np.random.choice(spawn_points)

        self.vehicle = self.world.spawn_actor(vehicle_bp, transform)

        # Cámara RGB
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.width))
        cam_bp.set_attribute("image_size_y", str(self.height))
        cam_bp.set_attribute("fov", str(self.fov))
        cam_bp.set_attribute("sensor_tick", str(1.0 / max(1, self.fps_sensor)))

        # Montaje en el capó
        cam_transform = carla.Transform(carla.Location(x=1.6, z=1.5))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)

        # Spectator follow (bonito para ver desde arriba/tercera persona)
        self.follow_vehicle_with_spectator()

        # Autopilot
        if self.autopilot:
            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 0)
        else:
            self.vehicle.set_autopilot(False)

        # Callback de imágenes
        self.camera.listen(self._on_image)

    def follow_vehicle_with_spectator(self):
        if not self.vehicle or not self.spectator:
            return
        transform = self.vehicle.get_transform()
        # offset detrás y un poco arriba
        offset = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-15))
        self.spectator.set_transform(transform * offset)

    def _on_image(self, image):
        # CARLA entrega BGRA uint8 lineal en image.raw_data
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # BGR
        self.image_queue.append(array)

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        canvas = np.zeros((h + self.hud_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = frame

        # Lectura de controles y velocidad
        control = self.vehicle.get_control() if self.vehicle is not None else carla.VehicleControl()
        steer = float(control.steer)
        throttle = float(control.throttle)
        brake = float(control.brake)
        speed_kmh = get_speed_kmh(self.vehicle) if self.vehicle is not None else 0.0

        pad = 20
        bar_w = 28
        bar_h = 100

        draw_bar(canvas, pad, h + pad, bar_w, bar_h, throttle, "throttle", (80, 220, 80))
        draw_bar(canvas, pad + bar_w + 12, h + pad, bar_w, bar_h, brake, "brake", (60, 60, 230))

        cx = w // 2
        draw_center_steer_bar(canvas, cx, h + 30, 360, 22, steer, (80, 220, 80), "steer")
        draw_steer_arrow(canvas, (cx, h + 90), 70, steer, (80, 220, 80), thickness=6)

        # Texto de estado
        txt = f"Speed: {speed_kmh:5.1f} km/h   Autopilot: {'ON' if self.autopilot else 'OFF'}   Sync: {self.sync}"
        cv2.putText(canvas, txt, (pad, h + self.hud_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

        return canvas

    def maybe_init_writer(self, sample_frame):
        if self.save_path and self.writer is None:
            H, W = sample_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.save_path, fourcc, float(self.fps_sensor), (W, H))

    def loop(self):
        print("Entrando al bucle principal. Teclas: Q/ESC = salir, P = pausa, A = toggle autopilot.")
        last_spec_update = time.time()
        paused = False

        while True:
            if self.sync:
                self.world.tick()
            else:
                self.world.wait_for_tick()

            # Refresca spectator cada ~0.5s para seguir al vehículo
            if time.time() - last_spec_update > 0.5:
                self.follow_vehicle_with_spectator()
                last_spec_update = time.time()

            if not self.image_queue:
                # No hay frame aún
                continue

            bgr = self.image_queue[-1]
            hud_frame = self.draw_hud(bgr)
            self.maybe_init_writer(hud_frame)

            cv2.imshow("CARLA Live Viewer", hud_frame)
            if self.writer:
                self.writer.write(hud_frame)

            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('a'):
                # Toggle autopilot en vivo
                self.autopilot = not self.autopilot
                self.vehicle.set_autopilot(self.autopilot, self.traffic_manager.get_port())

    def cleanup(self):
        print("Limpiando actores y restaurando ajustes…")
        try:
            if self.writer:
                self.writer.release()
            cv2.destroyAllWindows()
        except:
            pass

        if self.camera is not None:
            self.camera.stop()
        actors = [self.camera, self.vehicle]
        for a in actors:
            try:
                if a is not None:
                    a.destroy()
            except:
                pass

        # Restaurar settings del mundo
        if self.world and self.original_settings:
            try:
                self.world.apply_settings(self.original_settings)
            except:
                pass

        if self.traffic_manager:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except:
                pass

def main():
    parser = argparse.ArgumentParser(description="Visor en vivo conectado al API de CARLA")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="", help="Mapa/Town (opcional). Ej: Town03")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--fps_sensor", type=int, default=20, help="FPS de la cámara (y tick si sync)")
    parser.add_argument("--autopilot", action="store_true", help="Inicia con autopilot ON")
    parser.add_argument("--save", type=str, default=None, help="Ruta MP4 para guardar (opcional)")
    parser.add_argument("--sync", action="store_true", help="Usar modo síncrono")
    args = parser.parse_args()

    res = (args.width, args.height)
    viewer = CarlaLiveViewer(
        host=args.host,
        port=args.port,
        town=args.town if args.town else None,
        res=res,
        fov=args.fov,
        fps_sensor=args.fps_sensor,
        autopilot=args.autopilot,
        save_path=args.save,
        sync=args.sync
    )

    try:
        viewer.connect()
        viewer.spawn_vehicle_and_camera()
        viewer.loop()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        viewer.cleanup()

if __name__ == "__main__":
    main()
