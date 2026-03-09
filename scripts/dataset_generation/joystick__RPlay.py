import os
import csv
import pygame
import cv2
import numpy as np
import carla
from queue import Queue, Empty

# ---------- Parámetros ----------
HOST = "127.0.0.1"
PORT = 2000
OUT_DIR = "dataset_joystick"

IMG_W, IMG_H = 640, 360
CAM_FPS = 20
FIXED_DT = 1.0 / CAM_FPS
# -------------------------------

class JoystickHandler:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("[ERROR] No se detectó ningún Joystick conectado.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"[INFO] Control detectado: {self.joystick.get_name()}")

    def get_control(self):
        pygame.event.pump()
        
        # Axis 0: Steering (-1 a 1)
        steer = self.joystick.get_axis(0)
        
        # Axis 4: Throttle (-1 reposo a 1 a fondo) -> Mapear a (0 a 1)
        t_axis = self.joystick.get_axis(4)
        throttle = max(0.0, (t_axis + 1.0) / 2.0)
        
        # Axis 5: Brake (-1 reposo a 1 a fondo) -> Mapear a (0 a 1)
        b_axis = self.joystick.get_axis(5)
        brake = max(0.0, (b_axis + 1.0) / 2.0)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        return control

def to_bgr(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()

def apply_half_mask(frame_bgr):
    h = frame_bgr.shape[0]
    frame_bgr[0:h//2, :] = 0
    return frame_bgr

def main():
    # Crear carpetas de salida
    os.makedirs(os.path.join(OUT_DIR, "rgb"), exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "driving_log.csv")

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()

    # 1. Buscar o Spawnear Vehículo Ego correctamente
    vehicle = None
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') == 'ego':
            vehicle = actor
            break
            
    if vehicle is None:
        print("[INFO] Spawneando nuevo vehículo 'ego'...")
        bp_lib = world.get_blueprint_library()
        v_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        # CORRECCIÓN: El atributo se setea en el Blueprint
        v_bp.set_attribute('role_name', 'ego')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(v_bp, spawn_point)

    # 2. Configuración de Cámara (Igual a tu primer script)
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", "90")

    cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
    
    image_queue = Queue()
    camera.listen(image_queue.put)

    # 3. Configurar Modo Síncrono
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    js = JoystickHandler()

    print(f"[INFO] Grabación iniciada en: {OUT_DIR}")
    print("[HINT] Conduce con el joystick. Presiona 'Q' en la ventana de imagen para salir.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "image_path", "steer", "throttle", "brake"])

        step = 0
        try:
            while True:
                # Aplicar control del joystick antes del tick
                current_control = js.get_control()
                vehicle.apply_control(current_control)

                world.tick()

                try:
                    img = image_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Procesar imagen
                frame = to_bgr(img)
                frame_masked = apply_half_mask(frame)

                # Visualización
                cv2.imshow("Grabando Dataset (Joystick)", frame_masked)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Guardar imagen y telemetría
                img_filename = f"rgb/{step:06d}.jpg"
                img_path = os.path.join(OUT_DIR, img_filename)
                cv2.imwrite(img_path, frame_masked)

                writer.writerow([
                    step, 
                    img_filename, 
                    round(current_control.steer, 5), 
                    round(current_control.throttle, 5), 
                    round(current_control.brake, 5)
                ])
                
                step += 1
                if step % 100 == 0:
                    print(f"-> Frames grabados: {step}")

        finally:
            print("[INFO] Limpiando y restaurando...")
            settings.synchronous_mode = False
            world.apply_settings(settings)
            
            if camera is not None:
                camera.stop()
                camera.destroy()
            cv2.destroyAllWindows()
            pygame.quit()

if __name__ == "__main__":
    main()