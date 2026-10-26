import carla
import time
import cv2
import numpy as np
import queue  # <-- IMPORTANTE: Nueva librería estándar de Python

# Creamos una cola (Queue) para pasar los datos entre el sensor y el hilo principal
image_queue = queue.Queue()

def process_image(image):
    """Convierte los datos crudos y los envía a la cola en lugar de dibujarlos directamente"""
    raw_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    raw_data = np.reshape(raw_data, (image.height, image.width, 4)) # Formato BGRA
    image_rgb = raw_data[:, :, :3] # Nos quedamos con BGR para OpenCV
    
    # Metemos la imagen procesada a la cola
    image_queue.put(image_rgb)

def main():
    # Conexión al servidor
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    # Ruta de tu log
    log_path = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_dagger/dagger_log_20260510_191929.log"

    print(f"[INFO] Iniciando reproducción de: {log_path.split('/')[-1]}")
    client.replay_file(log_path, 0, 0, 0)

    print("[INFO] Esperando a que el mundo se asiente (7 segundos)...")
    time.sleep(7) 
    
    world = client.get_world()
    
    # --- 1. BUSCAR EL VEHÍCULO ---
    ego_vehicle = None
    vehicles = world.get_actors().filter('vehicle.*')
    
    for actor in vehicles:
        if 'tesla.model3' in actor.type_id:
            ego_vehicle = actor
            break
            
    if not ego_vehicle and len(vehicles) > 0:
        ego_vehicle = vehicles[0]
            
    if not ego_vehicle:
        print("[ERROR] No se encontró ningún vehículo en el mapa.")
        return

    print(f"[INFO] Vehículo objetivo encontrado (ID: {ego_vehicle.id}). Acoplando cámara...")

    # --- 2. CONFIGURAR LA CÁMARA ---
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=0.4, y=-0.3, z=1.3))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    # --- 3. INICIAR STREAM DE VIDEO ---
    camera.listen(lambda image: process_image(image))

    print("[INFO] Reproduciendo... Presiona ESC en la ventana de video o Ctrl+C en la terminal para salir.")
    
    try:
        # --- BUCLE PRINCIPAL (El hilo principal se encarga de dibujar) ---
        while True:
            # Si hay una imagen nueva en la cola, la sacamos y la mostramos
            if not image_queue.empty():
                image_rgb = image_queue.get()
                cv2.imshow("Camara Primera Persona - Dataset DAgger", image_rgb)
            
            # cv2.waitKey(1) procesa los eventos de la ventana para que Ubuntu no crea que se congeló.
            # Además, nos permite salir limpiamente presionando la tecla ESC (código 27)
            if cv2.waitKey(1) == 27:
                print("\n[INFO] Salida solicitada con la tecla ESC.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario en la terminal.")
    finally:
        # --- 4. LIMPIEZA ---
        print("[INFO] Limpiando sensores y cerrando ventanas...")
        if 'camera' in locals():
            camera.stop()
            camera.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()