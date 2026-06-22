import carla
import time
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
TOWNS = ["Town03", "Town04", "Town05"] # Añade los que confirmaste
TIME_PER_TOWN = 30 * 60  
LOG_DIR = os.path.abspath("./logs_crudos")
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)

    for town_name in TOWNS:
        vehicle = None
        try:
            print(f"\n[INFO] --- Iniciando fase: {town_name} ---")
            world = client.load_world(town_name)
            
            # Pausa breve para que el mapa se asiente
            time.sleep(3) 
            
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            vehicle_bp.set_attribute('role_name', 'ego')
            
            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                print(f"[ERROR] No hay puntos de spawn en {town_name}")
                continue
                
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
            vehicle.set_autopilot(True)
            
            # Iniciar Grabación
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"recording_{town_name}_{timestamp}.log"
            log_path = os.path.join(LOG_DIR, log_name)
            
            client.start_recorder(log_path)
            print(f"[INFO] Grabando en: {log_name}")

            start_time = time.time()
            while time.time() - start_time < TIME_PER_TOWN:
                time.sleep(10)
                elapsed = int(time.time() - start_time)
                print(f"  -> {town_name}: {elapsed//60}min / 30min", end='\r')

        except Exception as e:
            print(f"\n[ERROR] Interrupción en {town_name}: {e}")
        
        finally:
            # LIMPIEZA ORDENADA: Primero paramos recorder, luego actor
            print(f"\n[INFO] Limpiando {town_name}...")
            client.stop_recorder()
            if vehicle is not None:
                try:
                    vehicle.set_autopilot(False)
                    vehicle.destroy()
                except:
                    pass # Ya fue destruido por el cambio de mundo
            print(f"[INFO] Finalizada fase de {town_name}.")
            time.sleep(2) # Respiro para el Core de CARLA

    print("\n[ÉXITO] Sesión multiverso completada.")

if __name__ == "__main__":
    main()