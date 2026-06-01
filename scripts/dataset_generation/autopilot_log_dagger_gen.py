import carla
import time
import os
import random
from datetime import datetime

# --- CONFIGURACIÓN --- #
TOWNS = ["Town12", "Town13", "Town15"] 
SIM_TIME_PER_TOWN = 30 * 60  
LOG_DIR = os.path.abspath("./logs_crudos")
os.makedirs(LOG_DIR, exist_ok=True)

# Configuración de DAgger
PERTURBATION_INTERVAL = 10.0 
SHIFT_DISTANCE = 1.5         

# Configuración del Simulador
FPS = 20
FIXED_DELTA_SECONDS = 1.0 / FPS

def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)

    for town_name in TOWNS:
        vehicle = None
        collision_sensor = None
        has_collided = False # Bandera para detectar choques

        # Función callback que se ejecuta automáticamente al chocar
        def collision_handler(event):
            nonlocal has_collided
            has_collided = True
            actor_name = event.other_actor.type_id
            print(f"\n[ALERTA - CHOQUE] El vehículo impactó contra: {actor_name}")

        try:
            print(f"\n[INFO] --- Iniciando fase: {town_name} ---")
            world = client.load_world(town_name)
            
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
            world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            
            blueprint_library = world.get_blueprint_library()
            
            # 1. Spawn del Vehículo
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            vehicle_bp.set_attribute('role_name', 'ego')
            spawn_points = world.get_map().get_spawn_points()
            vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
            vehicle.set_autopilot(True, traffic_manager.get_port())
            
            # 2. Spawn del Sensor de Colisión (Atachado al vehículo)
            collision_bp = blueprint_library.find('sensor.other.collision')
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
            # Escuchamos los eventos de choque
            collision_sensor.listen(lambda event: collision_handler(event))
            
            # Iniciar Grabación
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"recording_{town_name}_{timestamp}.log"
            log_path = os.path.join(LOG_DIR, log_name)
            client.start_recorder(log_path)
            print(f"[INFO] Grabando en: {log_name}")

            world.tick() 
            sim_start_time = world.get_snapshot().timestamp.elapsed_seconds
            last_perturbation = sim_start_time
            shift_direction = 1 

            print("[INFO] Simulación corriendo... Sensor de colisiones ACTIVO.")

            while True:
                world.tick() 
                
                current_sim_time = world.get_snapshot().timestamp.elapsed_seconds
                elapsed_sim = current_sim_time - sim_start_time
                
                if elapsed_sim >= SIM_TIME_PER_TOWN:
                    break 
                
                # --- MANEJO DE COLISIONES ---
                if has_collided:
                    print("[INFO] Reubicando el vehículo para evitar grabar datos corruptos...")
                    # Reiniciamos el auto en un punto aleatorio nuevo
                    new_spawn = random.choice(spawn_points)
                    vehicle.set_transform(new_spawn)
                    
                    # Reseteamos variables
                    has_collided = False
                    last_perturbation = current_sim_time # Reiniciamos el reloj de saltos
                    continue # Saltamos el resto del ciclo para que se asiente
                # ----------------------------

                # --- LÓGICA DE PERTURBACIÓN (DAgger) ---
                if current_sim_time - last_perturbation >= PERTURBATION_INTERVAL:
                    transform = vehicle.get_transform()
                    right_vector = transform.get_right_vector()
                    
                    transform.location.x += right_vector.x * SHIFT_DISTANCE * shift_direction
                    transform.location.y += right_vector.y * SHIFT_DISTANCE * shift_direction
                    transform.location.z += 0.1 
                    
                    vehicle.set_transform(transform)
                    
                    side_str = "Derecha" if shift_direction == 1 else "Izquierda"
                    print(f"[DAgger - SimTime: {elapsed_sim:.1f}s] Perturbación a la {side_str}")
                    
                    shift_direction *= -1
                    last_perturbation = current_sim_time

        except Exception as e:
            print(f"\n[ERROR] Interrupción en {town_name}: {e}")
        
        finally:
            print(f"\n[INFO] Limpiando {town_name}...")
            client.stop_recorder()
            
            # Limpiamos los sensores también
            if collision_sensor is not None:
                collision_sensor.destroy()
            if vehicle is not None:
                try:
                    vehicle.set_autopilot(False)
                    vehicle.destroy()
                except:
                    pass
            
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(False)
            
            print(f"[INFO] Finalizada fase de {town_name}.")
            time.sleep(1) 

    print("\n[ÉXITO] Sesión multiverso completada con detección de choques.")

if __name__ == "__main__":
    main()