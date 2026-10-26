import carla
import os

def check_log_duration(log_filename):
    # 1. Conectar con el simulador
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    # 2. Obtener la ruta absoluta (CARLA requiere la ruta absoluta del archivo)
    log_path = os.path.abspath(log_filename)

    try:
        # 3. Extraer la información
        # El 'False' al final es vital: le dice a CARLA que SOLO te dé el resumen
        # y no imprima la información de cada uno de los miles de frames.
        info = client.show_recorder_file_info(log_path, False)
        print(info)
        
    except RuntimeError as e:
        print(f"Error al leer el archivo: {e}")

# Cambia esto por el nombre de tu archivo log
check_log_duration("/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_crudos/recording_20260308_205419.log")