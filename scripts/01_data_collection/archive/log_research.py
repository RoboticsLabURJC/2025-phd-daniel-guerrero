import carla

def main():
    client = carla.Client('localhost', 2000)
    log_path = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_crudos/recording_20260222_200431.log"
    
    print("[INFO] Leyendo el interior del archivo de grabación...")
    # El 'True' le dice que imprima todos los detalles de los actores
    info = client.show_recorder_file_info(log_path, True)
    
    # Guardamos el resultado en un txt para que lo leas más cómodo
    with open("contenido_del_log.txt", "w") as f:
        f.write(info)
        
    print("[ÉXITO] Se generó el archivo 'contenido_del_log.txt'. Ábrelo y busca qué vehículos hay.")

if __name__ == '__main__':
    main()