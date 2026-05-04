import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# --- Configuración ---
CSV_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260330_081603/driving_log.csv"
CSV_BALANCED_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260330_081603/driving_log_balanced.csv"
NUM_BINS = 25              # En cuántos grupos dividimos el volante (de -1 a 1)
MAX_SAMPLES_PER_BIN = 2000

def main():
    print(f"[INFO] Cargando dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    total_original = len(df)
    print(f"[INFO] Total de datos originales: {total_original}")

    # 1. Crear los "Bins" (Cajones) para agrupar los ángulos de giro
    # El volante va de -1.0 (Izquierda) a 1.0 (Derecha)
    bins = np.linspace(-1.0, 1.0, NUM_BINS)
    
    # Asignar a cada fila su cajón correspondiente según su valor de 'steer'
    df['bin'] = pd.cut(df['steer'], bins, labels=False, include_lowest=True)

    # 2. Balanceo (Undersampling)
    balanced_data = []
    
    for i in range(NUM_BINS - 1):
        # Obtener todas las filas que caen en este cajón
        bin_data = df[df['bin'] == i]
        
        # Mezclar los datos aleatoriamente
        bin_data = shuffle(bin_data)
        
        # Si el cajón tiene más datos del límite (ej. la línea recta de 0.0), lo recortamos
        if len(bin_data) > MAX_SAMPLES_PER_BIN:
            bin_data = bin_data[:MAX_SAMPLES_PER_BIN]
            
        balanced_data.append(bin_data)
        
    # Unir todos los cajones de vuelta en un solo DataFrame
    df_balanced = pd.concat(balanced_data)
    
    # Volver a mezclar todo el dataset final para que el entrenamiento sea aleatorio
    df_balanced = shuffle(df_balanced)
    
    # Quitar la columna temporal 'bin'
    df_balanced = df_balanced.drop('bin', axis=1)

    total_balanced = len(df_balanced)
    print(f"[INFO] Total después del balanceo: {total_balanced}")
    print(f"[INFO] Se eliminaron {total_original - total_balanced} ejemplos redundantes.")

    # Guardar el nuevo CSV
    df_balanced.to_csv(CSV_BALANCED_PATH, index=False)
    print(f"✅ Nuevo dataset guardado en: {CSV_BALANCED_PATH}")

    # 3. Visualización (Antes vs Después)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma Original
    axes[0].hist(df['steer'], bins=NUM_BINS, color='red', alpha=0.7)
    axes[0].axhline(MAX_SAMPLES_PER_BIN, color='black', linestyle='dashed', linewidth=2)
    axes[0].set_title('Antes del Balanceo (Sobrecarga de 0.0)')
    axes[0].set_xlabel('Ángulo de Volante')
    axes[0].set_ylabel('Cantidad de Imágenes')
    
    # Histograma Balanceado
    axes[1].hist(df_balanced['steer'], bins=NUM_BINS, color='green', alpha=0.7)
    axes[1].set_title('Después del Balanceo (Curvas Priorizadas)')
    axes[1].set_xlabel('Ángulo de Volante')
    axes[1].set_ylabel('Cantidad de Imágenes')

    plt.tight_layout()
    plt.savefig('grafica_balanceo.png')
    print("📊 Gráfica guardada como 'grafica_balanceo.png'. ¡Abrela para ver la magia!")

if __name__ == "__main__":
    main()