import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# --- Parámetros ---
CSV_IN = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/datasets_collection/dataset_extraido/driving_log.csv"          
CSV_OUT = "driving_log_balanced.csv" 
TARGET_COL = "steer"                # <-- Adaptado a tu nuevo formato de CSV
NUM_BINS = 51                       
MAX_SAMPLES_PER_BIN = 600           
# ------------------

def main():
    print(f"[INFO] Cargando dataset masivo: {CSV_IN}...")
    df = pd.read_csv(CSV_IN)
    
    # Limpieza de seguridad por si hay espacios extra en la cabecera
    df.columns = df.columns.str.strip()
    
    total_original = len(df)
    print(f"[INFO] Total de muestras originales: {total_original}")

    # 1. Crear los contenedores (bins) entre -1.0 y 1.0
    bins = np.linspace(-1.0, 1.0, NUM_BINS)
    
    # 2. Asignar cada fila a un contenedor
    df['bin'] = np.digitize(df[TARGET_COL], bins)

    balanced_df = pd.DataFrame()

    # 3. Iterar sobre cada contenedor y recortar el exceso
    for b in range(1, len(bins) + 1):
        bin_data = df[df['bin'] == b]
        
        if len(bin_data) > MAX_SAMPLES_PER_BIN:
            # Mezclamos aleatoriamente y cortamos el sobrante
            bin_data = shuffle(bin_data).iloc[:MAX_SAMPLES_PER_BIN]
            
        balanced_df = pd.concat([balanced_df, bin_data])

    # 4. Limpiar, mezclar el dataset final y guardar
    balanced_df = shuffle(balanced_df)
    balanced_df = balanced_df.drop('bin', axis=1) 
    
    balanced_df.to_csv(CSV_OUT, index=False)
    
    total_final = len(balanced_df)
    print(f"[INFO] Balanceo completado con éxito.")
    print(f"[INFO] Muestras retenidas: {total_final} (Se descartaron {total_original - total_final} frames).")
    print(f"[INFO] Guardado en: {CSV_OUT}")

    # --- 5. Visualización del Antes y Después ---
    print("[INFO] Generando gráfica de distribución...")
    plt.figure(figsize=(12, 5))
    
    # Gráfica Original
    plt.subplot(1, 2, 1)
    plt.hist(df[TARGET_COL], bins=NUM_BINS, color='red', alpha=0.7)
    plt.title('Distribución Original')
    plt.xlabel('Ángulo de Volante (steer)')
    plt.ylabel('Cantidad de Frames')
    
    # Gráfica Balanceada
    plt.subplot(1, 2, 2)
    plt.hist(balanced_df[TARGET_COL], bins=NUM_BINS, color='green', alpha=0.7)
    plt.title('Distribución Balanceada')
    plt.xlabel('Ángulo de Volante (steer)')
    plt.ylabel('Cantidad de Frames')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()