import cv2
import numpy as np
import os
import glob

# 1. Configuración de rutas (Ajusta estas rutas según tu sistema)
INPUT_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/Homograficas/dataset_masivo_continuo/frames"  
OUTPUT_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/Homograficas/dataset_masivo_continuo/frames_homographic"

# Crear la carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Matriz de Homografía H (Calculada previamente)
H_MATRIX = np.array([
    [-1.38322388e-01, -8.80401482e-01,  4.42338059e+02],
    [-4.51221359e-02, -2.59903503e+00,  1.07695709e+03],
    [-5.42655982e-05, -2.67570238e-03,  1.00000000e+00]
])

# Tamaño de la imagen resultante
BEV_WIDTH = 600
BEV_HEIGHT = 800

# 3. Procesamiento en lote
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
total_images = len(image_paths)

if total_images == 0:
    print(f"Error: No se encontraron imágenes .png en {INPUT_DIR}")
    exit()

print(f"Iniciando transformación de {total_images} imágenes a Vista Cenital (BEV)...")

for i, img_path in enumerate(image_paths):
    # Extraer solo el nombre del archivo (ej. "00000035.png")
    filename = os.path.basename(img_path)
    
    # Leer la imagen
    img = cv2.imread(img_path)
    if img is None:
        print(f"Advertencia: No se pudo leer {filename}")
        continue
        
    # --- MÁSCARA DEL HORIZONTE (Opcional pero recomendada) ---
    # Pinta de negro la mitad superior de la imagen original (el cielo/edificios) 
    # antes de transformarla para evitar que se distorsionen en la vista BEV.
    # Si ves que recorta mucho asfalto, reduce el 450. Si sale cielo, auméntalo.
    img[:450, :] = 0  
    # ---------------------------------------------------------
    
    # Aplicar la matriz de transformación
    bev_image = cv2.warpPerspective(img, H_MATRIX, (BEV_WIDTH, BEV_HEIGHT))
    
    # Guardar en la nueva carpeta con el mismo nombre
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, bev_image)
    
    # Barra de progreso simple en consola
    if (i + 1) % 500 == 0 or (i + 1) == total_images:
        porcentaje = ((i + 1) / total_images) * 100
        print(f"Progreso: {i + 1}/{total_images} [{porcentaje:.1f}%]")

print(f"\n¡Proceso completado! Las imágenes BEV se encuentran en: {OUTPUT_DIR}")