import cv2
import os
import sys

def aplicar_canny_a_carpeta(carpeta_entrada, carpeta_salida, umbral1=100, umbral2=200):
    os.makedirs(carpeta_salida, exist_ok=True)

    for archivo in os.listdir(carpeta_entrada):
        ruta_entrada = os.path.join(carpeta_entrada, archivo)

        if os.path.isfile(ruta_entrada) and archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            imagen = cv2.imread(ruta_entrada, cv2.IMREAD_GRAYSCALE)

            if imagen is None:
                print(f"No se pudo leer la imagen: {archivo}")
                continue

            bordes = cv2.Canny(imagen, umbral1, umbral2)

            ruta_salida = os.path.join(carpeta_salida, archivo)
            cv2.imwrite(ruta_salida, bordes)
            print(f"Procesada: {archivo}")

    print("✅ Proceso completado.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python canny_batch.py <carpeta_entrada> <carpeta_salida>")
        sys.exit(1)

    carpeta_entrada = sys.argv[1]
    carpeta_salida = sys.argv[2]

    aplicar_canny_a_carpeta(carpeta_entrada, carpeta_salida)
