import cv2
import numpy as np

# Variables globales para almacenar las coordenadas de los clics
src_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < 4:
            src_points.append([x, y])
            print(f"Punto guardado: ({x}, {y})")
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Paso 1: Selecciona 4 puntos", img_copy)

# Cargar la imagen (asegúrate de que "00000035.jpg" esté en el mismo directorio)
image_path = "00000035.png"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: No se pudo cargar la imagen {image_path}")
    exit()

img_copy = img.copy()

print("Haz clic en 4 puntos del asfalto que formen un rectángulo en la vida real.")
print("ORDEN ESTRICTO:")
print("1. Inferior Izquierdo")
print("2. Inferior Derecho")
print("3. Superior Derecho")
print("4. Superior Izquierdo")

cv2.imshow("Paso 1: Selecciona 4 puntos", img_copy)
cv2.setMouseCallback("Paso 1: Selecciona 4 puntos", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(src_points) == 4:
    # Convertir las coordenadas de origen a un array de numpy
    pts_src = np.array(src_points, dtype=np.float32)

    # Definir el tamaño del lienzo para la vista cenital (Bird's Eye View)
    width, height = 600, 800
    
    # Definir los 4 puntos de destino formando un rectángulo vertical perfecto
    # Ajusta el valor 'offset' (150) si el carril queda muy ancho o estrecho
    offset_x = 150
    pts_dst = np.array([
        [offset_x, height],               # 1. Inferior Izquierdo
        [width - offset_x, height],       # 2. Inferior Derecho
        [width - offset_x, 0],            # 3. Superior Derecho
        [offset_x, 0]                     # 4. Superior Izquierdo
    ], dtype=np.float32)

    # Calcular la matriz de Homografía H
    H, status = cv2.findHomography(pts_src, pts_dst)
    
    # Imprimir la matriz para que puedas copiarla y usarla en tu dataset masivo
    print("\n--- MATRIZ DE HOMOGRAFÍA H ---")
    print(repr(H))
    
    # Aplicar la matriz a la imagen original
    bev_image = cv2.warpPerspective(img, H, (width, height))

    # Mostrar la imagen resultante
    cv2.imshow("Paso 2: Vista Cenital (BEV)", bev_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Operación cancelada: No seleccionaste los 4 puntos.")