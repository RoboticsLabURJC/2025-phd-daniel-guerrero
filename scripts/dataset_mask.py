# crop_bottom_half_dataset.py
import os
import cv2

# Carpetas de entrada y salida
INPUT_DIR = "dataset_ego/images"          # aquí tienes tus imágenes originales
OUTPUT_DIR = "dataset/images_bottom"  # aquí se guardarán las recortadas

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_image_file(filename):
    ext = filename.lower().split(".")[-1]
    return ext in ["jpg", "jpeg", "png", "bmp"]

def crop_bottom_half(img):
    h, w = img.shape[:2]
    return img[h // 2 : h, 0 : w]  # mitad inferior

def main():
    ensure_dir(OUTPUT_DIR)

    files = sorted(os.listdir(INPUT_DIR))
    total = 0

    for fname in files:
        if not is_image_file(fname):
            continue

        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        img = cv2.imread(in_path)
        if img is None:
            print(f"[WARN] No se pudo leer: {in_path}")
            continue

        cropped = crop_bottom_half(img)
        ok = cv2.imwrite(out_path, cropped)
        if not ok:
            print(f"[WARN] No se pudo guardar: {out_path}")
            continue

        total += 1

    print(f"[INFO] Imágenes procesadas: {total}")
    print(f"[INFO] Guardadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
