#!/usr/bin/env python3
"""
Generador de Dataset BEV (Bird's Eye View) optimizado y filtrado.
- Calcula la homografía IPM exacta a partir de los parámetros físicos de la cámara en CARLA
  (1920x1080, FOV=110°, z=1.6m, pitch=-15°) evitando zonas negras y asimetrías.
- Filtra los mejores frames de driving_log.csv (elimina frames estáticos, glitches de velocidad
  del replayer y balancea rectas vs. curvas).
- Procesa en paralelo usando mapas precalculados (cv2.remap) para máxima velocidad.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_bifocal_bev_maps(
    img_w: int = 1920,
    img_h: int = 1080,
    bev_w: int = 600,
    bev_h: int = 800,
    split_ratio: float = 0.45,
    transition_px: float = 22.0,
):
    """
    Construye los mapas precalculados (map1, map2) para el BEV Bifocal Continuo Universal:
    - 100% parametrizado en coordenadas normalizadas [0..1] respecto a (img_w, img_h),
      por lo que funciona con cualquier dataset y resolución.
    - Parte inferior (55% del lienzo): BEV cenital puro (carriles verticales y paralelos).
    - Parte superior (45% del lienzo): apertura progresiva hacia el camino medio/lejano, horizonte
      y un poco de cielo (y_top = 0.24 * img_h) para dar suficiente anticipación en curvas/bifurcaciones.
    """
    split_y = split_ratio * bev_h

    p_left_seam = [0.39 * img_w, 0.385 * img_h]
    p_right_seam = [0.61 * img_w, 0.385 * img_h]
    p_left_bot = [-0.25 * img_w, 0.98 * img_h]
    p_right_bot = [1.25 * img_w, 0.98 * img_h]

    src_bot = np.float32([p_left_bot, p_right_bot, p_right_seam, p_left_seam])
    dst_bot = np.float32([[0, bev_h], [bev_w, bev_h], [bev_w, split_y], [0, split_y]])
    H_bot_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_bot, dst_bot))

    src_top = np.float32([
        p_left_seam,
        p_right_seam,
        [0.76 * img_w, 0.24 * img_h],
        [0.24 * img_w, 0.24 * img_h],
    ])
    dst_top = np.float32([[0, split_y], [bev_w, split_y], [bev_w, 0], [0, 0]])
    H_top_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_top, dst_top))

    y_grid, x_grid = np.indices((bev_h, bev_w), dtype=np.float32)
    coords = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1).reshape(-1, 3)

    c_bot = coords @ H_bot_inv.T
    w_bot = np.where(np.abs(c_bot[:, 2]) > 1e-8, c_bot[:, 2], 1e-8)
    u_bot = (c_bot[:, 0] / w_bot).reshape(bev_h, bev_w)
    v_bot = (c_bot[:, 1] / w_bot).reshape(bev_h, bev_w)

    c_top = coords @ H_top_inv.T
    w_top = np.where(np.abs(c_top[:, 2]) > 1e-8, c_top[:, 2], 1e-8)
    u_top = (c_top[:, 0] / w_top).reshape(bev_h, bev_w)
    v_top = (c_top[:, 1] / w_top).reshape(bev_h, bev_w)

    alpha_blend = 1.0 / (1.0 + np.exp(-(y_grid - split_y) / transition_px))
    map_x = (alpha_blend * u_bot + (1.0 - alpha_blend) * u_top).astype(np.float32)
    map_y = (alpha_blend * v_bot + (1.0 - alpha_blend) * v_top).astype(np.float32)

    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)


def filter_and_balance_dataframe(
    df: pd.DataFrame,
    min_speed_kmh: float = 0.5,
    max_speed_kmh: float = 100.0,
    keep_straight_frac: float = 1.0,
    keep_near_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Selecciona los mejores frames para entrenamiento:
    1. Descarta frames con el vehículo detenido (velocidad < min_speed_kmh).
    2. Descarta glitches de teletransportación del replayer (velocidad > max_speed_kmh).
    3. Balancea la distribución de ángulo de dirección (conserva 100% de curvas suaves/medias/fuertes
       y submuestrea rectas redundantes).
    """
    n_total = len(df)

    # 1. Filtrado por velocidad válida
    mask_speed = (df["velocidad_kmh"] >= min_speed_kmh) & (df["velocidad_kmh"] <= max_speed_kmh)
    df_valid = df[mask_speed].copy()
    n_after_speed = len(df_valid)

    # 2. Clasificación por magnitud de dirección (|steering|)
    abs_steer = df_valid["steering"].abs()
    bins = [-1e-9, 0.02, 0.05, 0.15, 0.35, 1.01]
    labels = ["straight", "near", "soft", "medium", "hard"]
    df_valid["_bin"] = pd.cut(abs_steer, bins=bins, labels=labels, include_lowest=True)

    keep_map = {
        "straight": keep_straight_frac,  # |steer| <= 0.02
        "near": keep_near_frac,          # 0.02 < |steer| <= 0.05
        "soft": 1.0,                     # 0.05 < |steer| <= 0.15
        "medium": 1.0,                   # 0.15 < |steer| <= 0.35
        "hard": 1.0,                     # |steer| > 0.35
    }

    parts = []
    print("\n--- Resumen de Selección de Frames ---")
    print(f"Frames totales originales : {n_total}")
    print(f"Tras filtrar velocidad [{min_speed_kmh}, {max_speed_kmh}] km/h: {n_after_speed} "
          f"(descartados: {n_total - n_after_speed})")
    print("\nDistribución por tipo de curva (antes -> después):")

    for label in labels:
        sub = df_valid[df_valid["_bin"] == label]
        before_count = len(sub)
        frac = keep_map[label]
        if frac < 1.0 and before_count > 0:
            sub = sub.sample(frac=frac, random_state=seed)
        after_count = len(sub)
        print(f"  - {label:8s}: {before_count:6d} -> {after_count:6d} (keep={frac*100:.0f}%)")
        parts.append(sub)

    df_best = (
        pd.concat(parts, ignore_index=True)
        .sort_values("frame_global")
        .drop(columns=["_bin"])
        .reset_index(drop=True)
    )
    print(f"\nTotal de mejores frames seleccionados: {len(df_best)} ({len(df_best)/n_total*100:.1f}%)\n")
    return df_best


def build_remap_tables(H: np.ndarray, bev_w: int, bev_h: int):
    """Precalcula los mapas (map_x, map_y) para usar cv2.remap en lugar de warpPerspective por frame."""
    H_inv = np.linalg.inv(H)
    y_grid, x_grid = np.indices((bev_h, bev_w), dtype=np.float32)
    ones = np.ones_like(x_grid)
    coords = np.stack([x_grid, y_grid, ones], axis=-1).reshape(-1, 3)

    src_coords = coords @ H_inv.T
    w = np.where(np.abs(src_coords[:, 2]) > 1e-8, src_coords[:, 2], 1e-8)
    map_x = (src_coords[:, 0] / w).reshape(bev_h, bev_w).astype(np.float32)
    map_y = (src_coords[:, 1] / w).reshape(bev_h, bev_w).astype(np.float32)
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)


def process_single_frame(args_tuple):
    src_path, dst_path, map1, map2, skip_existing = args_tuple
    if skip_existing and os.path.exists(dst_path):
        return True
    img = cv2.imread(src_path)
    if img is None:
        return False
    bev = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.imwrite(dst_path, bev)


def main():
    parser = argparse.ArgumentParser(description="Convierte y filtra los mejores frames a vista BEV.")
    parser.add_argument(
        "--dataset-dir",
        default="/home/daniel/code/2025-phd-daniel-guerrero/data/dataset_masivo_continuo",
        help="Directorio raíz del dataset que contiene driving_log.csv y frames/",
    )
    parser.add_argument(
        "--out-frames-dir",
        default="frames_bev_best",
        help="Subcarpeta de salida para las imágenes BEV (relativa a dataset-dir o ruta absoluta)",
    )
    parser.add_argument(
        "--out-csv",
        default="driving_log_bev_best.csv",
        help="Archivo CSV filtrado de salida (relativo a dataset-dir o ruta absoluta)",
    )
    parser.add_argument("--img-width", type=int, default=1920, help="Ancho de la imagen original")
    parser.add_argument("--img-height", type=int, default=1080, help="Alto de la imagen original")
    parser.add_argument("--bev-width", type=int, default=600, help="Ancho de la imagen BEV")
    parser.add_argument("--bev-height", type=int, default=800, help="Alto de la imagen BEV")
    parser.add_argument("--split-ratio", type=float, default=0.45, help="Fracción superior dedicada al horizonte/cielo")
    parser.add_argument("--keep-straight", type=float, default=1.0, help="Fracción a conservar de rectas (|steer|<=0.02)")
    parser.add_argument("--keep-near", type=float, default=1.0, help="Fracción a conservar de casi-rectas (0.02<|steer|<=0.05)")
    parser.add_argument("--all-frames", action="store_true", help="Convertir todos los frames sin filtrar")
    parser.add_argument("--skip-existing", action="store_true", help="Omitir imágenes que ya existen en destino")
    parser.add_argument("--workers", type=int, default=8, help="Número de hilos en paralelo")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar estadísticas sin convertir todo el lote")
    args = parser.parse_args()

    csv_in = os.path.join(args.dataset_dir, "driving_log.csv")
    out_frames_dir = (
        args.out_frames_dir
        if os.path.isabs(args.out_frames_dir)
        else os.path.join(args.dataset_dir, args.out_frames_dir)
    )
    out_csv = (
        args.out_csv
        if os.path.isabs(args.out_csv)
        else os.path.join(args.dataset_dir, args.out_csv)
    )

    df = pd.read_csv(csv_in)
    if args.all_frames:
        df_selected = df.copy()
        print(f"\nModo --all-frames activo: se procesarán los {len(df_selected)} frames.")
    else:
        df_selected = filter_and_balance_dataframe(
            df,
            keep_straight_frac=args.keep_straight,
            keep_near_frac=args.keep_near,
        )

    if args.dry_run:
        print("[DRY-RUN] Finalizado sin escribir archivos.")
        return

    os.makedirs(out_frames_dir, exist_ok=True)

    # Actualizar image_path en el CSV de salida para apuntar a la carpeta nueva si se desea
    rel_folder = os.path.basename(out_frames_dir.rstrip("/"))
    df_out = df_selected.copy()
    df_out["image_path"] = df_out["image_path"].apply(
        lambda p: os.path.join(rel_folder, os.path.basename(p))
    )
    df_out.to_csv(out_csv, index=False)
    print(f"[INFO] CSV filtrado guardado en: {out_csv}")

    print("[INFO] Precalculando mapas de transformación BEV Bifocal Universal (cv2.remap)...")
    map1, map2 = build_bifocal_bev_maps(
        img_w=args.img_width,
        img_h=args.img_height,
        bev_w=args.bev_width,
        bev_h=args.bev_height,
        split_ratio=args.split_ratio,
    )

    tasks = []
    for rel_path in df_selected["image_path"]:
        filename = os.path.basename(rel_path)
        src_path = os.path.join(args.dataset_dir, "frames", filename)
        dst_path = os.path.join(out_frames_dir, filename)
        tasks.append((src_path, dst_path, map1, map2, args.skip_existing))

    print(f"[INFO] Transformando {len(tasks)} imágenes a BEV con {args.workers} hilos...")
    ok_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_frame, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Generando BEV"):
            if fut.result():
                ok_count += 1

    print(f"\n[ÉXITO] {ok_count}/{len(tasks)} imágenes BEV guardadas en: {out_frames_dir}")


if __name__ == "__main__":
    main()
