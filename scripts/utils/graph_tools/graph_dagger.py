#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (activa proyección 3D)


def main():
    parser = argparse.ArgumentParser(description="Histograma 3D (CARLA labels.csv): expert_steer vs expert_throttle")
    parser.add_argument("--csv", required=True, help="Ruta a labels.csv")
    parser.add_argument("--steer_col", default="expert_steer", help="Columna de steer (default expert_steer)")
    parser.add_argument("--throttle_col", default="expert_throttle", help="Columna de throttle (default expert_throttle)")
    parser.add_argument("--scale_throttle_to_01", action="store_true",
                        help="Escala throttle a [0,1] asumiendo max=0.25 (throttle/0.25)")
    parser.add_argument("--throttle_max", type=float, default=0.25,
                        help="Máximo esperado de throttle (default 0.25). Solo usado si --scale_throttle_to_01")
    args = parser.parse_args()

    # === 1) Cargar datos =====================================================
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip().str.lower()

    steer_col = args.steer_col.strip().lower()
    thr_col = args.throttle_col.strip().lower()

    if steer_col not in df.columns:
        raise KeyError("No existe columna '{}'. Columnas: {}".format(steer_col, list(df.columns)))
    if thr_col not in df.columns:
        raise KeyError("No existe columna '{}'. Columnas: {}".format(thr_col, list(df.columns)))

    steer = df[steer_col].to_numpy(dtype=np.float32)
    throttle = df[thr_col].to_numpy(dtype=np.float32)

    # Recortar por seguridad
    steer = np.clip(steer, -1.0, 1.0)

    if args.scale_throttle_to_01:
        # throttle en labels suele estar 0..0.25 -> lo mapeamos a 0..1
        if args.throttle_max <= 0:
            raise ValueError("--throttle_max debe ser > 0")
        throttle = throttle / float(args.throttle_max)
        throttle = np.clip(throttle, 0.0, 1.0)
        throttle_label = "Throttle (escalado 0 a 1)"
        throttle_range = (0.0, 1.0)
        throttle_bins = np.linspace(0.0, 1.0, 7)  # 6 bins
    else:
        # throttle real (ej. 0..0.25)
        thr_min = float(np.nanmin(throttle))
        thr_max = float(np.nanmax(throttle))
        # si el rango es chico, igual lo recortamos a 0..max observado
        throttle = np.clip(throttle, 0.0, thr_max if thr_max > 0 else 1.0)
        throttle_label = "Throttle real"
        throttle_range = (0.0, thr_max if thr_max > 0 else 1.0)
        # bins adaptados al rango real
        throttle_bins = np.linspace(throttle_range[0], throttle_range[1], 7)

    # === 2) Definir bins y calcular histograma 2D ============================
    steer_bins = np.linspace(-1.0, 1.0, 9)  # 8 bins

    H, xedges, yedges = np.histogram2d(
        steer, throttle, bins=[steer_bins, throttle_bins]
    )

    # Centros de cada bin
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # Tamaños de cada bin (dx, dy)
    dx = np.diff(xedges)
    dy = np.diff(yedges)

    # Crear grilla para bar3d
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    xpos = xx.ravel()
    ypos = yy.ravel()
    zpos = np.zeros_like(xpos)

    dz = H.T.ravel()  # transpose para alinear ejes
    dx_grid, dy_grid = np.meshgrid(dx, dy, indexing="xy")
    dxr = dx_grid.ravel()
    dyr = dy_grid.ravel()

    # === 3) Colorear por altura =============================================
    cmap = mpl.colormaps["viridis"]
    vmax = dz.max() if dz.max() > 0 else 1
    norm = mpl.colors.Normalize(vmin=dz.min(), vmax=vmax)
    bar_colors = cmap(norm(dz))

    # === 4) Graficar 3D ======================================================
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(
        xpos, ypos, zpos,
        dxr, dyr, dz,
        shade=True,
        color=bar_colors,
        edgecolor="none",
        alpha=0.95,
    )

    ax.set_xlabel("Steer (-1 a 1)", labelpad=10)
    ax.set_ylabel(throttle_label, labelpad=10)
    ax.set_zlabel("Número de muestras", labelpad=10)

    title = "Distribución 3D: {} vs {}".format(steer_col, thr_col)
    if args.scale_throttle_to_01:
        title += " (throttle escalado)"
    ax.set_title(title, pad=15)

    # ticks (steer fijo, throttle depende del modo)
    ax.set_xticks(np.linspace(-1.0, 1.0, 9))
    ax.set_yticks(np.linspace(throttle_range[0], throttle_range[1], 6))

    ax.view_init(elev=22, azim=-55)

    try:
        ax.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass

    # === 5) Colorbar =========================================================
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(dz)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Número de muestras")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
