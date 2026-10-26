#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # necesario para activar proyección 3D


def main():
    # === 1) Cargar datos =====================================================
    csv_path = "dataset/controls_balanced.csv"  # cambia el nombre si tu archivo se llama distinto
    df = pd.read_csv(csv_path)

    # Asumimos columnas 'steer' y 'throttle'
    steer = df["steer"].to_numpy()
    throttle = df["throttle"].to_numpy()

    # Recortar por seguridad (rango típico)
    steer = np.clip(steer, -1.0, 1.0)
    throttle = np.clip(throttle, 0.0, 1.0)

    # === 2) Definir bins y calcular histograma 2D ============================
    # Ajusta la cantidad de bins según el nivel de detalle que quieras
    steer_bins = np.linspace(-1.0, 1.0, 9)    # 8 bins en X
    throttle_bins = np.linspace(0.0, 1.0, 7)  # 6 bins en Y

    H, xedges, yedges = np.histogram2d(
        steer, throttle, bins=[steer_bins, throttle_bins]
    )

    # Centros de cada bin
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # Tamaños de cada bin (dx, dy)
    dx = np.diff(xedges)
    dy = np.diff(yedges)

    # Crear la grilla de posiciones para bar3d
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    xpos = xx.ravel()
    ypos = yy.ravel()
    zpos = np.zeros_like(xpos)

    # Alturas = conteos del histograma
    dz = H.T.ravel()  # importante transponer para que coincidan los ejes

    # Repetir dx, dy para cada barra
    dx_grid, dy_grid = np.meshgrid(dx, dy, indexing="xy")
    dxr = dx_grid.ravel()
    dyr = dy_grid.ravel()

    # === 3) Colorear por altura (conteo) =====================================
    # Usar la nueva API de colormaps (evita DeprecationWarning)
    cmap = mpl.colormaps["viridis"]
    # Evitar que todo sea negro si dz.max() == 0
    vmax = dz.max() if dz.max() > 0 else 1
    norm = mpl.colors.Normalize(vmin=dz.min(), vmax=vmax)
    bar_colors = cmap(norm(dz))

    # === 4) Graficar en 3D ===================================================
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

    # Etiquetas y título
    ax.set_xlabel("Steer (-1 a 1)", labelpad=10)
    ax.set_ylabel("Throttle (0 a 1)", labelpad=10)
    ax.set_zlabel("Número de muestras", labelpad=10)
    ax.set_title("Distribución 3D de Steer vs Throttle", pad=15)

    # Ticks aproximados como en tu captura
    ax.set_xticks(np.linspace(-0.6, 0.6, 7))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    # Vista parecida a la que mostraste en la imagen
    ax.view_init(elev=22, azim=-55)

    # Ajuste de proporciones (si tu versión de Matplotlib lo soporta)
    try:
        ax.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass

    # === 5) Colorbar =========================================================
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(dz)

    # Ojo: pasamos ax=ax para que sepa de qué eje robar espacio
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Número de muestras")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
