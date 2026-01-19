import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar CSV
df = pd.read_csv("dataset/controls.csv")

# Crear figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar ejes
ax.scatter(df["steer"], df["throttle"], df["brake"],
           c=df["steer"], cmap='coolwarm', s=10, alpha=0.7)

# Etiquetas
ax.set_xlabel("Steer")
ax.set_ylabel("Throttle")
ax.set_zlabel("Brake")
ax.set_title("Distribución 3D de los controles del vehículo")

plt.show()
