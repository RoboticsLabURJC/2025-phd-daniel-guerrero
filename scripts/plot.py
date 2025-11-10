import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV
df = pd.read_csv("dataset/controls.csv")

# Ver las primeras filas
print(df.head())

# Crear figura
plt.figure(figsize=(12, 6))

# Graficar señales
plt.plot(df["steer"], label="Steer", alpha=0.8)
plt.plot(df["throttle"], label="Throttle", alpha=0.8)
plt.plot(df["brake"], label="Brake", alpha=0.8)

plt.title("Comportamiento de controles del vehículo")
plt.xlabel("Frame / Tiempo")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()
