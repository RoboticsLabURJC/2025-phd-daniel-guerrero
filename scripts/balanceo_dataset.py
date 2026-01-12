import pandas as pd

# Cargar el CSV
df = pd.read_csv('dataset/controls.csv')

# Filtro para rectas
mask_straight = df['steer'].abs() < 0.05

# Mantener solo el 15% de rectas
df_straight = df[mask_straight].sample(frac=0.15, random_state=42)

# Mantener todas las curvas
df_curves = df[~mask_straight]

# Combinar
df_balanced = pd.concat([df_straight, df_curves]).sample(frac=1)

df_balanced.to_csv('dataset/controls_balanced.csv', index=False)
print("Nuevo tamaño:", len(df_balanced))
