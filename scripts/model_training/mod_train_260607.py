import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- CONFIGURACIÓN ---
DATASET_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_masivo_20260606_152055"
CSV_FILE = os.path.join(DATASET_DIR, "driving_log.csv")
MODEL_SAVE_PATH = "pilotnet_model_260607.pth"

# Hiperparámetros optimizados para hardware de gama alta (Mucha RAM/VRAM y núcleos)
BATCH_SIZE = 128      
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_WORKERS = 8       

# Configuración del dispositivo (Busca ROCm/CUDA, si no, usa CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo de entrenamiento: {DEVICE}")

# --- 1. DEFINICIÓN DEL DATASET ---
class CarlaDrivingDataset(Dataset):
    def __init__(self, data_list, dataset_dir):
        self.data_list = data_list
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        row = self.data_list[idx]
        
        # El CSV tiene: [frame_global, throttle, steering, image_path, source_log]
        steering = float(row[2])
        img_rel_path = row[3]
        
        # Cargar imagen
        img_path = os.path.join(self.dataset_dir, img_rel_path)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        
        # Convertir de BGR (OpenCV) a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- PREPROCESAMIENTO CLAVE PARA PILOTNET ---
        # 1. Recortar la imagen (Cortamos el cielo/árboles y el capó del auto)
        # La imagen original es 800x600. Nos quedamos con la porción Y: [250 a 500]
        image = image[250:500, :, :] 
        
        # 2. Redimensionar al estándar de NVIDIA PilotNet (200x66)
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
        
        # 3. Normalizar y convertir a Tensor PyTorch (Formato: Canales, Alto, Ancho)
        image = image / 255.0 # Normalizar entre 0 y 1
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        steering_tensor = torch.tensor([steering], dtype=torch.float32)
        return image, steering_tensor

# --- 2. ARQUITECTURA PILOTNET (NVIDIA) ---
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
        # Capas Convolucionales para extracción de características
        self.conv_layers = nn.Sequential(
            # Input: 3 canales, Output: 24, Kernel: 5x5, Stride: 2
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU()
        )
        
        # Capas Densas (Completamente conectadas) para la regresión del mando
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100), # El tamaño aplanado depende de la entrada 200x66
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1) # Salida única: Ángulo del volante (Steering)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Aplanar (Flatten)
        x = self.linear_layers(x)
        return x

# --- 3. BUCLE PRINCIPAL DE ENTRENAMIENTO ---
def main():
    print("[INFO] Cargando datos del CSV...")
    data_list = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Saltar cabecera
        for row in reader:
            data_list.append(row)
            
    print(f"[INFO] Total de muestras encontradas: {len(data_list)}")

    # Dividir datos en 80% Entrenamiento y 20% Validación
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    print(f"[INFO] Set de Entrenamiento: {len(train_data)} | Set de Validación: {len(val_data)}")

    # Crear DataLoaders
    train_dataset = CarlaDrivingDataset(train_data, DATASET_DIR)
    val_dataset = CarlaDrivingDataset(val_data, DATASET_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Inicializar Modelo, Función de Pérdida (MSE) y Optimizador (Adam)
    model = PilotNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n[INFO] Iniciando Entrenamiento...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        running_loss = 0.0
        
        for i, (images, steerings) in enumerate(train_loader):
            images, steerings = images.to(DEVICE), steerings.to(DEVICE)
            
            optimizer.zero_grad() # Limpiar gradientes
            
            outputs = model(images) # Predicción
            loss = criterion(outputs, steerings) # Calcular error
            loss.backward() # Retropropagación
            optimizer.step() # Actualizar pesos
            
            running_loss += loss.item()
            
            # Imprimir progreso cada 50 lotes
            if i % 50 == 0:
                print(f"  -> Época [{epoch+1}/{EPOCHS}], Lote [{i}/{len(train_loader)}], Pérdida: {loss.item():.4f}", end='\r')

        train_loss = running_loss / len(train_loader)

        # --- FASE DE VALIDACIÓN ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # No calcular gradientes para ahorrar memoria y tiempo
            for images, steerings in val_loader:
                images, steerings = images.to(DEVICE), steerings.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, steerings)
                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)
        
        print(f"\n[Resumen Época {epoch+1}] Pérdida Entrenamiento: {train_loss:.4f} | Pérdida Validación: {val_loss:.4f}")

        # Guardar el modelo si mejora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [*] ¡Nuevo mejor modelo guardado en {MODEL_SAVE_PATH}!")

    print("\n[ÉXITO] Entrenamiento completado. Modelo listo para probarse en CARLA.")

if __name__ == '__main__':
    main()