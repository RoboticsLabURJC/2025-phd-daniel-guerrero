import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------- Parámetros de Entrenamiento ----------
DATA_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido"
CSV_FILE = os.path.join(DATA_DIR, "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido/driving_log.csv")
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4

# Detectar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Entrenando en: {device}")
# -----------------------------------------------

# 1. DEFINICIÓN DEL DATASET PERSONALIZADO
class CarlaDrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Lee el CSV. Las columnas son: frame, image_path, steer, throttle, brake.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Leer imagen
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # PyTorch prefiere RGB
        
        # Leer etiqueta (Solo Steering para PilotNet clásico)
        # La columna 2 es 'steer'
        steer = self.data.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
            
        # Devolvemos la imagen y el label (steer) como tensor float
        return image, torch.tensor([steer], dtype=torch.float32)


# 2. ARQUITECTURA PILOTNET (NVIDIA)
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
        # Originalmente PilotNet espera imágenes de 200x66 (Ancho x Alto) con 3 canales (RGB)
        # Usamos ELU en lugar de ReLU para evitar "neuronas muertas" en valores negativos
        self.conv_layers = nn.Sequential(
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
        
        # Capas totalmente conectadas (Fully Connected)
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 1164), # El tamaño 64x1x18 resulta tras los strides y kernels
            nn.ELU(),
            nn.Linear(1164, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1) # Salida única: Ángulo del volante (Steering)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x


# 3. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
def main():
    # Transformaciones: PilotNet requiere imágenes de 200x66
    # Como tu imagen original es 640x360, la redimensionamos para no saturar la red
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)), # Alto x Ancho
        transforms.ToTensor(), # Convierte a rango [0, 1] y dimensiona (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet norm
    ])

    # Cargar Dataset y DataLoader
    print("[INFO] Cargando Dataset...")
    dataset = CarlaDrivingDataset(csv_file=CSV_FILE, root_dir=DATA_DIR, transform=transform)
    
    # Dividir en Entrenamiento (80%) y Validación (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # Instanciar el Modelo, la Función de Pérdida y el Optimizador
    model = PilotNet().to(device)
    criterion = nn.MSELoss() # Mean Squared Error (ideal para regresión continua como el steering)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Iniciando Entrenamiento por {EPOCHS} épocas...")
    best_val_loss = float('inf')

    # Bucle de Épocas
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, steers) in enumerate(train_loader):
            images, steers = images.to(device), steers.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, steers)

            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, steers in val_loader:
                images, steers = images.to(device), steers.to(device)
                outputs = model(images)
                loss = criterion(outputs, steers)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        print(f"Época [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # Guardar el mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pilotnet_best.pth")
            print("  --> Modelo mejorado y guardado!")

    print("[INFO] Entrenamiento finalizado. Mejor modelo guardado como 'pilotnet_best.pth'")

if __name__ == "__main__":
    main()