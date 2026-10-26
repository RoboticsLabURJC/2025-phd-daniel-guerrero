import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------- Parámetros de Entrenamiento ----------
H5_FILE = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_masivo.h5" 
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4

# Detectar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Entrenando en: {device}")
# -----------------------------------------------

# 1. DEFINICIÓN DEL DATASET HDF5
class CarlaH5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.dataset = None
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['images'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')
            self.images = self.dataset['images']
            self.controls = self.dataset['controls']

        img_bgr = self.images[idx]
        img_rgb = img_bgr[:, :, ::-1].copy() # BGR a RGB
        
        steer = self.controls[idx][0]
        
        if self.transform:
            img_rgb = self.transform(img_rgb)
            
        return img_rgb, torch.tensor([steer], dtype=torch.float32)


# 2. ARQUITECTURA PILOTNET (NVIDIA)
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU()
        )
        
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 1164), nn.ELU(),
            nn.Linear(1164, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1) 
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x


# 3. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
def main():
    if not os.path.exists(H5_FILE):
        print(f"[ERROR] No se encuentra el archivo {H5_FILE}")
        return

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    print("[INFO] Conectando Dataset HDF5...")
    dataset = CarlaH5Dataset(h5_path=H5_FILE, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # CONFIGURACIÓN SEGURA PARA AMD: drop_last=True SOLO en entrenamiento
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    model = PilotNet().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Iniciando Entrenamiento por {EPOCHS} épocas...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Época [{epoch+1}/{EPOCHS}] - Entrenando", leave=False)
        
        for images, steers in train_pbar:
            images, steers = images.to(device), steers.to(device)

            outputs = model(images)
            loss = criterion(outputs, steers)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Blindaje por si el dataset es tan pequeño que no arma ni un lote
        if len(train_loader) > 0:
            avg_train_loss = train_loss / len(train_loader)
        else:
            avg_train_loss = float('inf')

        # --- FASE DE VALIDACIÓN ---
        model.eval()
        val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Época [{epoch+1}/{EPOCHS}] - Validando", leave=False)
        
        with torch.no_grad():
            for images, steers in val_pbar:
                images, steers = images.to(device), steers.to(device)
                outputs = model(images)
                loss = criterion(outputs, steers)
                val_loss += loss.item()
                
        # Blindaje contra división por cero en validación
        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = float('inf')
            print("\n⚠️ Advertencia: Dataset de validación muy pequeño para este BATCH_SIZE.")

        print(f"✅ Época [{epoch+1}/{EPOCHS}] Completada | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pilotnet_best.pth")
            print("  🏆 ¡Nuevo mejor modelo guardado (pilotnet_best.pth)!")

    print("\n[INFO] Entrenamiento finalizado exitosamente. Listo para inferencia.")

if __name__ == "__main__":
    main()