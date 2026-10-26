import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIGURACIÓN ---
CSV_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260423_162235/driving_log.csv"
DATASET_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260423_162235"
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DATASET CON MÁSCARA Y AUMENTO DE DATOS ---
class CarlaDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        df = pd.read_csv(csv_file)
        # Filtro Anti-Inercia (Elimina paradas técnicas)
        condicion_estacionado = (df['speed_kmh'] < 0.5) & (df['throttle'] < 0.05) & (df['brake'] < 0.05)
        self.data = df[~condicion_estacionado].reset_index(drop=True)
        self.root_dir = root_dir
        print(f"[INFO] Entrenamiento con {len(self.data)} frames útiles.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # APLICAR MÁSCARA (Fundamental para que coincida con inferencia)
        h = img.shape[0]
        img[0:h//2, :] = 0 
        
        img = cv2.resize(img, (320, 160)) 

        # AUMENTO DE DATOS: Brillo aleatorio (Simula cambios de sol en Town04)
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * ratio, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Normalización y Tensor
        img = img / 255.0
        img = img.transpose((2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.float32)
        
        speed = torch.tensor([row['speed_kmh'] / 50.0], dtype=torch.float32)
        steer = torch.tensor([row['steer']], dtype=torch.float32)
        throttle = torch.tensor([row['throttle']], dtype=torch.float32)
        
        return img_tensor, speed, steer, throttle

# --- 2. ARQUITECTURA PILOTNET CONDICIONAL ---
class ConditionalPilotNet(nn.Module):
    def __init__(self):
        super(ConditionalPilotNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ELU(),
            nn.Flatten()
        )
        
        # Cálculo automático del tamaño tras las convoluciones
        with torch.no_grad():
            dummy = self.features(torch.zeros(1, 3, 160, 320))
            flatten_size = dummy.shape[1]
        
        self.decision = nn.Sequential(
            nn.Linear(flatten_size + 1, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU()
        )
        self.out_steer = nn.Linear(10, 1)
        self.out_throttle = nn.Linear(10, 1)

    def forward(self, img, speed):
        x = self.features(img)
        x = torch.cat((x, speed), dim=1)
        x = self.decision(x)
        return self.out_steer(x), self.out_throttle(x)

# --- 3. BUCLE DE ENTRENAMIENTO ---
def main():
    dataset = CarlaDataset(CSV_PATH, DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = ConditionalPilotNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float('inf')

    print(f"[INFO] Entrenando en {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{EPOCHS}")

        for imgs, speeds, steers, throttles in pbar:
            imgs, speeds, steers, throttles = imgs.to(DEVICE), speeds.to(DEVICE), steers.to(DEVICE), throttles.to(DEVICE)

            optimizer.zero_grad()
            p_steer, p_throttle = model(imgs, speeds)

            # BALANCEO EN CALIENTE: Prioriza aprender a no salirse en curvas
            pesos = 1.0 + 25.0 * torch.abs(steers) # Subimos a 25 para ser más agresivos
            loss_steer = torch.mean(((p_steer - steers)**2) * pesos)
            loss_thr = criterion(p_throttle, throttles)

            loss = loss_steer + loss_thr
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{running_loss/(pbar.n+1):.4f}"})

        epoch_loss = running_loss / len(dataloader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "pilotnet_20260423_corregido.pth")
            print(f" [!] Modelo guardado: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()