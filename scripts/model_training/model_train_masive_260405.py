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
CSV_PATH =    "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260423_162235/driving_log.csv"
DATASET_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260423_162235"
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4

# Verificar hardware (Detectará tu RX 6750 XT a través de ROCm)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo de entrenamiento: {DEVICE}")

# --- 1. DEFINICIÓN DEL DATASET CON LIMPIEZA AUTOMÁTICA ---
class CarlaDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        df = pd.read_csv(csv_file)
        
        # --- FILTRO ANTI-INERCIA (El "Anti-Baño") ---
        # Identificamos las filas donde el coche iba a menos de 0.5 km/h 
        # Y además no estabas tocando el acelerador ni el freno
        condicion_estacionado = (df['speed_kmh'] < 0.5) & (df['throttle'] < 0.05) & (df['brake'] < 0.05)
        
        # Guardamos solo los datos donde NO se cumple esa condición (usando el símbolo ~)
        self.data = df[~condicion_estacionado].reset_index(drop=True)
        
        datos_eliminados = len(df) - len(self.data)
        print(f"[INFO] Dataset cargado: {len(df)} frames originales.")
        print(f"[INFO] Se eliminaron {datos_eliminados} frames de inactividad (estacionado).")
        print(f"[INFO] Frames útiles para entrenar: {len(self.data)}.")
        
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Cargar y preprocesar imagen
        img_path = os.path.join(self.root_dir, row['image_path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 160)) 
        
        # Normalizar y preparar tensor
        img = img / 255.0
        img = img.transpose((2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.float32)
        
        # Velocidad normalizada (escalada aprox. a un máximo de 50 km/h)
        speed = row['speed_kmh'] / 50.0  
        speed_tensor = torch.tensor([speed], dtype=torch.float32)
        
        # Etiquetas
        steer = torch.tensor([row['steer']], dtype=torch.float32)
        throttle = torch.tensor([row['throttle']], dtype=torch.float32)
        
        return img_tensor, speed_tensor, steer, throttle

# --- 2. LA ARQUITECTURA PILOTNET CONDICIONAL ---
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
        
        dummy_input = torch.zeros(1, 3, 160, 320)
        flatten_size = self.features(dummy_input).shape[1]
        
        combined_size = flatten_size + 1 # +1 por la inyección de velocidad
        
        self.decision = nn.Sequential(
            nn.Linear(combined_size, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU()
        )
        
        self.out_steer = nn.Linear(10, 1)
        self.out_throttle = nn.Linear(10, 1)

    def forward(self, img, speed):
        visual_features = self.features(img)
        fused_data = torch.cat((visual_features, speed), dim=1)
        decision_features = self.decision(fused_data)
        
        steer = self.out_steer(decision_features)
        throttle = self.out_throttle(decision_features)
        return steer, throttle

# --- 3. BUCLE DE ENTRENAMIENTO ---
def main():
    print("[INFO] Preparando DataLoader...")
    dataset = CarlaDataset(CSV_PATH, DATASET_DIR)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, # Seguro para MIOpen en AMD
        drop_last=True,
        pin_memory=True
    )

    model = ConditionalPilotNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("[INFO] ¡Comenzando el entrenamiento!")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss_steer = 0.0
        running_loss_throttle = 0.0
        
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{EPOCHS}", unit="batch")
        
        for imgs, speeds, steers, throttles in pbar:
            imgs = imgs.to(DEVICE)
            speeds = speeds.to(DEVICE)
            steers = steers.to(DEVICE)
            throttles = throttles.to(DEVICE)

            optimizer.zero_grad()
            
            # Predicción visual y de velocidad
            pred_steer, pred_throttle = model(imgs, speeds)
            
            # --- BALANCEO EN CALIENTE PARA EL VOLANTE ---
            # Multiplicador matemático: 1.0 en recta, más de 20.0 en curvas extremas
            pesos_dinamicos = 1.0 + 20.0 * torch.abs(steers) 
            error_steer_puro = (pred_steer - steers)**2
            loss_steer = torch.mean(error_steer_puro * pesos_dinamicos)
            
            # Pérdida normal para el acelerador
            loss_throttle = criterion(pred_throttle, throttles)
            
            # Backpropagation
            total_loss = loss_steer + loss_throttle
            total_loss.backward()
            optimizer.step()
            
            # Actualizar métricas
            running_loss_steer += loss_steer.item()
            running_loss_throttle += loss_throttle.item()
            
            pbar.set_postfix({
                'L_Steer': f"{(running_loss_steer/(pbar.n+1)):.4f}", 
                'L_Thr': f"{(running_loss_throttle/(pbar.n+1)):.4f}"
            })
            
        epoch_loss = (running_loss_steer + running_loss_throttle) / len(dataloader)
        print(f"-> Fin Época {epoch+1} | Pérdida Total: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "pilotnet_20260423.pth")
            print("   [!] Nuevo mejor modelo guardado (pilotnet_best.pth)")

if __name__ == "__main__":
    main()