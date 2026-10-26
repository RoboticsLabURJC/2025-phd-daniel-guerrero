import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from datetime import datetime

# ==========================================
# 1. ARQUITECTURA EXACTA DEL BRAIN (PILOTNET)
# ==========================================
# Esta clase imita exactamente a behavior_metrics/brains/CARLA/utils/pilotnet.py
class PilotNet(nn.Module):
    def __init__(self, image_shape=(66, 200), num_labels=2):
        super(PilotNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        
        # Calcular el tamaño del aplanado dinámicamente
        dummy = torch.zeros(1, 3, image_shape[0], image_shape[1])
        dummy_out = self.conv_layers(dummy)
        flatten_size = dummy_out.view(1, -1).size(1)
        
        self.linear_layers = nn.Sequential(
            nn.Linear(flatten_size, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, num_labels)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# ==========================================
# 2. DATASET CON PREPROCESAMIENTO CLONADO DE BM
# ==========================================
class BehaviorMetricsDataset(Dataset):
    def __init__(self, csv_file, base_dir):
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        
        # Verificar columnas mínimas según el DAgger report
        required_cols = {'frame_global', 'throttle', 'steering', 'image_path', 'source_log'}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"El CSV debe contener las columnas: {required_cols}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Cargar la imagen original RAW (asumiendo 800x600)
        img_rel_path = row['image_path']
        img_path = os.path.join(self.base_dir, img_rel_path)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Error al leer: {img_path}")
            
        # Convertir a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. APLICAR EL CONTRATO ESTRICTO DEL BRAIN FRONTAL API
        # Crop: filas 240 a 480, columnas 0 a 640
        if image.shape[0] >= 480 and image.shape[1] >= 640:
            image = image[240:480, 0:640]
        else:
            raise ValueError(f"La imagen {img_path} es muy pequeña para el crop de Behavior Metrics.")
            
        # Resize a 200x66
        image = cv2.resize(image, (200, 66))
        
        # Convertir a Tensor Normalizado [0, 1] (sin ImageNet norm)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # 3. Etiquetas [throttle, steer] exactamente en ese orden
        targets = torch.tensor([row['throttle'], row['steering']], dtype=torch.float32)
        
        return image_tensor, targets, row['source_log']

# ==========================================
# 3. SPLIT POR EPISODIO (SOURCE_LOG)
# ==========================================
def split_by_log(dataset, val_split=0.2):
    unique_logs = dataset.data['source_log'].unique()
    np.random.shuffle(unique_logs)
    
    val_logs_count = max(1, int(len(unique_logs) * val_split))
    val_logs = set(unique_logs[:val_logs_count])
    
    train_indices = []
    val_indices = []
    
    for i, log_name in enumerate(dataset.data['source_log']):
        if log_name in val_logs:
            val_indices.append(i)
        else:
            train_indices.append(i)
            
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
def main():
    # --- RUTAS (APUNTA A TUS IMÁGENES RAW 800x600, NO BEV) ---
    BASE_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/Homograficas/dataset_masivo_continuo" 
    CSV_FILE = os.path.join(BASE_DIR, "driving_log.csv")
    
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    # Carga de Dataset
    dataset = BehaviorMetricsDataset(csv_file=CSV_FILE, base_dir=BASE_DIR)
    train_dataset, val_dataset = split_by_log(dataset, val_split=0.2)
    
    # Custom collate para omitir el 'source_log' en el batch
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return images, targets

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f"[INFO] Train: {len(train_dataset)} muestras | Val: {len(val_dataset)} muestras (Split por source_log)")

    model = PilotNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        # Validación
        model.eval()
        val_loss, val_throttle_loss, val_steer_loss = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                # Métricas separadas
                val_throttle_loss += nn.MSELoss()(outputs[:, 0], targets[:, 0]).item() * images.size(0)
                val_steer_loss += nn.MSELoss()(outputs[:, 1], targets[:, 1]).item() * images.size(0)
                
        avg_train = train_loss / len(train_dataset)
        avg_val = val_loss / len(val_dataset)
        avg_val_t = val_throttle_loss / len(val_dataset)
        avg_val_s = val_steer_loss / len(val_dataset)
        
        print(f"Epoca {epoch+1:02d}/{EPOCHS} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f} (Throttle: {avg_val_t:.4f}, Steer: {avg_val_s:.4f})")
        
        # Guardar mejor modelo en CPU para compatibilidad total con BM
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f"bm_pilotnet_frontal_{timestamp}.pth"
            
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, model_save_path)

    print(f"\n[ÉXITO] El mejor modelo compatible con BM fue guardado en: {model_save_path}")
    print("Para probarlo en Behavior Metrics, usa 'default_carla_front_image_api.yml' y pon esta ruta en 'SavedModelPath'.")

if __name__ == '__main__':
    main()