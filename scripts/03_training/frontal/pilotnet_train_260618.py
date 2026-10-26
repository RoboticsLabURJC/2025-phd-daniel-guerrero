import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ==========================================
# 1. ARQUITECTURA PILOTNET (NVIDIA)
# ==========================================
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        # Entrada: 3 canales (RGB), Imagen: 66x200
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
        
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1) # Salida: Ángulo del volante (Steering)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Aplanar tensor
        x = self.linear_layers(x)
        return x

# ==========================================
# 2. TRANSFORMACIONES Y RECORTES (CROP)
# ==========================================
def preprocess_image(image):
    """
    Recorta el cielo (top 300px) de la imagen original 800x600 capturada desde la
    posición frontal central y redimensiona a 200x66.
    """
    # TF.crop(img, top, left, height, width)
    cropped = TF.crop(image, top=300, left=0, height=300, width=800)
    resized = TF.resize(cropped, size=[66, 200])
    return resized

custom_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(preprocess_image),
    transforms.ToTensor(),
    # Normalización estándar para ayudar a la convergencia
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# ==========================================
# 3. DATASET PERSONALIZADO
# ==========================================
class CarlaDrivingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Asume que el log ya fue depurado mediante filtro de acelerador (throttle-based)
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image_path'])
        
        # Leer imagen y convertir de BGR (OpenCV) a RGB
        image_bgr = cv2.imread(img_name)
        if image_bgr is None:
            raise FileNotFoundError(f"Imagen no encontrada: {img_name}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Extraer el ángulo del volante
        steering = float(self.data_frame.iloc[idx]['steering'])
        
        if self.transform:
            image_rgb = self.transform(image_rgb)
            
        return image_rgb, torch.tensor([steering], dtype=torch.float32)

# ==========================================
# 4. BALANCEO EN CALIENTE (PRIORIZAR CURVAS)
# ==========================================
def create_balanced_sampler(csv_file):
    """
    Asigna probabilidades altas a las curvas y probabilidades bajas a las rectas.
    """
    print("[INFO] Calculando pesos estadísticos para el balanceo en caliente...")
    df = pd.read_csv(csv_file)
    weights = []
    
    for steer in df['steering']:
        # Umbral para considerar una línea recta
        if abs(steer) < 0.05:
            weights.append(0.1) # Baja probabilidad (10%) de ser seleccionada
        else:
            # A mayor giro, mayor prioridad en el entrenamiento
            weights.append(1.0 + (abs(steer) * 10.0)) 
            
    weights_tensor = torch.DoubleTensor(weights)
    
    # El sampler sacará lotes priorizando los pesos más altos
    sampler = WeightedRandomSampler(
        weights=weights_tensor, 
        num_samples=len(weights_tensor), 
        replacement=True
    )
    return sampler

# ==========================================
# 5. CICLO DE ENTRENAMIENTO
# ==========================================
def main():
    # Rutas relativas a tu dataset maestro
    DATASET_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_masivo_20260614_162217"
    CSV_FILE = os.path.join(DATASET_DIR, "driving_log.csv")
    
    # Hiperparámetros
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de entrenamiento: {device}")

    # Inicializar Dataset y Muestreador Balanceado
    dataset = CarlaDrivingDataset(csv_file=CSV_FILE, root_dir=DATASET_DIR, transform=custom_transform)
    sampler = create_balanced_sampler(CSV_FILE)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)

    # Inicializar Modelo, Función de Pérdida (MSE) y Optimizador (Adam)
    model = PilotNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Iniciando entrenamiento por {EPOCHS} épocas...\n")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Barra de progreso para cada iteración
        progress_bar = tqdm(dataloader, desc=f"Época {epoch+1}/{EPOCHS}")
        
        for images, steerings in progress_bar:
            images = images.to(device)
            steerings = steerings.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, steerings)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        print(f"--- Fin Época {epoch+1} | Pérdida MSE Promedio: {epoch_loss:.6f}")

    # Guardar los pesos finales
    model_path = "pilotnet_balanced_weights.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[ÉXITO] Modelo entrenado y guardado en {model_path}")

if __name__ == '__main__':
    main()