import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm
from datetime import datetime

# ==========================================
# 1. CARGADOR DE DATOS (DATASET)
# ==========================================
class DrivingDataset(Dataset):
    def __init__(self, csv_file, frames_dir):
        self.data = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extraemos el nombre y construimos la ruta absoluta
        img_filename = os.path.basename(row['image_path'])
        img_path = os.path.join(self.frames_dir, img_filename)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No se encontró la imagen: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image)
        
        # Usamos Throttle como entrada numérica (ya está entre 0.0 y 1.0)
        throttle_in = torch.tensor([row['throttle']], dtype=torch.float32)
        
        # Etiqueta: solo Steering
        steering_out = torch.tensor([row['steering']], dtype=torch.float32)
        
        return image_tensor, throttle_in, steering_out

# ==========================================
# 2. ARQUITECTURA DEL MODELO (RESNET-18 + FC)
# ==========================================
class DualInputModel(nn.Module):
    def __init__(self):
        super(DualInputModel, self).__init__()
        
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity() 
        
        # Fusión: 512 (Visual) + 1 (Throttle) = 513 entradas
        self.fusion_fc1 = nn.Linear(num_features + 1, 128)
        self.fusion_fc2 = nn.Linear(128, 64)
        
        # Capa de salida: 1 neurona (Steering)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, image, throttle_val):
        visual_features = self.cnn(image) 
        combined = torch.cat((visual_features, throttle_val), dim=1) 
        
        x = F.relu(self.fusion_fc1(combined))
        x = F.relu(self.fusion_fc2(x))
        
        output = self.output_layer(x)
        return output

# ==========================================
# 3. PÉRDIDA PONDERADA (HOT BALANCING)
# ==========================================
def weighted_steering_loss(pred_steering, true_steering, base_weight=1.0, curve_multiplier=10.0):
    raw_loss = F.huber_loss(pred_steering, true_steering, reduction='none')
    weights = base_weight + (torch.abs(true_steering) * curve_multiplier)
    return torch.mean(raw_loss * weights)

# ==========================================
# 4. BUCLE DE ENTRENAMIENTO PRINCIPAL
# ==========================================
def main():
    # --- CONFIGURACIÓN ---
    CSV_FILE = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/Homograficas/dataset_masivo_continuo/driving_log.csv"
    FRAMES_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/Homograficas/dataset_masivo_continuo/frames_homographic" 
    
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Utilizando dispositivo: {device}")

    # --- CARGA DE DATOS ---
    dataset = DrivingDataset(csv_file=CSV_FILE, frames_dir=FRAMES_DIR)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- INICIALIZAR MODELO ---
    model = DualInputModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"[INFO] Iniciando entrenamiento por {EPOCHS} épocas...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Train]")
        for images, throttles_in, steerings in progress_bar:
            images = images.to(device)
            throttles_in = throttles_in.to(device)
            steerings = steerings.to(device)
            
            optimizer.zero_grad()
            
            # Predicción (ahora solo devuelve el volante)
            pred_steering = model(images, throttles_in)
            
            # Calcular pérdida ponderada
            loss = weighted_steering_loss(pred_steering, steerings, curve_multiplier=12.0)
            
            # Retropropagación
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # --- VALIDACIÓN ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, throttles_in, steerings in val_loader:
                images = images.to(device)
                throttles_in = throttles_in.to(device)
                steerings = steerings.to(device)
                
                pred_steering = model(images, throttles_in)
                loss = weighted_steering_loss(pred_steering, steerings, curve_multiplier=12.0)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Resultado Época {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Guardar los pesos finales con timestamp para evitar sobreescritura
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"modelo_resnet18_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[ÉXITO] Modelo guardado en: {model_save_path}")

if __name__ == '__main__':
    main()