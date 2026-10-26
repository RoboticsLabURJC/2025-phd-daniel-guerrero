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

# ==========================================
# 1. CARGADOR DE DATOS (DATASET)
# ==========================================
class DrivingDataset(Dataset):
    def __init__(self, csv_file, frames_dir, max_speed=80.0):
        self.data = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.max_speed = max_speed
        
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
        
        # Extraemos SOLO el nombre del archivo (ignorando si el CSV dice "frames/001.png")
        img_filename = os.path.basename(row['image_path'])
        
        # Lo unimos estrictamente con la carpeta de imágenes procesadas que tú decidas
        img_path = os.path.join(self.frames_dir, img_filename)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No se encontró la imagen: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image)
        
        velocidad_norm = min(row['velocidad_kmh'] / self.max_speed, 1.0)
        speed_tensor = torch.tensor([velocidad_norm], dtype=torch.float32)
        
        steering = torch.tensor([row['steering']], dtype=torch.float32)
        throttle = torch.tensor([row['throttle']], dtype=torch.float32)
        
        return image_tensor, speed_tensor, steering, throttle
# ==========================================
# 2. ARQUITECTURA DEL MODELO (RESNET-18 + FC)
# ==========================================
class DualInputModel(nn.Module):
    def __init__(self):
        super(DualInputModel, self).__init__()
        
        # Rama Visual: ResNet-18 preentrenada
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Eliminamos la última capa de clasificación (fc) para extraer solo características (512 valores)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity() 
        
        # Rama de Fusión: 512 (Visual) + 1 (Velocidad) = 513 entradas
        self.fusion_fc1 = nn.Linear(num_features + 1, 128)
        self.fusion_fc2 = nn.Linear(128, 64)
        
        # Capa de salida: 2 neuronas (Steering, Throttle)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, image, speed):
        # Extraer características visuales
        visual_features = self.cnn(image) 
        
        # Concatenar con la velocidad
        combined = torch.cat((visual_features, speed), dim=1) 
        
        # Pasar por las capas de fusión
        x = F.relu(self.fusion_fc1(combined))
        x = F.relu(self.fusion_fc2(x))
        
        # Salida final
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
    # Ajusta estas rutas a los nombres reales de tus carpetas
    CSV_FILE = "dataset_masivo_continuo/driving_log.csv"
    FRAMES_DIR = "dataset_masivo_continuo/frames_homographic" # <-- Apunta directo a las BEV
    
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Utilizando dispositivo: {device}")

    # --- CARGA DE DATOS ---
    # Instanciamos pasando el frames_dir
    dataset = DrivingDataset(csv_file=CSV_FILE, frames_dir=FRAMES_DIR)
    
    # Separar en 80% entrenamiento y 20% validación
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
        for images, speeds, steerings, throttles in progress_bar:
            images = images.to(device)
            speeds = speeds.to(device)
            steerings = steerings.to(device)
            throttles = throttles.to(device)
            
            optimizer.zero_grad()
            
            # Predicción
            outputs = model(images, speeds)
            pred_steering = outputs[:, 0].unsqueeze(1)
            pred_throttle = outputs[:, 1].unsqueeze(1)
            
            # Calcular pérdidas (Volante ponderado + Acelerador estándar)
            loss_steer = weighted_steering_loss(pred_steering, steerings, curve_multiplier=12.0)
            loss_throttle = F.huber_loss(pred_throttle, throttles)
            
            # Combinar pérdidas (Le damos doble importancia a equivocarse en el volante)
            total_loss = (loss_steer * 2.0) + loss_throttle
            
            # Retropropagación
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            progress_bar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        # --- VALIDACIÓN ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, speeds, steerings, throttles in val_loader:
                images, speeds, steerings, throttles = images.to(device), speeds.to(device), steerings.to(device), throttles.to(device)
                
                outputs = model(images, speeds)
                pred_steering = outputs[:, 0].unsqueeze(1)
                pred_throttle = outputs[:, 1].unsqueeze(1)
                
                loss_steer = weighted_steering_loss(pred_steering, steerings, curve_multiplier=12.0)
                loss_throttle = F.huber_loss(pred_throttle, throttles)
                val_loss += ((loss_steer * 2.0) + loss_throttle).item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Resultado Época {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Guardar los pesos finales
    model_save_path = "modelo_conduccion_autonoma.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[ÉXITO] Modelo guardado en: {model_save_path}")

if __name__ == '__main__':
    main()