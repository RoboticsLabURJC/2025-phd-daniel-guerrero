import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

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
        
        img_filename = os.path.basename(row['image_path'])
        img_path = os.path.join(self.frames_dir, img_filename)
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No se encontró la imagen: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image)
        
        # Entrada Numérica
        throttle_in = torch.tensor([row['throttle']], dtype=torch.float32)
        
        # Etiqueta de Salida
        steering_out = torch.tensor([row['steering']], dtype=torch.float32)
        
        return image_tensor, throttle_in, steering_out

# ==========================================
# 2. ARQUITECTURA DEL MODELO (PILOTNET + FC)
# ==========================================
class DualInputPilotNet(nn.Module):
    def __init__(self, input_shape=(3, 240, 320)):
        super(DualInputPilotNet, self).__init__()
        
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
        
        self._to_linear = None
        self._calculate_flatten_size(input_shape)
        
        # Fusión: Características visuales aplanadas + 1 (Throttle)
        self.fusion_fc1 = nn.Linear(self._to_linear + 1, 100)
        self.fusion_fc2 = nn.Linear(100, 50)
        self.fusion_fc3 = nn.Linear(50, 10)
        self.output_layer = nn.Linear(10, 1)

    def _calculate_flatten_size(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv_layers(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, image, throttle_val):
        x = self.conv_layers(image)
        visual_features = x.view(x.size(0), -1) 
        
        combined = torch.cat((visual_features, throttle_val), dim=1) 
        
        x = F.elu(self.fusion_fc1(combined))
        x = F.elu(self.fusion_fc2(x))
        x = F.elu(self.fusion_fc3(x))
        
        return self.output_layer(x)

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

    # --- INICIALIZAR MODELO PILOTNET ---
    model = DualInputPilotNet().to(device)
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
            
            pred_steering = model(images, throttles_in)
            loss = weighted_steering_loss(pred_steering, steerings, curve_multiplier=12.0)
            
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

    # Guardar los pesos finales
    model_save_path = "modelo_conduccion_autonoma.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[ÉXITO] Modelo guardado en: {model_save_path}")

if __name__ == '__main__':
    main()