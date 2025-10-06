import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import os
import cv2
import numpy as np

# ------------------ Dataset personalizado ------------------
class CarlaDataset(Dataset):
    def __init__(self, csv_path, img_root_dir):
        # Lee y normaliza nombres de columnas
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip().str.lower()

        # Valida columnas requeridas
        required = {'frame', 'steer', 'throttle', 'brake'}
        missing = required - set(self.data.columns)
        if missing:
            raise KeyError(f"Faltan columnas en el CSV: {missing}. "
                           f"Encontradas: {list(self.data.columns)}")

        self.img_root_dir = img_root_dir

        # Opcional: precomputar rutas y labels (más eficiente y evita chained indexing)
        self.paths = [os.path.join(self.img_root_dir, p) for p in self.data['frame'].astype(str)]
        self.labels = self.data[['steer', 'throttle', 'brake']].astype('float32').to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.paths[idx]

        image = cv2.imread(img_path)  # BGR
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

        image = cv2.resize(image, (200, 66))
        image = image.astype(np.float32) / 255.0
        # Convierte BGR->RGB (opcional, si tu entrenamiento lo asume en RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image)  # dtype=float32 ya

        label = torch.from_numpy(self.labels[idx])  # shape (3,)

        return image_tensor, label

# ------------------ Modelo simple tipo PilotNet ------------------
class CarlaPilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)
            flattened_size = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 3)  # steer, throttle, brake
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

# ------------------ Entrenamiento ------------------
if __name__ == "__main__":
    # Rutas
    csv_path = "dataset/controls.csv"
    img_root_dir = "dataset/images"

    # Dataset y DataLoader
    dataset = CarlaDataset(csv_path, img_root_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Modelo, optimizador y pérdida
    model = CarlaPilotNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Entrenamiento
    for epoch in range(10):
        total_loss = 0.0
        for imgs, labels in dataloader:
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"📦 Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # Guardar modelo completo
    torch.save(model, "carla_model.pth")
    print("✅ Modelo guardado como carla_model.pth")
