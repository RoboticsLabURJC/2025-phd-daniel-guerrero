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
        self.data = pd.read_csv(csv_path)
        self.img_root_dir = img_root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root_dir, self.data.iloc[idx]['seg_path'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
        image = cv2.resize(image, (200, 66))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        image_tensor = torch.tensor(image)

        label = torch.tensor([
            self.data.iloc[idx]['steer'],
            self.data.iloc[idx]['throttle'],
            self.data.iloc[idx]['brake']
        ], dtype=torch.float32)

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
        total_loss = 0
        for imgs, labels in dataloader:
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📦 Epoch {epoch+1} | Loss: {total_loss:.4f}")

    # Guardar modelo completo
    torch.save(model, "carla_model.pth")
    print("✅ Modelo guardado como carla_model_opt.pth")
