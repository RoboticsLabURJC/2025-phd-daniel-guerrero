##Mobile net training

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import os
import cv2
import numpy as np
from torchvision import models, transforms

# ===================== DATASET =====================
class CarlaDataset(Dataset):
    def __init__(self, csv_path, img_root_dir):
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip().str.lower()

        required = {'frame', 'steer', 'throttle', 'brake'}
        missing = required - set(self.data.columns)
        if missing:
            raise KeyError(f"Faltan columnas: {missing}")

        self.img_root_dir = img_root_dir
        self.paths = [os.path.join(self.img_root_dir, p) for p in self.data['frame'].astype(str)]
        self.labels = self.data[['steer', 'throttle', 'brake']].astype('float32').to_numpy()

        # Transformaciones (IMPORTANTE para MobileNet)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # Recomendación de ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])  # BGR
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar {self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)  # APLICA TRANSFORM

        label = torch.from_numpy(self.labels[idx])  # (3,)
        return img, label

# ===================== MODELO =====================
class CarlaMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Congelar capas base si quieres transfer learning:
        for param in self.base.features.parameters():
            param.requires_grad = False

        # Reemplazar último head para regresión
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base.last_channel, 3)  # steer, throttle, brake
        )

    def forward(self, x):
        return self.base(x)

# ===================== ENTRENAMIENTO =====================
if __name__ == "__main__":
    csv_path = "dataset/controls_balanced.csv"
    img_root_dir = "dataset/images"

    device = torch.device("cpu")  # Puedes cambiar si luego ROCm te detecta la GPU

    dataset = CarlaDataset(csv_path, img_root_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CarlaMobileNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(10):
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📦 Epoch {epoch+1}/10 | Loss: {total_loss:.4f}")

    torch.save(model, "carla_model_mobilenet.pth")
    print("✅ Modelo guardado como carla_model_mobilenet.pth")
