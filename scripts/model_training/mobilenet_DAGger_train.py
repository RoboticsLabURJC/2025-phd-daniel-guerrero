# train_mobilenet_carla_lane_following.py

import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

# ===================== DATASET =====================
class CarlaDaggerDataset(Dataset):
    """
    Lee el CSV generado por el script de CARLA (continuous noise + expert labeling).
    Usa como etiqueta los controles del EXPERTO: expert_steer, expert_throttle, expert_brake.
    """
    def __init__(self, csv_path, run_root_dir, use_expert=True):
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip().str.lower()

        # Esperamos estas columnas del CSV nuevo
        required = {"image", "expert_steer", "expert_throttle", "expert_brake"}
        missing = required - set(self.data.columns)
        if missing:
            raise KeyError(f"Faltan columnas en CSV: {missing}")

        self.run_root_dir = run_root_dir  # carpeta dagger_runs/<run_id>/
        self.img_paths = [
            os.path.join(self.run_root_dir, p) for p in self.data["image"].astype(str)
        ]

        if use_expert:
            ycols = ["expert_steer", "expert_throttle", "expert_brake"]
        else:
            # alternativa (NO recomendado para aprender lane-following): entrenar lo aplicado con ruido
            ycols = ["applied_steer", "applied_throttle", "applied_brake"]

        self.labels = self.data[ycols].astype("float32").to_numpy()

        # MobileNet: input 224x224 + normalización ImageNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])  # BGR
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar {self.img_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        y = torch.from_numpy(self.labels[idx])  # (3,)
        return img, y


# ===================== MODELO =====================
class CarlaMobileNet(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        if freeze_backbone:
            for p in self.base.features.parameters():
                p.requires_grad = False

        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base.last_channel, 3)  # steer, throttle, brake
        )

    def forward(self, x):
        return self.base(x)


# ===================== LOSS (enfocado a seguir carril) =====================
class WeightedMSE(nn.Module):
    """
    Pesa más el error de steer para lane-following.
    """
    def __init__(self, w_steer=3.0, w_throttle=1.0, w_brake=1.0):
        super().__init__()
        self.w = torch.tensor([w_steer, w_throttle, w_brake], dtype=torch.float32)

    def forward(self, pred, target):
        # pred/target: (B,3)
        w = self.w.to(pred.device).view(1, 3)
        return torch.mean(w * (pred - target) ** 2)


# ===================== ENTRENAMIENTO =====================
def main():
    # Apunta a TU run
    run_root_dir = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/datasets_collection/dagger_runs/20260201_211356"
    csv_path = os.path.join(run_root_dir, "labels_balanced.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = CarlaDaggerDataset(csv_path, run_root_dir, use_expert=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))

    model = CarlaMobileNet(freeze_backbone=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = WeightedMSE(w_steer=3.0, w_throttle=1.0, w_brake=1.0)

    epochs = 15
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)

            # Opcional: limitar rangos (steer [-1,1], throttle/brake [0,1])
            preds = torch.stack([
                torch.tanh(preds[:, 0]),                  # steer
                torch.sigmoid(preds[:, 1]),               # throttle
                torch.sigmoid(preds[:, 2]),               # brake
            ], dim=1)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # Guardado recomendado: state_dict
    out_path = os.path.join(run_root_dir, "carla_mobilenet_balance.pth")
    torch.save(model.state_dict(), out_path)
    print("✅ Modelo guardado en:", out_path)


if __name__ == "__main__":
    main()
