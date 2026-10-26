# train_mobilenet_carla_lane_following_with_crop.py

import os
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ===================== DATASET =====================
class CarlaExpertDatasetCropped(Dataset):
    """
    Lee un CSV (labels.csv / labels_balanced.csv) y entrena con etiquetas del EXPERTO:
      expert_steer, expert_throttle, expert_brake

    Aplica recorte fijo (top/bottom) y luego:
      Resize manteniendo ratio + CenterCrop(224) + Normalize ImageNet
    """
    def __init__(
        self,
        csv_path: str,
        run_root_dir: str,
        crop_top_px: int = 300,
        crop_bottom_px: int = 80,
        filter_noise_enabled_zero: bool = False,  # <-- pon True si quieres usar solo noise_enabled==0
    ):
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip().str.lower()

        required = {"image", "expert_steer", "expert_throttle", "expert_brake"}
        missing = required - set(self.data.columns)
        if missing:
            raise KeyError(f"Faltan columnas en CSV: {missing}")

        # Opcional: filtrar solo filas sin ruido si existe la columna
        if filter_noise_enabled_zero and "noise_enabled" in self.data.columns:
            before = len(self.data)
            self.data = self.data[self.data["noise_enabled"].astype(int) == 0].reset_index(drop=True)
            after = len(self.data)
            print(f"[INFO] Filtrado noise_enabled==0: {before} -> {after} filas")

        if len(self.data) == 0:
            raise ValueError("Dataset vacío tras filtros.")

        self.run_root_dir = run_root_dir
        self.img_paths = [os.path.join(self.run_root_dir, p) for p in self.data["image"].astype(str)]

        ycols = ["expert_steer", "expert_throttle", "expert_brake"]
        self.labels = self.data[ycols].astype("float32").to_numpy()

        self.crop_top_px = int(crop_top_px)
        self.crop_bottom_px = int(crop_bottom_px)

        # IMPORTANT: evita deformar (no Resize((224,224)) directo)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),      # mantiene aspect ratio, lado corto -> 256
            transforms.CenterCrop(224),  # recorta a 224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path)  # BGR
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # --- Recorte fijo (quita cielo/edificios + capó) ---
        top = min(self.crop_top_px, h - 2)
        bottom = min(self.crop_bottom_px, h - top - 1)
        img = img[top:h - bottom, :, :]

        # Transform final a (3,224,224)
        x = self.transform(img)

        y = torch.from_numpy(self.labels[idx])  # (3,)
        return x, y


# ===================== MODELO =====================
class CarlaMobileNet(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
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


# ===================== LOSS =====================
class WeightedMSE(nn.Module):
    """
    Pesa más steer (clave para lane-following).
    """
    def __init__(self, w_steer=5.0, w_throttle=1.0, w_brake=0.5):
        super().__init__()
        self.w = torch.tensor([w_steer, w_throttle, w_brake], dtype=torch.float32)

    def forward(self, pred, target):
        w = self.w.to(pred.device).view(1, 3)
        return torch.mean(w * (pred - target) ** 2)


# ===================== ENTRENAMIENTO =====================
def main():
    # ---- AJUSTA ESTO ----
    run_root_dir = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/datasets_collection/dataset_runs/20260208_195832"
    csv_path = os.path.join(run_root_dir, "labels.csv")  # o labels_balanced.csv

    # Recorte recomendado para tus frames como el ejemplo (1280x920)
    CROP_TOP = 300
    CROP_BOTTOM = 80

    # Si quieres excluir filas donde hubo ruido (si existe noise_enabled)
    FILTER_NOISE_ZERO = False
    # ---------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = CarlaExpertDatasetCropped(
        csv_path=csv_path,
        run_root_dir=run_root_dir,
        crop_top_px=CROP_TOP,
        crop_bottom_px=CROP_BOTTOM,
        filter_noise_enabled_zero=FILTER_NOISE_ZERO
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    model = CarlaMobileNet(freeze_backbone=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = WeightedMSE(w_steer=5.0, w_throttle=1.0, w_brake=0.5)

    epochs = 20
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            raw = model(imgs)

            # salidas en rangos válidos
            preds = torch.stack([
                torch.tanh(raw[:, 0]),        # steer [-1,1]
                torch.sigmoid(raw[:, 1]),     # throttle [0,1]
                torch.sigmoid(raw[:, 2]),     # brake [0,1]
            ], dim=1)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    out_path = os.path.join(run_root_dir, "carla_mobilenet_cropped.pth")
    torch.save(model.state_dict(), out_path)
    print("✅ Modelo guardado en:", out_path)


if __name__ == "__main__":
    main()
