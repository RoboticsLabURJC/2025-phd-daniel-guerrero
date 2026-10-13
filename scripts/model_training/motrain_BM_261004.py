import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================================================
# CONFIGURACIÓN
# =========================================================
CSV_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260423_162235/driving_log.csv"
DATASET_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/dataset_extraido_20260423_162235"

MODEL_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/model_training/pilotnet_bm_compatible.pth"
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Este es el color de carretera que BM compara
ROAD_RGB = np.array([128, 64, 128], dtype=np.uint8)

# =========================================================
# MODELO COMPATIBLE CON BM
# =========================================================
class PilotNetLegacy(nn.Module):
    def __init__(self):
        super().__init__()

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
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# =========================================================
# PREPROCESADO BM-COMPATIBLE
# =========================================================
def build_road_mask(seg_bgr):
    seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    diff = np.abs(seg_rgb.astype(np.int16) - ROAD_RGB.astype(np.int16))
    mask = np.all(diff <= 5, axis=-1).astype(np.uint8) * 255

    if np.count_nonzero(mask) < 500:
        hsv = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 0), (179, 140, 180))
        mask = cv2.bitwise_not(mask)
        mask = cv2.medianBlur(mask, 5)

    return mask


def preprocess_for_bm(seg_bgr):
    mask = build_road_mask(seg_bgr)

    h, w = mask.shape[:2]
    y0 = int(h * 0.40)
    cropped = mask[y0:h, :]

    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_NEAREST)
    resized = resized.astype(np.float32) / 255.0

    x = np.stack([resized, resized, resized], axis=0).astype(np.float32)
    return x

# =========================================================
# DATASET DESDE EL CSV
# =========================================================
class CarlaBMCompatibleDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        df = pd.read_csv(csv_file)

        required = {"image_path", "steer"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas: {sorted(missing)}")

        if {"speed_kmh", "throttle", "brake"}.issubset(df.columns):
            parado = (
                (df["speed_kmh"] < 0.5) &
                (df["throttle"] < 0.05) &
                (df["brake"] < 0.05)
            )
            df = df[~parado].reset_index(drop=True)

        df = df[np.isfinite(df["steer"])].copy()
        df["steer"] = df["steer"].clip(-1.0, 1.0)

        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir

        print(f"[INFO] Frames útiles para entrenamiento: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No existe la imagen: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Imagen corrupta o vacía: {img_path}")

        x = preprocess_for_bm(img)
        y = np.array([float(row["steer"])], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)

# =========================================================
# ENTRENAMIENTO
# =========================================================
def train_model():
    dataset = CarlaBMCompatibleDataset(CSV_PATH, DATASET_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = PilotNetLegacy().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.6f}")

        epoch_loss = running_loss / len(loader)
        print(f"[EPOCH {epoch + 1}] loss = {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[SAVE] Mejor modelo guardado: {MODEL_PATH}")

    print("[INFO] Entrenamiento finalizado.")
    print(f"[INFO] Modelo final: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()