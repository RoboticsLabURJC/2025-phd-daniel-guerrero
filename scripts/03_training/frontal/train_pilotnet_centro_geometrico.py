#!/usr/bin/env python3
"""Entrenamiento de PilotNet con Imágenes Canónicas Transformadas al Centro Geométrico del Auto.

Transforma al vuelo (en <0.2 ms vía cv2.remap) las imágenes canónicas de
dataset_masivo_continuo/frames (capturadas en x=2.0, y=0.0, z=1.6, pitch=-15.0, fov=110.0)
hacia la perspectiva de una cámara posicionada en el centro geométrico del vehículo:
  - Posición: x=0.0, y=0.0, z=1.6
  - Orientación: pitch=-6.0, yaw=0.0, roll=0.0
  - Óptica: 1920x1080, FOV=85.0°, recorte inferior CropBottomY=680 (para librar cofre/techo)
  - Escalado directo a (320, 240) para PilotNetSteeringOnly.
"""

import os
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def worker_init_fn(_worker_id):
    cv2.setNumThreads(0)


# ==========================================
# 1. MAPEO EXACTO C1 (x=2.0m) -> C2 CENTRO GEOMÉTRICO (x=0.0m)
# ==========================================
def build_direct_center_camera_maps(
    src_w: int = 1920,
    src_h: int = 1080,
    fov1_deg: float = 110.0,
    cam1_pos=(2.0, 0.0, 1.6),
    cam1_pitch_deg: float = -15.0,
    cam2_pos=(0.0, 0.0, 1.6),
    cam2_pitch_deg: float = -6.0,
    fov2_deg: float = 85.0,
    crop_bottom_y: int = 680,
    out_w: int = 320,
    out_h: int = 240,
):
    """Precalcula (map1, map2) de tamaño (out_h, out_w) = (240, 320) para cv2.remap."""
    f1 = (src_w / 2.0) / np.tan(np.radians(fov1_deg / 2.0))
    f2 = (src_w / 2.0) / np.tan(np.radians(fov2_deg / 2.0))
    cx, cy = src_w / 2.0, src_h / 2.0

    row_grid, col_grid = np.indices((out_h, out_w), dtype=np.float32)
    u2 = (col_grid + 0.5) * (src_w / float(out_w)) - 0.5
    v2 = (row_grid + 0.5) * (crop_bottom_y / float(out_h)) - 0.5

    dx_c = (u2 - cx) / f2
    dy_c = (v2 - cy) / f2

    alpha2 = np.radians(-cam2_pitch_deg)
    cos2, sin2 = np.cos(alpha2), np.sin(alpha2)

    Rx = cos2 - dy_c * sin2
    Ry = dx_c
    Rz = -sin2 - dy_c * cos2

    inv_t_ground = (-Rz) / cam2_pos[2]
    min_inv_t = 1.0 / 60.0
    scale = 0.02
    inv_t = min_inv_t + scale * np.log1p(np.exp((inv_t_ground - min_inv_t) / scale))
    t = 1.0 / inv_t

    X_w = cam2_pos[0] + t * Rx
    Y_w = cam2_pos[1] + t * Ry
    Z_w = cam2_pos[2] + t * Rz

    dX1 = X_w - cam1_pos[0]
    dY1 = Y_w - cam1_pos[1]
    dZ1 = Z_w - cam1_pos[2]

    alpha1 = np.radians(-cam1_pitch_deg)
    cos1, sin1 = np.cos(alpha1), np.sin(alpha1)

    z_c1 = dX1 * cos1 - dZ1 * sin1
    x_c1 = dY1
    y_c1 = -dX1 * sin1 - dZ1 * cos1

    valid = z_c1 > 0.1
    z_c1_safe = np.where(valid, z_c1, 0.1)

    u1 = cx + f1 * (x_c1 / z_c1_safe)
    v1 = cy + f1 * (y_c1 / z_c1_safe)

    map_x = np.where(valid, u1, -1.0).astype(np.float32)
    map_y = np.where(valid, v1, -1.0).astype(np.float32)
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)


# ==========================================
# 2. DATASET ULTRA-RÁPIDO (UINT8 -> GPU NORMALIZE)
# ==========================================
class CenterCameraDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, frames_dir: str, map1, map2):
        df = dataframe.reset_index(drop=True)
        self.img_paths = [
            os.path.join(frames_dir, os.path.basename(p))
            for p in df["image_path"].tolist()
        ]
        self.steerings = df["steering"].to_numpy(dtype=np.float32)
        self.map1 = map1
        self.map2 = map2

    def __len__(self):
        return len(self.steerings)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

        # Reproyección al centro geométrico (x=0.0, y=0.0, z=1.6) + Resize a (240, 320) en 1 paso
        warped_bgr = cv2.remap(
            image,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
        chw_uint8 = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        return torch.from_numpy(chw_uint8), torch.tensor(
            [self.steerings[idx]], dtype=torch.float32
        )


# ==========================================
# 3. ARQUITECTURA PILOTNET (STEERING ONLY)
# ==========================================
class PilotNetSteeringOnly(nn.Module):
    def __init__(self, input_shape=(3, 240, 320)):
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
            nn.ELU(),
        )
        self._to_linear = None
        self._calculate_flatten_size(input_shape)
        self.fc1 = nn.Linear(self._to_linear, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.output_layer = nn.Linear(10, 1)

    def _calculate_flatten_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv_layers(x)
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, image):
        x = self.conv_layers(image)
        x = x.reshape(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        return self.output_layer(x)


def weighted_steering_loss(pred_steering, true_steering, base_weight=1.0, curve_multiplier=12.0):
    raw_loss = F.huber_loss(pred_steering, true_steering, reduction="none")
    weights = base_weight + (torch.abs(true_steering) * curve_multiplier)
    return torch.mean(raw_loss * weights)


def main():
    CSV_FILE = "/home/daniel/code/2025-phd-daniel-guerrero/data/dataset_masivo_continuo/driving_log_bev_best.csv"
    FRAMES_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/data/dataset_masivo_continuo/frames"
    MODEL_SAVE_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/models/frontal/modelo_pilotnet_centro_geometrico.pth"

    BATCH_SIZE = 128
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WORKERS = 12

    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    mean_gpu = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    print(f"[INFO] Usando dispositivo: {device}")
    print("[INFO] Precalculando mapas de transformación al Centro Geométrico (x=0.0, y=0.0, z=1.6, FOV=85°, pitch=-6°)...")
    map1, map2 = build_direct_center_camera_maps()

    df = pd.read_csv(CSV_FILE)
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_size = int(0.85 * len(df_shuffled))
    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]

    train_dataset = CenterCameraDataset(df_train, FRAMES_DIR, map1, map2)
    val_dataset = CenterCameraDataset(df_val, FRAMES_DIR, map1, map2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )

    model = PilotNetSteeringOnly().to(device, memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    best_val_loss = float("inf")

    print(f"[INFO] Iniciando entrenamiento ({EPOCHS} épocas) sobre {len(train_dataset):,} Train / {len(val_dataset):,} Val frames...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_gpu = torch.zeros(1, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS:02d} [Train Centro Geométrico]")
        for step, (images_u8, steerings) in enumerate(pbar, 1):
            images = images_u8.to(device, dtype=torch.float32, memory_format=torch.channels_last, non_blocking=True)
            images.mul_(1.0 / 255.0).sub_(mean_gpu).div_(std_gpu)
            steerings = steerings.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)
                loss = weighted_steering_loss(preds, steerings, curve_multiplier=12.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_gpu += loss.detach()
            if step % 25 == 0 or step == len(train_loader):
                pbar.set_postfix({"avg_loss": f"{(train_loss_gpu.item() / step):.4f}"})

        model.eval()
        val_loss_gpu = torch.zeros(1, device=device)
        with torch.no_grad():
            for images_u8, steerings in val_loader:
                images = images_u8.to(device, dtype=torch.float32, memory_format=torch.channels_last, non_blocking=True)
                images.mul_(1.0 / 255.0).sub_(mean_gpu).div_(std_gpu)
                steerings = steerings.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(images)
                    loss = weighted_steering_loss(preds, steerings, curve_multiplier=12.0)
                val_loss_gpu += loss.detach()

        avg_train_loss = train_loss_gpu.item() / len(train_loader)
        avg_val_loss = val_loss_gpu.item() / len(val_loader)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1:02d} Result | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}",
            flush=True,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> [BEST] Guardado mejor modelo (Val Loss: {best_val_loss:.4f}) en {MODEL_SAVE_PATH}", flush=True)

    # También guardar el modelo final en caso de que la última época tenga el mejor ajuste sobre la ruta completa
    final_path = MODEL_SAVE_PATH.replace(".pth", "_last.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[SUCCESS] Entrenamiento finalizado. Mejor modelo en: {MODEL_SAVE_PATH}", flush=True)


if __name__ == "__main__":
    main()
