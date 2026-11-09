#!/usr/bin/env python3
"""Entrenamiento sobre Bench2Drive_Light con Transformación BEV Bifocal AL VUELO (On-the-fly).

- No requiere guardar copias BEV en disco (ahorra >30 GB).
- Precalcula los mapas cv2.remap (1600x900 -> 320x240 BEV directamente) una sola
vez en RAM.
- Soporta PilotNet (--arch pilotnet) y ResNet-18 (--arch resnet18).
- Usa split por clip (train_driving_log.csv y val_driving_log.csv), Flip
Simétrico y AMP FP16.
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ==========================================
# 1. MAPAS PRECALCULADOS BEV BIFOCAL UNIVERSAL
# ==========================================
def build_bifocal_bev_maps(
    img_w: int = 1600,
    img_h: int = 900,
    bev_w: int = 320,
    bev_h: int = 240,
    split_ratio: float = 0.45,
    transition_px: float = 10.0,
):
  """Precalcula los mapas (map1, map2) directamente al tamaño de entrada de la red (320x240)

  para que cv2.remap haga el BEV + Resize en una sola operación ultrarrápida
  (<0.3 ms).
  """
  split_y = split_ratio * bev_h

  p_left_seam = [0.39 * img_w, 0.385 * img_h]
  p_right_seam = [0.61 * img_w, 0.385 * img_h]
  p_left_bot = [-0.25 * img_w, 0.98 * img_h]
  p_right_bot = [1.25 * img_w, 0.98 * img_h]

  src_bot = np.float32([p_left_bot, p_right_bot, p_right_seam, p_left_seam])
  dst_bot = np.float32(
      [[0, bev_h], [bev_w, bev_h], [bev_w, split_y], [0, split_y]]
  )
  H_bot_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_bot, dst_bot))

  src_top = np.float32([
      p_left_seam,
      p_right_seam,
      [0.76 * img_w, 0.24 * img_h],
      [0.24 * img_w, 0.24 * img_h],
  ])
  dst_top = np.float32([[0, split_y], [bev_w, split_y], [bev_w, 0], [0, 0]])
  H_top_inv = np.linalg.inv(cv2.getPerspectiveTransform(src_top, dst_top))

  y_grid, x_grid = np.indices((bev_h, bev_w), dtype=np.float32)
  coords = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1).reshape(
      -1, 3
  )

  c_bot = coords @ H_bot_inv.T
  w_bot = np.where(np.abs(c_bot[:, 2]) > 1e-8, c_bot[:, 2], 1e-8)
  u_bot = (c_bot[:, 0] / w_bot).reshape(bev_h, bev_w)
  v_bot = (c_bot[:, 1] / w_bot).reshape(bev_h, bev_w)

  c_top = coords @ H_top_inv.T
  w_top = np.where(np.abs(c_top[:, 2]) > 1e-8, c_top[:, 2], 1e-8)
  u_top = (c_top[:, 0] / w_top).reshape(bev_h, bev_w)
  v_top = (c_top[:, 1] / w_top).reshape(bev_h, bev_w)

  alpha_blend = 1.0 / (1.0 + np.exp(-(y_grid - split_y) / transition_px))
  map_x = (alpha_blend * u_bot + (1.0 - alpha_blend) * u_top).astype(np.float32)
  map_y = (alpha_blend * v_bot + (1.0 - alpha_blend) * v_top).astype(np.float32)

  return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)


# ==========================================
# 2. DATASET CON BEV AL VUELO
# ==========================================
class Bench2DriveOnTheFlyBEVDataset(Dataset):

  def __init__(self, csv_file: str, base_dir: str, map1, map2, augment: bool = False):
    self.data = pd.read_csv(csv_file)
    self.base_dir = base_dir
    self.map1 = map1
    self.map2 = map2
    self.augment = augment

    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    img_path = os.path.join(self.base_dir, row["image_path"])

    image = cv2.imread(img_path)
    if image is None:
      raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

    # Transformación BEV Bifocal + Resize a (240, 320) en un solo paso
    bev = cv2.remap(
        image,
        self.map1,
        self.map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
    steer_val = float(row["steering"])

    if self.augment and torch.rand(1).item() < 0.5:
      bev = cv2.flip(bev, 1)
      steer_val = -steer_val

    image_tensor = self.transform(bev)
    steering_out = torch.tensor([steer_val], dtype=torch.float32)
    return image_tensor, steering_out


# ==========================================
# 3. ARQUITECTURAS (PILOTNET & RESNET-18)
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
    with torch.no_grad():
      dummy = torch.zeros(1, *input_shape)
      flat_dim = self.conv_layers(dummy).view(1, -1).size(1)

    self.fc1 = nn.Linear(flat_dim, 100)
    self.fc2 = nn.Linear(100, 50)
    self.fc3 = nn.Linear(50, 10)
    self.output_layer = nn.Linear(10, 1)

  def forward(self, x):
    x = self.conv_layers(x)
    x = x.view(x.size(0), -1)
    x = F.elu(self.fc1(x))
    x = F.elu(self.fc2(x))
    x = F.elu(self.fc3(x))
    return self.output_layer(x)


class ResNet18SteeringOnly(nn.Module):

  def __init__(self):
    super().__init__()
    self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = self.cnn.fc.in_features
    self.cnn.fc = nn.Identity()

    self.fc1 = nn.Linear(num_features, 128)
    self.fc2 = nn.Linear(128, 64)
    self.output_layer = nn.Linear(64, 1)

  def forward(self, x):
    x = self.cnn(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.output_layer(x)


# ==========================================
# 4. PÉRDIDA PONDERADA
# ==========================================
def weighted_steering_loss(
    pred_steering, true_steering, base_weight=1.0, curve_multiplier=12.0
):
  raw_loss = F.huber_loss(pred_steering, true_steering, reduction="none")
  weights = base_weight + (torch.abs(true_steering) * curve_multiplier)
  return torch.mean(raw_loss * weights)


# ==========================================
# 5. BUCLE PRINCIPAL DE ENTRENAMIENTO
# ==========================================
def main():
  parser = argparse.ArgumentParser(
      description="Entrenar modelo BEV al vuelo en Bench2Drive_Light"
  )
  parser.add_argument(
      "--base-dir", default="/home/daniel/datasets/Bench2Drive_Light"
  )
  parser.add_argument(
      "--arch",
      choices=["pilotnet", "resnet18"],
      default="pilotnet",
      help="Arquitectura a entrenar (default: pilotnet)",
  )
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--epochs", type=int, default=15)
  parser.add_argument("--lr", type=float, default=1.5e-4)
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument(
      "--poweroff",
      action="store_true",
      help="Intentar apagar el equipo al finalizar el entrenamiento",
  )
  args = parser.parse_args()

  torch.backends.cudnn.benchmark = True

  train_csv = os.path.join(args.base_dir, "train_driving_log.csv")
  val_csv = os.path.join(args.base_dir, "val_driving_log.csv")
  save_dir = "/home/daniel/code/2025-phd-daniel-guerrero/models/bev"
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, f"modelo_{args.arch}_bench2drive_bev.pth")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  use_amp = device.type == "cuda"
  print(
      f"[INFO] Dispositivo: {device} | AMP FP16: {use_amp} | Arquitectura:"
      f" {args.arch.upper()} (BEV On-The-Fly)"
  )

  print(
      "[INFO] Precalculando mapas BEV Bifocal (1600x900 -> 320x240) en RAM..."
  )
  map1, map2 = build_bifocal_bev_maps(
      img_w=1600, img_h=900, bev_w=320, bev_h=240
  )

  train_dataset = Bench2DriveOnTheFlyBEVDataset(
      train_csv, args.base_dir, map1, map2, augment=True
  )
  val_dataset = Bench2DriveOnTheFlyBEVDataset(
      val_csv, args.base_dir, map1, map2, augment=False
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.workers,
      pin_memory=True,
      drop_last=True,
      persistent_workers=(args.workers > 0),
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.workers,
      pin_memory=True,
      drop_last=True,
      persistent_workers=(args.workers > 0),
  )

  if args.arch == "resnet18":
    model = ResNet18SteeringOnly().to(device)
  else:
    model = PilotNetSteeringOnly().to(device)

  optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=args.epochs, eta_min=1e-6
  )
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

  best_val_loss = float("inf")
  print(
      f"[INFO] Entrenando con {len(train_dataset):,} frames Train |"
      f" {len(val_dataset):,} frames Val (split por clip)..."
  )

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    pbar = tqdm(
        train_loader,
        desc=f"Época {epoch+1}/{args.epochs} [Train BEV {args.arch}]",
    )
    for images, steerings in pbar:
      images = images.to(device, non_blocking=True)
      steerings = steerings.to(device, non_blocking=True)

      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", enabled=use_amp):
        preds = model(images)
        loss = weighted_steering_loss(preds, steerings, curve_multiplier=12.0)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      train_loss += loss.item()
      pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for images, steerings in val_loader:
        images = images.to(device, non_blocking=True)
        steerings = steerings.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
          preds = model(images)
          loss = weighted_steering_loss(preds, steerings, curve_multiplier=12.0)
        val_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    scheduler.step()

    print(
        f"Época {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train:.4f} |"
        f" Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
    )

    if avg_val < best_val_loss:
      best_val_loss = avg_val
      torch.save(model.state_dict(), save_path)
      print(
          f"  --> [BEST] Guardado mejor modelo (Val Loss: {best_val_loss:.4f})"
          f" en {save_path}"
      )

  print(
      f"\n[ÉXITO] Entrenamiento completado. Mejor Val Loss: {best_val_loss:.4f}"
      f" -> {save_path}",
      flush=True,
  )

  if args.poweroff:
    print("[INFO] Apagando el equipo tras finalizar el entrenamiento...", flush=True)
    os.system("sudo -n poweroff 2>/dev/null || systemctl poweroff -i 2>/dev/null || poweroff 2>/dev/null")


if __name__ == "__main__":
  main()
