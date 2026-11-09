#!/usr/bin/env python3
"""Entrenamiento Ultra-Optimizado para AMD Radeon RX 6750 XT (12GB VRAM, ROCm 6.3) + 24 hilos CPU.

Optimizaciones de hardware aplicadas:
  1. Pipeline C++ OpenCV + Transferencia uint8 por PCIe (4x menos ancho de banda
  PCIe):
     - Los workers de CPU leen y redimensionan con cv2.INTER_AREA en uint8 y
     entregan tensores uint8.
     - La conversión a float32 y normalización ImageNet ocurren 100% dentro de
     la GPU en paralelo.
  2. Formato de memoria `torch.channels_last` (NHWC) + `AMP FP16` +
  `cudnn.benchmark = True`:
     - Maximiza el throughput de los kernels MIOpen en arquitectura RDNA2.
  3. Cero bloqueos CPU<->GPU por batch:
     - Acumula el loss en un tensor en VRAM y solo sincroniza con CPU cada 20
     batches para el tqdm.
  4. Saturación de 24 hilos CPU y 12 GB VRAM:
     - `batch_size=256`, `workers=16`, `prefetch_factor=4`,
     `persistent_workers=True`, `drop_last=True`,
       y `cv2.setNumThreads(0)` por worker para evitar contención de hilos.
  5. Soporte nativo para el dataset de Lane-Following Suavizado (`--lanefollow`)
  y Fine-Tuning (`--pretrained`).
"""

import argparse
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
from torchvision import models
from tqdm import tqdm


def worker_init_fn(_worker_id):
  # Evita que cada uno de los 16 workers cree múltiples hilos OpenCV internos
  cv2.setNumThreads(0)


# ==========================================
# 1. DATASET ULTRA-RÁPIDO (UINT8 -> GPU NORMALIZE)
# ==========================================
class FastBench2DriveFrontalDataset(Dataset):

  def __init__(
      self,
      csv_file: str,
      base_dir: str,
      augment: bool = False,
      crop_top_ratio: float = 0.35,
  ):
    df = pd.read_csv(csv_file)
    # Pre-construir arreglos en RAM para evitar overhead de pandas.iloc en cada __getitem__
    self.img_paths = [
        os.path.join(base_dir, p) for p in df["image_path"].tolist()
    ]
    self.steerings = df["steering"].to_numpy(dtype=np.float32)
    self.augment = augment
    self.crop_top_ratio = crop_top_ratio

  def __len__(self):
    return len(self.steerings)

  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
      raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

    if self.crop_top_ratio > 0.0:
      y0 = int(image.shape[0] * self.crop_top_ratio)
      image = image[y0:, :, :]

    # Resize directo en C++ OpenCV sobre uint8 (mucho más rápido que PIL)
    resized = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    # BGR -> RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    steer_val = self.steerings[idx]

    if self.augment and np.random.rand() < 0.5:
      rgb = cv2.flip(rgb, 1)
      steer_val = -steer_val

    # HWC -> CHW como uint8 contiguo (4x más ligero de transferir por PCIe que float32)
    chw_uint8 = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(chw_uint8), torch.tensor(
        [steer_val], dtype=torch.float32
    )


# ==========================================
# 2. ARQUITECTURAS (PILOTNET & RESNET-18)
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
    x = x.reshape(x.size(0), -1)
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
# 3. FUNCIÓN DE PÉRDIDA PONDERADA SUAVE
# ==========================================
def weighted_steering_loss(
    pred_steering, true_steering, base_weight=1.0, curve_multiplier=2.0
):
  raw_loss = F.huber_loss(
      pred_steering, true_steering, reduction="none", delta=0.05
  )
  weights = base_weight + (torch.abs(true_steering) * curve_multiplier)
  return torch.mean(raw_loss * weights)


# ==========================================
# 4. BUCLE PRINCIPAL DE ENTRENAMIENTO GPU-MAX
# ==========================================
def main():
  parser = argparse.ArgumentParser(
      description="Entrenamiento GPU-Max en Bench2Drive_Light"
  )
  parser.add_argument(
      "--base-dir", default="/home/daniel/datasets/Bench2Drive_Light"
  )
  parser.add_argument(
      "--arch", choices=["resnet18", "pilotnet"], default="resnet18"
  )
  parser.add_argument(
      "--lanefollow",
      action="store_true",
      default=True,
      help=(
          "Usar los CSVs limpios y suavizados de Lane-Following"
          " (train_lanefollow_log.csv)"
      ),
  )
  parser.add_argument(
      "--pretrained",
      default="/home/daniel/code/2025-phd-daniel-guerrero/models/frontal/modelo_resnet18_bench2drive_frontal.pth",
      help="Ruta a pesos previos para Fine-Tuning (vacío para desde cero)",
  )
  parser.add_argument("--crop-top", type=float, default=0.35)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--lr", type=float, default=1.5e-4)
  parser.add_argument("--curve-mult", type=float, default=2.0)
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument("--poweroff", action="store_true")
  args = parser.parse_args()

  # En ROCm/MIOpen, cudnn.benchmark=False + MIOPEN_FIND_MODE=FAST usa la Find-DB precompilada
  # instantáneamente sin detenerse 10 minutos haciendo búsqueda exhaustiva por capa.
  torch.backends.cudnn.benchmark = False
  cv2.setNumThreads(0)

  if args.lanefollow and os.path.exists(
      os.path.join(args.base_dir, "train_lanefollow_log.csv")
  ):
    train_csv = os.path.join(args.base_dir, "train_lanefollow_log.csv")
    val_csv = os.path.join(args.base_dir, "val_lanefollow_log.csv")
    suffix = "lanefollow"
  else:
    train_csv = os.path.join(args.base_dir, "train_driving_log.csv")
    val_csv = os.path.join(args.base_dir, "val_driving_log.csv")
    suffix = "frontal"

  save_dir = "/home/daniel/code/2025-phd-daniel-guerrero/models/frontal"
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(
      save_dir, f"modelo_{args.arch}_bench2drive_{suffix}.pth"
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  use_amp = device.type == "cuda"

  # Constantes de normalización en VRAM (formato channels_last)
  mean_gpu = torch.tensor(
      [0.485, 0.456, 0.406], device=device, dtype=torch.float32
  ).view(1, 3, 1, 1)
  std_gpu = torch.tensor(
      [0.229, 0.224, 0.225], device=device, dtype=torch.float32
  ).view(1, 3, 1, 1)

  print(
      f"[INFO] Dispositivo: {device} ({torch.cuda.get_device_name(0) if use_amp else 'CPU'})"
  )
  print(
      f"[INFO] Config GPU-Max: Batch={args.batch_size} | Workers={args.workers}"
      f" | AMP FP16={use_amp} | ChannelsLast=True"
  )
  print(f"[INFO] Dataset: {train_csv}")

  train_dataset = FastBench2DriveFrontalDataset(
      train_csv, args.base_dir, augment=True, crop_top_ratio=args.crop_top
  )
  val_dataset = FastBench2DriveFrontalDataset(
      val_csv, args.base_dir, augment=False, crop_top_ratio=args.crop_top
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.workers,
      pin_memory=True,
      drop_last=True,
      persistent_workers=(args.workers > 0),
      prefetch_factor=4 if args.workers > 0 else None,
      worker_init_fn=worker_init_fn,
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.workers,
      pin_memory=True,
      drop_last=False,
      persistent_workers=(args.workers > 0),
      prefetch_factor=4 if args.workers > 0 else None,
      worker_init_fn=worker_init_fn,
  )

  if args.arch == "resnet18":
    model = ResNet18SteeringOnly()
  else:
    model = PilotNetSteeringOnly()

  if args.pretrained and os.path.exists(args.pretrained):
    print(f"[INFO] Cargando pesos preentrenados para Fine-Tuning: {args.pretrained}")
    model.load_state_dict(
        torch.load(args.pretrained, map_location="cpu", weights_only=True)
    )

  model = model.to(device, memory_format=torch.channels_last)

  optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=args.epochs, eta_min=1e-6
  )
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

  best_val_loss = float("inf")
  print(
      f"[INFO] Entrenando {args.epochs} épocas sobre {len(train_dataset):,} frames"
      f" Train | {len(val_dataset):,} frames Val..."
  )

  for epoch in range(args.epochs):
    model.train()
    train_loss_gpu = torch.zeros(1, device=device)

    pbar = tqdm(
        train_loader,
        desc=f"Época {epoch+1:02d}/{args.epochs:02d} [Train {args.arch}]",
    )
    for step, (images_u8, steerings) in enumerate(pbar, 1):
      # Transferencia uint8 no bloqueante + normalización instantánea en GPU
      images = images_u8.to(
          device,
          dtype=torch.float32,
          memory_format=torch.channels_last,
          non_blocking=True,
      )
      images.mul_(1.0 / 255.0).sub_(mean_gpu).div_(std_gpu)
      steerings = steerings.to(device, non_blocking=True)

      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", enabled=use_amp):
        preds = model(images)
        loss = weighted_steering_loss(
            preds, steerings, curve_multiplier=args.curve_mult
        )

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      train_loss_gpu += loss.detach()
      if step % 20 == 0 or step == len(train_loader):
        pbar.set_postfix(
            {"avg_loss": f"{(train_loss_gpu.item() / step):.5f}"}
        )

    model.eval()
    val_loss_gpu = torch.zeros(1, device=device)
    with torch.no_grad():
      for images_u8, steerings in val_loader:
        images = images_u8.to(
            device,
            dtype=torch.float32,
            memory_format=torch.channels_last,
            non_blocking=True,
        )
        images.mul_(1.0 / 255.0).sub_(mean_gpu).div_(std_gpu)
        steerings = steerings.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
          preds = model(images)
          loss = weighted_steering_loss(
              preds, steerings, curve_multiplier=args.curve_mult
          )
        val_loss_gpu += loss.detach()

    avg_train = train_loss_gpu.item() / len(train_loader)
    avg_val = val_loss_gpu.item() / len(val_loader)
    scheduler.step()

    print(
        f"Época {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train:.5f} |"
        f" Val Loss: {avg_val:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}",
        flush=True,
    )

    if avg_val < best_val_loss:
      best_val_loss = avg_val
      torch.save(model.state_dict(), save_path)
      print(
          f"  --> [BEST] Guardado mejor modelo (Val Loss: {best_val_loss:.5f})"
          f" en {save_path}",
          flush=True,
      )

  print(
      f"\n[ÉXITO] Entrenamiento finalizado. Mejor Val Loss: {best_val_loss:.5f}"
      f" -> {save_path}",
      flush=True,
  )

  if args.poweroff:
    print("[INFO] Apagando el equipo...", flush=True)
    os.system(
        "sudo -n poweroff 2>/dev/null || systemctl poweroff -i 2>/dev/null ||"
        " poweroff 2>/dev/null"
    )


if __name__ == "__main__":
  main()
