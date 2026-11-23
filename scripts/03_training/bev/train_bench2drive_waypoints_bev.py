#!/usr/bin/env python3
"""Entrenamiento GPU-Max de Seguimiento de Carril por Waypoints sobre BEV Métrico Canónico (Puente entre Vehículos).

Diseñado como puente agnóstico al vehículo/cámara entre:
  - Dataset Público Bench2Drive (Lincoln MKZ 2020: 1600x900, FOV=70°, x=0.8m, z=1.6m, pitch=0.0°)
  - Inferencia en BehaviorMetrics con Tesla Model 3 (ej. 1920x1080, FOV=110°, x=2.0m, z=1.6m, pitch=-15.0°
    o cualquier otra configuración de cámara definida en el YAML).

Características principales:
  1. Grilla BEV Métrica Canónica (X_ego en [4.8m .. 22.0m], Y_ego en [-5.0m .. +5.0m]):
     - Mapeo bifocal suave (55% inferior para campo cercano 4.8m..11.5m, 45% superior para 11.5m..22.0m).
     - Máscara de Cono Común (Common Frustum Mask de 70°) para que cámaras de distinto FOV (70° vs 110°)
       produzcan exactamente los mismos bordes válidos sin domain gap.
  2. Objetivo Geométrico Universal (10 Waypoints por longitud de arco wpa1..wpa10, 2m..20m):
     - Coordenadas en metros respecto al centro del chasis (x_fwd, y_right), 100% independientes del vehículo.
  3. Arquitectura ResNet18Waypoints con Spatial Bottleneck (4x6) + Pipeline GPU-Max (ROCm 6.3, AMP FP16, channels_last).
"""

import argparse
import json
import math
import os
import signal

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen_db")
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", "/tmp/miopen_cache")
os.makedirs("/tmp/miopen_db", exist_ok=True)
os.makedirs("/tmp/miopen_cache", exist_ok=True)
try:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
except Exception:
    pass

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
import timm


def worker_init_fn(_worker_id):
    cv2.setNumThreads(0)


# ==========================================
# 1. PUENTE GEOMÉTRICO: BEV MÉTRICO CANÓNICO ENTRE VEHÍCULOS
# ==========================================
def build_canonical_metric_bev_grid(
    bev_w: int = 256,
    bev_h: int = 256,
    x_min_m: float = 4.85,
    x_mid_m: float = 11.5,
    x_max_m: float = 22.0,
    y_half_width_m: float = 5.0,
    split_ratio: float = 0.45,
):
    """Construye las matrices (X_ego, Y_ego) en metros reales respecto al centro del vehículo (0,0)
    para cada píxel (v_bev, u_bev) del lienzo BEV canónico.
      - Fila v=0 (arriba): X_ego = x_max_m (22.0 m adelante)
      - Fila v=split_y:    X_ego = x_mid_m (11.5 m adelante)
      - Fila v=bev_h-1:    X_ego = x_min_m (4.85 m adelante)
      - Columna u=0..bev_w-1: Y_ego de -y_half_width_m (izquierda) a +y_half_width_m (derecha)
    """
    v_idx, u_idx = np.indices((bev_h, bev_w), dtype=np.float32)
    split_v = split_ratio * float(bev_h - 1)

    # Perfil longitudinal bifocal continuo y suave
    x_top = x_max_m - (v_idx / max(split_v, 1.0)) * (x_max_m - x_mid_m)
    x_bot = x_mid_m - ((v_idx - split_v) / max(float(bev_h - 1) - split_v, 1.0)) * (
        x_mid_m - x_min_m
    )
    alpha = 1.0 / (1.0 + np.exp(-(v_idx - split_v) / (0.04 * bev_h)))
    x_ego = (alpha * x_bot + (1.0 - alpha) * x_top).astype(np.float32)

    y_ego = (
        (u_idx / float(bev_w - 1) - 0.5) * (2.0 * y_half_width_m)
    ).astype(np.float32)
    return x_ego, y_ego


def build_canonical_metric_bev_maps(
    img_w: int = 1600,
    img_h: int = 900,
    fov_deg: float = 70.0,
    cam_x: float = 0.8,
    cam_y: float = 0.0,
    cam_z: float = 1.6,
    pitch_deg: float = 0.0,
    bev_w: int = 256,
    bev_h: int = 256,
    x_min_m: float = 4.85,
    x_mid_m: float = 11.5,
    x_max_m: float = 22.0,
    y_half_width_m: float = 5.0,
    split_ratio: float = 0.45,
    apply_common_frustum: bool = True,
    ref_fov_deg: float = 70.0,
    ref_cam_x: float = 0.8,
):
    """Genera los mapas (map1, map2) de cv2.remap para transformar la imagen de CUALQUIER cámara/vehículo
    al mismo BEV Métrico Canónico de tamaño (bev_w, bev_h).
    """
    x_ego, y_ego = build_canonical_metric_bev_grid(
        bev_w=bev_w,
        bev_h=bev_h,
        x_min_m=x_min_m,
        x_mid_m=x_mid_m,
        x_max_m=x_max_m,
        y_half_width_m=y_half_width_m,
        split_ratio=split_ratio,
    )

    f_px = (0.5 * float(img_w)) / math.tan(math.radians(0.5 * float(fov_deg)))
    cx = 0.5 * float(img_w)
    cy = 0.5 * float(img_h)

    # En CARLA, pitch negativo mira hacia abajo (ej. -15° -> tilt_down = +15°)
    tilt_down = math.radians(-float(pitch_deg))
    cos_t = math.cos(tilt_down)
    sin_t = math.sin(tilt_down)

    dx = x_ego - float(cam_x)
    dy = y_ego - float(cam_y)
    dz_down = float(cam_z)  # El suelo Z=0 está a +cam_z hacia abajo de la cámara

    z_cam = dx * cos_t + dz_down * sin_t
    y_cam_down = -dx * sin_t + dz_down * cos_t
    x_cam_right = dy

    z_safe = np.where(z_cam > 0.1, z_cam, 0.1)
    u_img = (cx + f_px * (x_cam_right / z_safe)).astype(np.float32)
    v_img = (cy + f_px * (y_cam_down / z_safe)).astype(np.float32)

    # Máscara de Frustum Común (iguala el cono de 70° del dataset con el de 110° del Tesla Model 3)
    if apply_common_frustum:
        max_lat = (x_ego - float(ref_cam_x)) * math.tan(
            math.radians(0.5 * float(ref_fov_deg))
        )
        outside = (x_ego < x_min_m) | (np.abs(y_ego) > max_lat) | (z_cam <= 0.1)
        u_img[outside] = -10.0
        v_img[outside] = -10.0

    return cv2.convertMaps(u_img, v_img, cv2.CV_16SC2)


def ego_xy_to_canonical_bev_px(
    x_fwd: float,
    y_right: float,
    bev_w: int = 256,
    bev_h: int = 256,
    x_min_m: float = 4.85,
    x_mid_m: float = 11.5,
    x_max_m: float = 22.0,
    y_half_width_m: float = 5.0,
    split_ratio: float = 0.45,
):
    """Convierte una coordenada métrica del suelo (x_fwd, y_right) a píxeles (u_bev, v_bev) en el BEV canónico."""
    u = int(round(((y_right / (2.0 * y_half_width_m)) + 0.5) * float(bev_w - 1)))
    split_v = split_ratio * float(bev_h - 1)
    if x_fwd >= x_mid_m:
        frac = (x_max_m - x_fwd) / max(x_max_m - x_mid_m, 1e-6)
        v = int(round(frac * split_v))
    else:
        frac = (x_mid_m - x_fwd) / max(x_mid_m - x_min_m, 1e-6)
        v = int(round(split_v + frac * (float(bev_h - 1) - split_v)))
    return u, v


# ==========================================
# 2. DATASET GPU-MAX ON-THE-FLY BEV + WAYPOINTS
# ==========================================
WP_COLS = [f"wpa{k}_{axis}" for k in range(1, 11) for axis in ("x", "y")]


class FastBench2DriveWaypointsBEVDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        base_dir: str,
        map1: np.ndarray,
        map2: np.ndarray,
        augment: bool = False,
        flip_prob: float = 0.5,
    ):
        df = pd.read_csv(csv_file)
        self.img_paths = [os.path.join(base_dir, p) for p in df["image_path"].tolist()]
        self.waypoints = df[WP_COLS].to_numpy(dtype=np.float32).reshape(len(df), 10, 2)
        self.steerings = df["steering"].to_numpy(dtype=np.float32)
        self.map1 = map1
        self.map2 = map2
        self.augment = bool(augment)
        self.flip_prob = float(flip_prob)

    def __len__(self):
        return len(self.steerings)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

        # Remapeo directo en C++ de 1600x900 frontal -> 256x256 BEV Métrico Canónico (<0.25 ms)
        bev_bgr = cv2.remap(
            image,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        bev_rgb = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
        wp = self.waypoints[idx].copy()
        steer = float(self.steerings[idx])

        if self.augment and np.random.rand() < self.flip_prob:
            bev_rgb = cv2.flip(bev_rgb, 1)
            wp[:, 1] = -wp[:, 1]
            steer = -steer

        chw_uint8 = np.ascontiguousarray(bev_rgb.transpose(2, 0, 1))
        return (
            torch.from_numpy(chw_uint8),
            torch.from_numpy(wp),
            torch.tensor([steer], dtype=torch.float32),
        )


# ==========================================
# 3. ARQUITECTURA RESNET-18 SPATIAL WAYPOINTS
# ==========================================
class ResNet18Waypoints(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 10,
        pretrained_imagenet: bool = True,
        dropout: float = 0.1,
        init_arc_step: float = 2.0,
    ):
        super().__init__()
        self.n_waypoints = int(n_waypoints)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        backbone = models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.spatial_neck = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),
        )
        flat_dim = 64 * 4 * 6

        self.shared_mlp = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
        )
        self.wp_head = nn.Linear(128, self.n_waypoints * 2)
        self.steer_head = nn.Linear(128, 1)
        self._init_heads(init_arc_step)

    def _init_heads(self, init_arc_step: float):
        nn.init.normal_(self.wp_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.wp_head.bias)
        if init_arc_step > 0.0:
            with torch.no_grad():
                bias_xy = self.wp_head.bias.view(self.n_waypoints, 2)
                bias_xy[:, 0] = (
                    torch.arange(1, self.n_waypoints + 1, dtype=torch.float32)
                    * float(init_arc_step)
                )
        nn.init.normal_(self.steer_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.steer_head.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spatial_neck(x)
        x = x.reshape(x.size(0), -1)
        feat = self.shared_mlp(x)
        wp = self.wp_head(feat).view(-1, self.n_waypoints, 2)
        steer = self.steer_head(feat)
        return wp, steer


class FastViTWaypoints(nn.Module):
    def __init__(
        self,
        model_name: str = "fastvit_t8",
        n_waypoints: int = 10,
        dropout: float = 0.1,
        init_arc_step: float = 2.0,
    ):
        super().__init__()
        self.n_waypoints = int(n_waypoints)
        self.cnn = timm.create_model(model_name, pretrained=False, num_classes=0)
        num_features = int(self.cnn.num_features)  # 768 en fastvit_t8

        self.spatial_neck = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),
        )
        flat_dim = 64 * 4 * 6

        self.shared_mlp = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
        )
        self.wp_head = nn.Linear(128, self.n_waypoints * 2)
        self.steer_head = nn.Linear(128, 1)
        self._init_heads(init_arc_step)

    def _init_heads(self, init_arc_step: float):
        nn.init.normal_(self.wp_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.wp_head.bias)
        if init_arc_step > 0.0:
            with torch.no_grad():
                bias_xy = self.wp_head.bias.view(self.n_waypoints, 2)
                bias_xy[:, 0] = (
                    torch.arange(1, self.n_waypoints + 1, dtype=torch.float32)
                    * float(init_arc_step)
                )
        nn.init.normal_(self.steer_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.steer_head.bias)

    def forward(self, x):
        x = self.cnn.forward_features(x)
        x = self.spatial_neck(x)
        x = x.reshape(x.size(0), -1)
        feat = self.shared_mlp(x)
        wp = self.wp_head(feat).view(-1, self.n_waypoints, 2)
        steer = self.steer_head(feat)
        return wp, steer


# ==========================================
# 4. PÉRDIDA MULTI-HORIZONTE PONDERADA EN METROS
# ==========================================
class WaypointLaneFollowLoss(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 10,
        lateral_weight: float = 2.5,
        horizon_decay: float = 0.88,
        curve_gain: float = 1.5,
        smooth_weight: float = 0.05,
        aux_steer_weight: float = 0.15,
        huber_delta: float = 0.20,
    ):
        super().__init__()
        h_weights = torch.tensor(
            [horizon_decay**k for k in range(n_waypoints)], dtype=torch.float32
        )
        h_weights = h_weights / h_weights.mean()
        self.register_buffer("h_weights", h_weights.view(1, n_waypoints, 1))
        axis_weights = torch.tensor([1.0, lateral_weight], dtype=torch.float32).view(
            1, 1, 2
        )
        self.register_buffer("axis_weights", axis_weights)
        self.curve_gain = float(curve_gain)
        self.smooth_weight = float(smooth_weight)
        self.aux_steer_weight = float(aux_steer_weight)
        self.huber_delta = float(huber_delta)

    def forward(self, pred_wp, true_wp, pred_steer, true_steer):
        err = F.huber_loss(
            pred_wp, true_wp, reduction="none", delta=self.huber_delta
        )
        weighted_err = err * self.h_weights * self.axis_weights

        # Mayor peso a curvas moderadas según desplazamiento lateral a 10m (wpa5_y, índice 4)
        lat_10m = torch.clamp(true_wp[:, 4, 1].abs(), max=3.0)
        sample_w = (1.0 + self.curve_gain * (lat_10m / 3.0)).view(-1, 1, 1)
        wp_loss = (weighted_err * sample_w).mean()

        if pred_wp.size(1) >= 3 and self.smooth_weight > 0.0:
            d2 = pred_wp[:, 2:, :] - 2.0 * pred_wp[:, 1:-1, :] + pred_wp[:, :-2, :]
            smooth_loss = (d2.pow(2)).mean()
        else:
            smooth_loss = pred_wp.new_tensor(0.0)

        steer_loss = F.huber_loss(
            pred_steer, true_steer, reduction="mean", delta=0.05
        )
        total = (
            wp_loss
            + self.smooth_weight * smooth_loss
            + self.aux_steer_weight * steer_loss
        )
        return total


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento GPU-Max Waypoints + BEV Métrico Canónico en Bench2Drive"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/daniel/datasets/Bench2Drive_Light",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "fastvit"],
    )
    parser.add_argument("--lanefollow", action="store_true", default=True)
    parser.add_argument("--full-dataset", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--bev-size", type=int, default=256)
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/home/daniel/code/2025-phd-daniel-guerrero/models/bev",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    if args.full_dataset:
        train_csv = os.path.join(args.base_dir, "train_bench2drive_waypoints.csv")
        val_csv = os.path.join(args.base_dir, "val_bench2drive_waypoints.csv")
        model_tag = f"modelo_{args.arch}_bench2drive_waypoints_bev"
    else:
        train_csv = os.path.join(
            args.base_dir, "train_bench2drive_lanefollow_waypoints.csv"
        )
        val_csv = os.path.join(
            args.base_dir, "val_bench2drive_lanefollow_waypoints.csv"
        )
        model_tag = f"modelo_{args.arch}_bench2drive_lanefollow_waypoints_bev"

    # Precalcular mapas BEV Métricos Canónicos para la cámara de Bench2Drive (1600x900, FOV=70°, x=0.8, z=1.6, pitch=0°)
    map1, map2 = build_canonical_metric_bev_maps(
        img_w=1600,
        img_h=900,
        fov_deg=70.0,
        cam_x=0.8,
        cam_y=0.0,
        cam_z=1.6,
        pitch_deg=0.0,
        bev_w=args.bev_size,
        bev_h=args.bev_size,
    )

    train_ds = FastBench2DriveWaypointsBEVDataset(
        train_csv, args.base_dir, map1, map2, augment=True
    )
    val_ds = FastBench2DriveWaypointsBEVDataset(
        val_csv, args.base_dir, map1, map2, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(4, args.workers // 2),
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )

    if args.arch == "fastvit":
        model = FastViTWaypoints(n_waypoints=10).to(
            device, memory_format=torch.channels_last
        )
    else:
        model = ResNet18Waypoints(n_waypoints=10, pretrained_imagenet=True).to(
            device, memory_format=torch.channels_last
        )
    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        filtered_ckpt = {
            k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        model.load_state_dict(filtered_ckpt, strict=False)
        print(
            f"[INFO] Pesos preentrenados cargados ({len(filtered_ckpt)}/{len(model_dict)} tensores) desde: {args.pretrained}"
        )

    criterion = WaypointLaneFollowLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)

    best_val_loss = float("inf")
    best_path = os.path.join(args.save_dir, f"{model_tag}_best.pth")
    meta_path = os.path.join(args.save_dir, f"{model_tag}_meta.json")
    history = []

    print(
        f"[INFO] Entrenando ResNet18Waypoints (BEV Métrico Canónico {args.bev_size}x{args.bev_size}) "
        f"en {device} | Train: {len(train_ds):,} | Val: {len(val_ds):,} | Épocas: {args.epochs}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = torch.zeros(1, device=device)
        pbar = tqdm(
            train_loader,
            desc=f"Época {epoch:02d}/{args.epochs:02d} [Train BEV-WP]",
            leave=True,
        )
        for step, (imgs_u8, true_wp, true_steer) in enumerate(pbar, 1):
            imgs = (
                imgs_u8.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                ).div_(255.0)
            )
            imgs = (imgs - mean) / std
            true_wp = true_wp.to(device, non_blocking=True)
            true_steer = true_steer.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_wp, pred_steer = model(imgs)
                loss = criterion(pred_wp, true_wp, pred_steer, true_steer)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach()
            if step % 15 == 0 or step == len(train_loader):
                pbar.set_postfix(loss=f"{(running_loss.item() / step):.4f}")

        train_loss = running_loss.item() / max(len(train_loader), 1)
        scheduler.step()

        # Validación en metros físicos
        model.eval()
        val_loss_acc = 0.0
        ade_acc = 0.0
        lat6_acc = 0.0
        lat10_acc = 0.0
        fde20_acc = 0.0
        n_val_samples = 0

        with torch.inference_mode():
            for imgs_u8, true_wp, true_steer in val_loader:
                bs = imgs_u8.size(0)
                imgs = (
                    imgs_u8.to(
                        device,
                        dtype=torch.float32,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    ).div_(255.0)
                )
                imgs = (imgs - mean) / std
                true_wp = true_wp.to(device, non_blocking=True)
                true_steer = true_steer.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred_wp, pred_steer = model(imgs)
                    v_loss = criterion(pred_wp, true_wp, pred_steer, true_steer)

                diff = pred_wp.float() - true_wp.float()
                dist_per_wp = torch.linalg.norm(diff, dim=-1)  # [B, 10]
                val_loss_acc += v_loss.item() * bs
                ade_acc += dist_per_wp.mean(dim=-1).sum().item()
                lat6_acc += diff[:, 2, 1].abs().sum().item()   # wpa3 (6m)
                lat10_acc += diff[:, 4, 1].abs().sum().item()  # wpa5 (10m)
                fde20_acc += dist_per_wp[:, 9].sum().item()    # wpa10 (20m)
                n_val_samples += bs

        val_loss = val_loss_acc / max(n_val_samples, 1)
        ade = ade_acc / max(n_val_samples, 1)
        lat6 = lat6_acc / max(n_val_samples, 1)
        lat10 = lat10_acc / max(n_val_samples, 1)
        fde20 = fde20_acc / max(n_val_samples, 1)
        cur_lr = optimizer.param_groups[0]["lr"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        row_stat = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "ade_m": round(ade, 4),
            "lat_6m": round(lat6, 4),
            "lat_10m": round(lat10, 4),
            "fde_20m": round(fde20, 4),
            "lr": cur_lr,
            "best": is_best,
        }
        history.append(row_stat)
        with open(meta_path, "w") as f:
            json.dump({"best_path": best_path, "history": history}, f, indent=2)

        tag = " [BEST]" if is_best else ""
        print(
            f"  -> Época {epoch:02d}: Train={train_loss:.4f} | Val={val_loss:.4f} | "
            f"ADE={ade:.3f}m | Lat@6m={lat6:.3f}m | Lat@10m={lat10:.3f}m | FDE@20m={fde20:.3f}m{tag}"
        )


if __name__ == "__main__":
    main()
