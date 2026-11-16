#!/usr/bin/env python3
"""Entrenamiento GPU-Max de Seguimiento de Carril por Waypoints (dataset_waypoints).

Diseñado para AMD Radeon RX 6750 XT (12GB VRAM, ROCm) + 24 hilos CPU:
  1. Objetivo Geométrico Multi-Horizonte (Waypoints por Arco wpa1..wpa10, 2m..20m):
     - Invariante a la velocidad e inmune a pulsos discretos de volante.
     - Codifica explícitamente las trayectorias de centrado y recuperación DAgger.
  2. ResNet-18 con Cuello Espacial (Spatial Bottleneck) + Cabeza Auxiliar de Steering:
     - Reemplaza el Global AvgPool (1x1) por una reducción espacial 1x1 + grilla (4x6),
       preservando la localización horizontal y vertical de las líneas del carril.
  3. Partición Train/Val por Segmento Estratificada por Sesión (Sin Data Leakage):
     - Agrupa por (session, segment) para no mezclar cuadros consecutivos a 10 Hz
       y garantizar representación proporcional de Town01, Town02, Town03 y Town04.
  4. Pérdida Huber Ponderada por Horizonte, Eje Lateral (2.0x) y Curvatura Suave:
     - Prioriza el centrado en los primeros 2m..10m (lookahead de Pure Pursuit) y
       reporta errores físicos en metros (ADE, Error Lateral a 6m/10m y FDE a 20m).
  5. Pipeline GPU-Max (OpenCV C++ uint8 -> PCIe -> Normalización en VRAM + AMP FP16 + channels_last).
"""

import argparse
import json
import math
import os
import signal
import sys

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
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


def worker_init_fn(_worker_id):
    cv2.setNumThreads(0)


# ==========================================
# 1. UTILIDAD DE CONTROL PURE PURSUIT (PARA INFERENCIA)
# ==========================================
def waypoints_to_steer_pure_pursuit(
    waypoints_xy: np.ndarray,
    speed_kmh: float = 30.0,
    wheelbase_m: float = 2.87,
    max_steer_rad: float = 0.70,
    min_lookahead_m: float = 4.0,
    max_lookahead_m: float = 10.0,
    lookahead_gain: float = 0.22,
) -> float:
    """Convierte los 10 waypoints predichos (wpa1..wpa10 en el marco del ego: x=adelante, y=derecha)
    en un comando de dirección normalizado [-1.0, 1.0] mediante Pure Pursuit adaptativo.
    """
    wp = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
    ld = float(np.clip(lookahead_gain * float(speed_kmh), min_lookahead_m, max_lookahead_m))
    dists = np.hypot(wp[:, 0], wp[:, 1])

    if ld <= dists[0]:
        tx, ty = wp[0, 0], wp[0, 1]
    elif ld >= dists[-1]:
        tx, ty = wp[-1, 0], wp[-1, 1]
    else:
        idx = int(np.searchsorted(dists, ld))
        d0, d1 = dists[idx - 1], dists[idx]
        alpha = (ld - d0) / max(d1 - d0, 1e-6)
        tx = (1.0 - alpha) * wp[idx - 1, 0] + alpha * wp[idx, 0]
        ty = (1.0 - alpha) * wp[idx - 1, 1] + alpha * wp[idx, 1]

    l2 = max(tx * tx + ty * ty, 0.5)
    curvature = (2.0 * ty) / l2
    steer_rad = math.atan(wheelbase_m * curvature)
    return float(np.clip(steer_rad / max_steer_rad, -1.0, 1.0))


# ==========================================
# 2. PARTICIÓN POR SEGMENTO SIN DATA LEAKAGE
# ==========================================
def split_by_segment_stratified(df: pd.DataFrame, val_ratio: float = 0.15, seed: int = 42):
    """Divide en Train/Val por bloques de segmentos completos dentro de cada sesión (Town01..04).
    Evita que frames vecinos a 10 Hz caigan en particiones distintas.
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []

    for _session, g in df.groupby("session", sort=False):
        seg_counts = g.groupby("segment", sort=False).size()
        segments = seg_counts.index.to_numpy().copy()
        rng.shuffle(segments)

        target_val_rows = int(round(len(g) * val_ratio))
        val_segs = set()
        acc_val = 0
        for seg_id in segments:
            if acc_val < target_val_rows:
                val_segs.add(seg_id)
                acc_val += int(seg_counts.loc[seg_id])

        mask_val = g["segment"].isin(val_segs)
        train_parts.append(g[~mask_val])
        val_parts.append(g[mask_val])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


# ==========================================
# 3. DATASET ULTRA-RÁPIDO (UINT8 -> GPU)
# ==========================================
class WaypointLaneFollowDataset(Dataset):
    """Carga imágenes frontales o BEV en uint8 y devuelve:
      - imagen CHW uint8
      - waypoints [N_WP, 2] float32 en metros (x=adelante, y=derecha)
      - steering [1] float32
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str,
        wp_cols: list,
        view: str = "front",
        img_w: int = 320,
        img_h: int = 240,
        crop_top_ratio: float = 0.35,
        augment: bool = False,
        flip_prob: float = 0.5,
    ):
        df = df.reset_index(drop=True)
        path_col = "front_path" if view == "front" else "bev_path"
        self.img_paths = [os.path.join(base_dir, p) for p in df[path_col].tolist()]
        self.waypoints = df[wp_cols].to_numpy(dtype=np.float32).reshape(len(df), -1, 2)
        self.steerings = df["steering"].to_numpy(dtype=np.float32)

        self.view = view
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.crop_top_ratio = float(crop_top_ratio) if view == "front" else 0.0
        self.augment = bool(augment)
        self.flip_prob = float(flip_prob)

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

        resized = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        wp = self.waypoints[idx].copy()  # [N_WP, 2] -> (x_fwd, y_right)
        steer = float(self.steerings[idx])

        if self.augment and np.random.rand() < self.flip_prob:
            rgb = cv2.flip(rgb, 1)
            wp[:, 1] = -wp[:, 1]  # Invierte eje lateral (derecha <-> izquierda)
            steer = -steer

        chw_uint8 = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        return (
            torch.from_numpy(chw_uint8),
            torch.from_numpy(wp),
            torch.tensor([steer], dtype=torch.float32),
        )


# ==========================================
# 4. ARQUITECTURAS (RESNET-18 ESPACIAL & PILOTNET)
# ==========================================
class ResNet18Waypoints(nn.Module):
    """ResNet-18 con cuello espacial (Spatial Bottleneck) que preserva la geometría 2D
    antes de predecir los N_WP waypoints (x, y) y la salida auxiliar de dirección.
    """

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

        # Reducción de canales 1x1 + grilla espacial fija (4x6) para conservar coordenadas (u, v)
        self.spatial_neck = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),
        )
        flat_dim = 64 * 4 * 6  # 1536 features con estructura espacial

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
                bias_xy[:, 0] = torch.arange(1, self.n_waypoints + 1, dtype=torch.float32) * float(init_arc_step)
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


class PilotNetWaypoints(nn.Module):
    """PilotNet adaptada para predecir N_WP waypoints (x, y) + steering auxiliar."""

    def __init__(
        self,
        n_waypoints: int = 10,
        input_shape=(3, 240, 320),
        init_arc_step: float = 2.0,
    ):
        super().__init__()
        self.n_waypoints = int(n_waypoints)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = self.conv_layers(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
        )
        self.wp_head = nn.Linear(128, self.n_waypoints * 2)
        self.steer_head = nn.Linear(128, 1)
        if init_arc_step > 0.0:
            with torch.no_grad():
                nn.init.zeros_(self.wp_head.bias)
                bias_xy = self.wp_head.bias.view(self.n_waypoints, 2)
                bias_xy[:, 0] = torch.arange(1, self.n_waypoints + 1, dtype=torch.float32) * float(init_arc_step)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        feat = self.fc(x)
        wp = self.wp_head(feat).view(-1, self.n_waypoints, 2)
        steer = self.steer_head(feat)
        return wp, steer


# ==========================================
# 5. FUNCIÓN DE PÉRDIDA GEOMÉTRICA DE SEGUIMIENTO DE CARRIL
# ==========================================
class WaypointLaneFollowLoss(nn.Module):
    """Pérdida combinada para seguimiento de carril:
      - Huber Loss ponderada por horizonte (prioriza 2m..10m) y por eje (2.0x en eje lateral y).
      - Ponderación suave por curvatura/recuperación usando |wpa5_y| (a 10 m).
      - Regularización de suavidad de segunda derivada espacial entre waypoints consecutivos.
      - Pérdida auxiliar de dirección (steering).
    """

    def __init__(
        self,
        n_waypoints: int = 10,
        horizon_decay: float = 0.08,
        lat_weight: float = 2.0,
        curve_mult: float = 1.5,
        smooth_weight: float = 0.05,
        steer_weight: float = 0.15,
        huber_delta: float = 0.10,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        ks = torch.arange(n_waypoints, dtype=torch.float32, device=device)
        hw = torch.exp(-horizon_decay * ks)
        hw = hw / hw.mean()  # Media 1.0, forma [N_WP]
        self.register_buffer("horizon_weights", hw.view(1, n_waypoints, 1))

        axis_w = torch.tensor([1.0, float(lat_weight)], dtype=torch.float32, device=device)
        axis_w = axis_w / axis_w.mean()  # Media 1.0, forma [2]
        self.register_buffer("axis_weights", axis_w.view(1, 1, 2))

        self.curve_mult = float(curve_mult)
        self.smooth_weight = float(smooth_weight)
        self.steer_weight = float(steer_weight)
        self.huber_delta = float(huber_delta)

    def forward(self, pred_wp, true_wp, pred_steer, true_steer):
        # 1. Error punto a punto en waypoints [B, N_WP, 2]
        wp_err = F.huber_loss(pred_wp, true_wp, reduction="none", delta=self.huber_delta)
        wp_err = wp_err * self.horizon_weights * self.axis_weights  # [B, N_WP, 2]
        sample_wp_loss = wp_err.mean(dim=(1, 2))  # [B]

        # 2. Ponderación suave por curvatura a ~10m (índice 4 si N_WP >= 5)
        ref_idx = min(4, true_wp.size(1) - 1)
        lat_10m = torch.abs(true_wp[:, ref_idx, 1])
        sample_weights = 1.0 + self.curve_mult * torch.clamp(lat_10m / 2.0, max=1.5)
        loss_wp = torch.mean(sample_wp_loss * sample_weights)

        # 3. Suavidad geométrica (segunda diferencia a lo largo del horizonte)
        if pred_wp.size(1) >= 3 and self.smooth_weight > 0.0:
            d2_pred = pred_wp[:, 2:, :] - 2.0 * pred_wp[:, 1:-1, :] + pred_wp[:, :-2, :]
            d2_true = true_wp[:, 2:, :] - 2.0 * true_wp[:, 1:-1, :] + true_wp[:, :-2, :]
            loss_smooth = F.smooth_l1_loss(d2_pred, d2_true, beta=0.05)
        else:
            loss_smooth = pred_wp.new_zeros(())

        # 4. Auxiliar de steering
        if self.steer_weight > 0.0:
            steer_err = F.huber_loss(pred_steer, true_steer, reduction="none", delta=0.05).squeeze(-1)
            steer_w = 1.0 + self.curve_mult * torch.abs(true_steer.squeeze(-1))
            loss_steer = torch.mean(steer_err * steer_w)
        else:
            loss_steer = pred_wp.new_zeros(())

        return loss_wp + self.smooth_weight * loss_smooth + self.steer_weight * loss_steer


# ==========================================
# 6. BUCLE PRINCIPAL DE ENTRENAMIENTO GPU-MAX
# ==========================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="/home/daniel/code/2025-phd-daniel-guerrero/data/dataset_waypoints")
    p.add_argument("--csv", default="", help="Ruta opcional al CSV (por defecto <data-dir>/dataset.csv)")
    p.add_argument("--view", choices=["front", "bev"], default="front", help="Cámara de entrada")
    p.add_argument("--target", choices=["arc", "time"], default="arc", help="Waypoints por arco (wpa) o tiempo (wpt)")
    p.add_argument("--arch", choices=["resnet18", "pilotnet"], default="resnet18")
    p.add_argument("--img-w", type=int, default=320)
    p.add_argument("--img-h", type=int, default=240)
    p.add_argument("--crop-top", type=float, default=0.35, help="Recorte superior de cielo para vista frontal")
    p.add_argument("--min-speed", type=float, default=1.0, help="Velocidad mínima en km/h")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Fracción de segmentos para validación")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lat-weight", type=float, default=2.0, help="Peso relativo del eje lateral y frente a x")
    p.add_argument("--curve-mult", type=float, default=1.5, help="Multiplicador suave en curvas/recuperaciones DAgger")
    p.add_argument("--smooth-weight", type=float, default=0.05)
    p.add_argument("--steer-weight", type=float, default=0.15)
    p.add_argument("--flip-prob", type=float, default=0.5, help="Probabilidad de flip horizontal simétrico en Train")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", default="", help="Checkpoint previo para reanudar o hacer fine-tuning")
    p.add_argument("--resume", action="store_true", help="Reanudar automáticamente desde *_last.pth y *_meta.json")
    p.add_argument("--out-dir", default="", help="Carpeta de salida para los modelos (.pth)")
    p.add_argument("--max-batches", type=int, default=0, help="Limitar batches por época (0 = época completa, útil para pruebas)")
    p.add_argument("--poweroff", action="store_true", help="Apagar el minipc al finalizar el entrenamiento")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)

    csv_path = args.csv if args.csv else os.path.join(args.data_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_path}")

    print(f"[INFO] Leyendo CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)

    valid_col = "wp_arc_valid" if args.target == "arc" else "wp_time_valid"
    prefix = "wpa" if args.target == "arc" else "wpt"
    n_wp = 10 if args.target == "arc" else 8
    wp_cols = []
    for k in range(1, n_wp + 1):
        wp_cols += [f"{prefix}{k}_x", f"{prefix}{k}_y"]

    mask = (
        (df["has_image"] == 1)
        & (df[valid_col] == 1)
        & (df["speed_kmh"] >= args.min_speed)
    )
    df_clean = df[mask].copy()
    df_clean = df_clean.dropna(subset=wp_cols + ["steering", "front_path", "bev_path"]).reset_index(drop=True)

    train_df, val_df = split_by_segment_stratified(df_clean, val_ratio=args.val_ratio, seed=args.seed)

    if args.out_dir:
        save_dir = args.out_dir
    else:
        sub = "frontal" if args.view == "front" else "bev"
        save_dir = f"/home/daniel/code/2025-phd-daniel-guerrero/models/{sub}"
    os.makedirs(save_dir, exist_ok=True)

    model_tag = f"modelo_{args.arch}_waypoints_{args.target}_{args.view}"
    best_weights_path = os.path.join(save_dir, f"{model_tag}_best.pth")
    last_weights_path = os.path.join(save_dir, f"{model_tag}_last.pth")
    meta_out_path = os.path.join(save_dir, f"{model_tag}_meta.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    mean_gpu = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_gpu = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    print("=" * 78)
    print(f"[INFO] Dispositivo : {device} ({torch.cuda.get_device_name(0) if use_amp else 'CPU'})")
    print(f"[INFO] Vista / Arq : view={args.view} ({args.img_w}x{args.img_h}, crop_top={args.crop_top if args.view=='front' else 0.0}) | arch={args.arch} | target={args.target} ({n_wp} waypoints)")
    print(f"[INFO] Muestras    : Total válidas={len(df_clean):,} | Train={len(train_df):,} | Val={len(val_df):,} (split por segmento)")
    print(f"[INFO] Hiperparám. : Batch={args.batch_size} | Epochs={args.epochs} | LR={args.lr} | LatWeight={args.lat_weight} | CurveMult={args.curve_mult}")
    print(f"[INFO] Guardado en : {best_weights_path}")
    print("=" * 78)

    train_dataset = WaypointLaneFollowDataset(
        train_df,
        args.data_dir,
        wp_cols=wp_cols,
        view=args.view,
        img_w=args.img_w,
        img_h=args.img_h,
        crop_top_ratio=args.crop_top,
        augment=True,
        flip_prob=args.flip_prob,
    )
    val_dataset = WaypointLaneFollowDataset(
        val_df,
        args.data_dir,
        wp_cols=wp_cols,
        view=args.view,
        img_w=args.img_w,
        img_h=args.img_h,
        crop_top_ratio=args.crop_top,
        augment=False,
        flip_prob=0.0,
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

    resume_ckpt = last_weights_path if (args.resume and os.path.exists(last_weights_path)) else ""
    ckpt_to_load = resume_ckpt or args.pretrained

    init_step = 2.0 if args.target == "arc" else 0.0
    if args.arch == "resnet18":
        model = ResNet18Waypoints(
            n_waypoints=n_wp,
            pretrained_imagenet=not bool(ckpt_to_load),
            init_arc_step=init_step,
        )
    else:
        model = PilotNetWaypoints(
            n_waypoints=n_wp,
            input_shape=(3, args.img_h, args.img_w),
            init_arc_step=init_step,
        )

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"[INFO] Cargando pesos previos desde: {ckpt_to_load}", flush=True)
        state = torch.load(ckpt_to_load, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)

    model = model.to(device, memory_format=torch.channels_last)

    criterion = WaypointLaneFollowLoss(
        n_waypoints=n_wp,
        lat_weight=args.lat_weight,
        curve_mult=args.curve_mult,
        smooth_weight=args.smooth_weight,
        steer_weight=args.steer_weight,
        device=device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    history = []
    start_epoch = 1

    if args.resume and os.path.exists(meta_out_path) and os.path.exists(last_weights_path):
        with open(meta_out_path, "r") as mf:
            prev_meta = json.load(mf)
        history = list(prev_meta.get("history", []))
        best_val_loss = float(prev_meta.get("best_val_loss", float("inf")))
        start_epoch = len(history) + 1
        for _ in range(len(history)):
            optimizer._step_count = 1
            scheduler.step()
        print(
            f"[INFO] Reanudando entrenamiento en época {start_epoch:02d}/{args.epochs:02d} "
            f"(Mejor Val Loss previo: {best_val_loss:.5f}, LR actual: {scheduler.get_last_lr()[0]:.2e})",
            flush=True,
        )

    idx_6m = min(2, n_wp - 1)   # wpa3 = 6m
    idx_10m = min(4, n_wp - 1)  # wpa5 = 10m

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_gpu = torch.zeros(1, device=device)
        train_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"Época {epoch:02d}/{args.epochs:02d} [Train {args.arch}]",
            disable=not sys.stderr.isatty(),
        )
        for step, (images_u8, true_wp, true_steer) in enumerate(pbar, 1):
            images = images_u8.to(device, dtype=torch.float32, memory_format=torch.channels_last, non_blocking=True)
            images.mul_(1.0 / 255.0).sub_(mean_gpu).div_(std_gpu)
            true_wp = true_wp.to(device, non_blocking=True)
            true_steer = true_steer.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_wp, pred_steer = model(images)
                loss = criterion(pred_wp, true_wp, pred_steer, true_steer)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_gpu += loss.detach()
            train_steps += 1
            if step % 20 == 0 or step == len(train_loader):
                pbar.set_postfix({"loss": f"{(train_loss_gpu.item() / train_steps):.5f}"})

            if args.max_batches > 0 and step >= args.max_batches:
                break

        # Validación con métricas geométricas en metros
        model.eval()
        val_loss_gpu = torch.zeros(1, device=device)
        ade_sum = torch.zeros(1, device=device)
        fde_sum = torch.zeros(1, device=device)
        lat6_sum = torch.zeros(1, device=device)
        lat10_sum = torch.zeros(1, device=device)
        steer_mae_sum = torch.zeros(1, device=device)
        val_samples = 0
        val_steps = 0

        with torch.no_grad():
            for step, (images_u8, true_wp, true_steer) in enumerate(val_loader, 1):
                images = images_u8.to(device, dtype=torch.float32, memory_format=torch.channels_last, non_blocking=True)
                images.mul_(1.0 / 255.0).sub_(mean_gpu).div_(std_gpu)
                true_wp = true_wp.to(device, non_blocking=True)
                true_steer = true_steer.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_wp, pred_steer = model(images)
                    loss = criterion(pred_wp, true_wp, pred_steer, true_steer)

                bsz = images.size(0)
                val_loss_gpu += loss.detach()
                val_steps += 1
                val_samples += bsz

                dist_err = torch.linalg.norm(pred_wp.float() - true_wp.float(), dim=-1)  # [B, N_WP]
                ade_sum += dist_err.mean(dim=1).sum()
                fde_sum += dist_err[:, -1].sum()
                lat6_sum += torch.abs(pred_wp[:, idx_6m, 1].float() - true_wp[:, idx_6m, 1].float()).sum()
                lat10_sum += torch.abs(pred_wp[:, idx_10m, 1].float() - true_wp[:, idx_10m, 1].float()).sum()
                steer_mae_sum += torch.abs(pred_steer.float() - true_steer.float()).sum()

                if args.max_batches > 0 and step >= args.max_batches:
                    break

        avg_train = train_loss_gpu.item() / max(train_steps, 1)
        avg_val = val_loss_gpu.item() / max(val_steps, 1)
        ade_m = ade_sum.item() / max(val_samples, 1)
        fde_m = fde_sum.item() / max(val_samples, 1)
        lat6_m = lat6_sum.item() / max(val_samples, 1)
        lat10_m = lat10_sum.item() / max(val_samples, 1)
        steer_mae = steer_mae_sum.item() / max(val_samples, 1)

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"Época {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f} | "
            f"ADE: {ade_m:.3f}m | Lat@6m: {lat6_m:.3f}m | Lat@10m: {lat10_m:.3f}m | FDE@20m: {fde_m:.3f}m | "
            f"SteerMAE: {steer_mae:.4f} | LR: {cur_lr:.2e}",
            flush=True,
        )

        torch.save(model.state_dict(), last_weights_path)
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_weights_path)
            print(
                f"  --> [BEST] Nuevo mejor modelo guardado en {best_weights_path} "
                f"(Val Loss: {best_val_loss:.5f}, Lat@6m: {lat6_m:.3f}m, ADE: {ade_m:.3f}m)",
                flush=True,
            )

        history.append(
            dict(
                epoch=epoch,
                train_loss=round(avg_train, 6),
                val_loss=round(avg_val, 6),
                ade_m=round(ade_m, 4),
                lat_err_6m=round(lat6_m, 4),
                lat_err_10m=round(lat10_m, 4),
                fde_20m=round(fde_m, 4),
                steer_mae=round(steer_mae, 5),
                lr=cur_lr,
                best=is_best,
            )
        )
        with open(meta_out_path, "w") as mf:
            json.dump(
                dict(
                    config=vars(args),
                    wp_columns=wp_cols,
                    train_samples=len(train_df),
                    val_samples=len(val_df),
                    best_val_loss=best_val_loss,
                    history=history,
                ),
                mf,
                indent=2,
            )

    print(f"\n[ÉXITO] Entrenamiento finalizado. Mejor Val Loss: {best_val_loss:.5f} -> {best_weights_path}", flush=True)

    if args.poweroff:
        print("[INFO] Apagando el equipo...", flush=True)
        os.system("sudo -n poweroff 2>/dev/null || systemctl poweroff -i 2>/dev/null || poweroff 2>/dev/null")


if __name__ == "__main__":
    main()
