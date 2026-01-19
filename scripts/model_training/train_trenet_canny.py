#!/usr/bin/env python3
# train_trenet_canny.py
import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# ----------------------------
# Dataset secuencial con Canny
# ----------------------------
class CarlaSequenceDataset(Dataset):
    """
    CSV esperado con columnas: frame, steer, throttle, brake
    - frame: ruta relativa o nombre de archivo dentro de img_root_dir
    - Las secuencias son ventanas deslizantes de longitud seq_len
      y la etiqueta es la del último frame de la ventana.
    """
    def __init__(
        self,
        csv_path: str,
        img_root_dir: str,
        seq_len: int = 5,
        canny_low: int = 100,
        canny_high: int = 200,
        canny_blur: int = 5,
        canny_sigma: float = 1.0,
        use_canny: bool = True,
        img_size: Tuple[int, int] = (200, 66),  # (W, H)
    ):
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip().str.lower()

        required = {'frame', 'steer', 'throttle', 'brake'}
        missing = required - set(self.data.columns)
        if missing:
            raise KeyError(f"Faltan columnas en el CSV: {missing}. "
                           f"Encontradas: {list(self.data.columns)}")

        # Asegurar orden temporal por 'frame' (si es numerable, lo ordena bien;
        # si son strings, al menos queda determinista)
        self.data = self.data.sort_values('frame').reset_index(drop=True)

        self.img_root_dir = img_root_dir
        self.paths = [os.path.join(self.img_root_dir, str(p)) for p in self.data['frame']]
        self.labels = self.data[['steer', 'throttle', 'brake']].astype('float32').to_numpy()

        self.seq_len = int(seq_len)
        if self.seq_len < 2:
            raise ValueError("seq_len debe ser >= 2 para aprovechar temporalidad.")

        self.use_canny = use_canny
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.canny_blur = int(canny_blur)
        self.canny_sigma = float(canny_sigma)
        self.img_w, self.img_h = img_size

    def __len__(self):
        return max(0, len(self.paths) - self.seq_len + 1)

    def _read_image_proc(self, path: str) -> np.ndarray:
        """Lee imagen, aplica resize (200x66) y Canny -> [1, H, W] float32 en [0,1]"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

        if self.use_canny:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.canny_blur and self.canny_blur > 0:
                k = self.canny_blur if self.canny_blur % 2 == 1 else self.canny_blur + 1
                gray = cv2.GaussianBlur(gray, (k, k), self.canny_sigma)
            edges = cv2.Canny(gray, self.canny_low, self.canny_high).astype(np.float32) / 255.0
            # Añadir canal: [1,H,W]
            edges = edges[None, :, :]
            return edges
        else:
            # Alternativa: RGB normalizado [0,1] y convertir a 1 canal (luma)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Luma aproximada
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)
            gray = gray[None, :, :]
            return gray

    def __getitem__(self, idx):
        # Secuencia [idx ... idx+seq_len-1]
        paths_seq = self.paths[idx: idx + self.seq_len]
        imgs = [self._read_image_proc(p) for p in paths_seq]  # lista de [1,H,W]
        # Tensor (T, C, H, W) -> (seq_len, 1, 66, 200)
        imgs = np.stack(imgs, axis=0).astype(np.float32)

        # Etiqueta: la del último frame de la ventana
        label = self.labels[idx + self.seq_len - 1]  # (3,)
        return torch.from_numpy(imgs), torch.from_numpy(label)


# ----------------------
# Modelo TreNet CNN+LSTM
# ----------------------
class CarlaTreNet(nn.Module):
    """
    Entrada: x de forma (B, T, C, H, W) = (batch, seq_len, 1, 66, 200)
    CNN procesa cada frame -> vector de features
    LSTM agrega temporalmente -> predice [steer, throttle, brake]
    """
    def __init__(self, in_channels: int = 1, hidden_size: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 66, 200)
            cnn_out = self.cnn(dummy).shape[1]

        self.lstm = nn.LSTM(input_size=cnn_out, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 3)  # steer, throttle, brake
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)              # (B*T, F)
        feats = feats.view(B, T, -1)     # (B, T, F)
        _, (h_n, _) = self.lstm(feats)   # h_n: (num_layers, B, hidden)
        h_last = h_n[-1]                 # (B, hidden)
        out = self.fc(h_last)            # (B, 3)
        return out


# ---------------
# Entrenamiento
# ---------------
def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=None):
    model.train()
    running = 0.0
    for seqs, labels in loader:
        seqs = seqs.to(device)         # (B,T,1,66,200)
        labels = labels.to(device)     # (B,3)

        preds = model(seqs)
        loss = criterion(preds, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running += loss.item()
    return running


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for seqs, labels in loader:
        seqs = seqs.to(device)
        labels = labels.to(device)
        preds = model(seqs)
        loss = criterion(preds, labels)
        total += loss.item()
    return total


def main():
    parser = argparse.ArgumentParser(description="Entrena TreNet (CNN+LSTM) con secuencias e imágenes Canny")
    parser.add_argument("--csv", type=str, default="dataset/controls.csv")
    parser.add_argument("--images", type=str, default="dataset/images")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fracción para validación (0 desactiva)")

    # Canny / prepro
    parser.add_argument("--use_canny", action="store_true", help="Aplica Canny (recomendado)")
    parser.add_argument("--canny_low", type=int, default=100)
    parser.add_argument("--canny_high", type=int, default=200)
    parser.add_argument("--canny_blur", type=int, default=5)
    parser.add_argument("--canny_sigma", type=float, default=1.0)

    # Varios
    parser.add_argument("--save_path", type=str, default="carla_trenet_canny.pth")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("Args:", args)

    # Dataset completo
    full_ds = CarlaSequenceDataset(
        csv_path=args.csv,
        img_root_dir=args.images,
        seq_len=args.seq_len,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        canny_blur=args.canny_blur,
        canny_sigma=args.canny_sigma,
        use_canny=args.use_canny,
        img_size=(200, 66),
    )

    # Split train/val si se pide
    if args.val_split and 0.0 < args.val_split < 0.5:
        n = len(full_ds)
        n_val = int(n * args.val_split)
        n_train = n - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val],
                                                         generator=torch.Generator().manual_seed(42))
    else:
        train_ds, val_ds = full_ds, None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(args.device.startswith("cuda"))
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=max(1, args.workers // 2), pin_memory=(args.device.startswith("cuda"))
        )

    # Modelo, optim, loss
    device = torch.device(args.device)
    model = CarlaTreNet(in_channels=1, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Dispositivo: {device}")
    print(f"Tamaño train: {len(train_ds)}  |  Tamaño val: {len(val_ds) if val_ds is not None else 0}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.max_grad_norm)
        msg = f"Epoch {epoch:02d} | train_loss={train_loss:.4f}"

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            msg += f" | val_loss={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model, args.save_path)
                msg += f"  -> ✅ guardado: {args.save_path}"
        else:
            # Si no hay val_split, guarda cada epoch
            torch.save(model, args.save_path)
            msg += f"  -> 💾 guardado: {args.save_path}"

        print(msg)

    print("✅ Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
