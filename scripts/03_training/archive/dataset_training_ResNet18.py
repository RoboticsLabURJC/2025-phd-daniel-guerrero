import os, csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

class CarlaSteerDataset(Dataset):
    def __init__(self, root_dir, csv_name="labels.csv"):
        self.root_dir = root_dir
        self.rows = []
        with open(os.path.join(root_dir, csv_name), "r") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append(row)

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.root_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)

        steer = float(row["applied_steer"])
        y = torch.tensor([steer], dtype=torch.float32)
        return x, y

class ResNet18Steer(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.backbone = m

    def forward(self, x):
        y = self.backbone(x)
        return torch.tanh(y)  # [-1, 1]

def weighted_huber(pred, target):
    # pred/target: (B,1)
    huber = nn.SmoothL1Loss(reduction="none")(pred, target)
    w = 1.0 + 2.0 * target.abs()
    return (huber * w).mean()

def main():
    root = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/datasets_collection/dataset_runs/20260215_201946"  # <- dataset
    ds = CarlaSteerDataset(root)

    n = len(ds)
    n_train = int(0.85 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ROCm aparece como "cuda"
    print("device:", device)

    model = ResNet18Steer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best = 1e9
    for epoch in range(1, 21):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = weighted_huber(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                va_loss += weighted_huber(pred, y).item()

        tr_loss /= max(1, len(train_loader))
        va_loss /= max(1, len(val_loader))
        print(f"epoch {epoch:02d} | train {tr_loss:.5f} | val {va_loss:.5f}")

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict()}, os.path.join(root, "resnet18_steer_best.pt"))
            print("  saved best")

if __name__ == "__main__":
    main()
