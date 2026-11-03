import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm


# ==========================================
# 1. DATA LOADER (IMAGE AND STEERING ONLY - BEV)
# ==========================================
class DrivingDataset(Dataset):
    def __init__(self, dataframe, frames_dir, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.frames_dir = frames_dir
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 320)),
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

        img_filename = os.path.basename(row["image_path"])
        img_path = os.path.join(self.frames_dir, img_filename)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        steer_val = float(row["steering"])

        # Aumento simétrico: 50% probabilidad de invertir horizontalmente imagen y ángulo
        if self.augment and torch.rand(1).item() < 0.5:
            image = cv2.flip(image, 1)
            steer_val = -steer_val

        image_tensor = self.transform(image)
        steering_out = torch.tensor([steer_val], dtype=torch.float32)
        return image_tensor, steering_out


# ==========================================
# 2. ARCHITECTURE (PILOTNET -> STEERING ONLY)
# ==========================================
class PilotNetSteeringOnly(nn.Module):
    def __init__(self, input_shape=(3, 240, 320)):
        super(PilotNetSteeringOnly, self).__init__()

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
        x = torch.zeros(1, *shape)
        x = self.conv_layers(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, image):
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        return self.output_layer(x)


# ==========================================
# 3. WEIGHTED LOSS (HOT BALANCING)
# ==========================================
def weighted_steering_loss(pred_steering, true_steering, base_weight=1.0, curve_multiplier=10.0):
    raw_loss = F.huber_loss(pred_steering, true_steering, reduction="none")
    weights = base_weight + (torch.abs(true_steering) * curve_multiplier)
    return torch.mean(raw_loss * weights)


# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    CSV_FILE = "/home/daniel/code/2025-phd-daniel-guerrero/data/dataset_masivo_continuo/driving_log_bev_best.csv"
    FRAMES_DIR = "/home/daniel/code/2025-phd-daniel-guerrero/data/dataset_masivo_continuo/frames_bev_best"
    MODEL_SAVE_PATH = "/home/daniel/code/2025-phd-daniel-guerrero/models/bev/modelo_pilotnet_bev_steering_only.pth"

    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Training PilotNet (Image Only) on BEV images: {FRAMES_DIR}")

    df = pd.read_csv(CSV_FILE)
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_size = int(0.85 * len(df_shuffled))
    df_train = df_shuffled.iloc[:train_size]
    df_val = df_shuffled.iloc[train_size:]

    train_dataset = DrivingDataset(df_train, frames_dir=FRAMES_DIR, augment=False)
    val_dataset = DrivingDataset(df_val, frames_dir=FRAMES_DIR, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = PilotNetSteeringOnly().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    best_val_loss = float("inf")

    print(f"[INFO] Starting training for {EPOCHS} epochs on {len(train_dataset)} train / {len(val_dataset)} val frames...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train BEV]")
        for images, steerings in progress_bar:
            images = images.to(device, non_blocking=True)
            steerings = steerings.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_steering = model(images)
            loss = weighted_steering_loss(pred_steering, steerings, curve_multiplier=12.0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, steerings in val_loader:
                images = images.to(device, non_blocking=True)
                steerings = steerings.to(device, non_blocking=True)

                pred_steering = model(images)
                loss = weighted_steering_loss(pred_steering, steerings, curve_multiplier=12.0)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch + 1} Result | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> [BEST] Guardado mejor modelo (Val Loss: {best_val_loss:.4f}) en {MODEL_SAVE_PATH}")

    print(f"\n[SUCCESS] Entrenamiento finalizado. Mejor Val Loss: {best_val_loss:.4f} en: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
