import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import os
import cv2
import numpy as np
import torchvision.models as models


class CarlaDataset(Dataset):
    def __init__(self, csv_path, img_root_dir):
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip().str.lower()

        required = {'frame', 'steer', 'throttle', 'brake'}
        missing = required - set(self.data.columns)
        if missing:
            raise KeyError(
                f"Faltan columnas en el CSV: {missing}. "
                f"Encontradas: {list(self.data.columns)}"
            )

        self.img_root_dir = img_root_dir
        self.paths = [
            os.path.join(self.img_root_dir, p)
            for p in self.data['frame'].astype(str)
        ]
        self.labels = (
            self.data[['steer', 'throttle', 'brake']]
            .astype('float32')
            .to_numpy()
        )

        # Para normalizar como ImageNet (MobileNet preentrenada)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = cv2.imread(img_path)  # BGR
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

        # Redimensionamos (MobileNet puede trabajar con otros tamaños,
        # pero mantenemos tu 200x66)
        image = cv2.resize(image, (200, 66))

        # Convertir a float y escalar [0,1]
        image = image.astype(np.float32) / 255.0

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        # Normalización tipo ImageNet
        image = (image - self.mean) / self.std

        image_tensor = torch.from_numpy(image)      # (3, 66, 200)
        label = torch.from_numpy(self.labels[idx])  # (3,)

        return image_tensor, label


class CarlaMobileNet(nn.Module):
    def __init__(self, freeze_features=True):
        super().__init__()
        # Cargamos MobileNetV2 preentrenada
        self.backbone = models.mobilenet_v2(pretrained=True)

        # Opcional: congelar capas convolucionales
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Reemplazamos la cabeza (classifier) para regresión de 3 valores
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_feats, 3)

    def forward(self, x):
        return self.backbone(x)


# ------------------ Entrenamiento ------------------
if __name__ == "__main__":
    csv_path = "dataset/controls_balanced.csv"
    img_root_dir = "dataset/images"

    dataset = CarlaDataset(csv_path, img_root_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    model = CarlaMobileNet(freeze_features=True)

    # Solo optimizamos los parámetros que tienen gradiente activado
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_optimize, lr=1e-4)

    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        print(f"📦 Epoch {epoch+1} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "carla_model_mobilenet.pth")
    print("✅ Modelo guardado como carla_model_mobilenet.pth")
