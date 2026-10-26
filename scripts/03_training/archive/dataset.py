import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np

class CarlaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["frame"])
        image = cv2.imread(img_path)
        image = cv2.resize(image, (200, 66))  # tamaño típico para PilotNet
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # canal primero
        control = np.array([row["steer"], row["throttle"], row["brake"]], dtype=np.float32)
        return torch.tensor(image), torch.tensor(control)
