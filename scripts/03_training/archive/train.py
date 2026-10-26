import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import CarlaPilotNet
from dataset import CarlaDataset

dataset = CarlaDataset("../dataset/controls.csv", "../dataset/images")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = CarlaPilotNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for imgs, labels in dataloader:
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

torch.save(model, "carla_model.pth")