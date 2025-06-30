import torch
import torch.nn as nn

class CarlaPilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 66, 200)  # tu tamaño de entrada
            output = self.cnn(dummy)
            flattened_size = output.shape[1]  # resultado correcto: 3840

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 3)  # steer, throttle, brake
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)
