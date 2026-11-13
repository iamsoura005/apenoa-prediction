import torch
import torch.nn as nn
from typing import Tuple

class Apnea1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for Sleep Apnea Event Prediction from PPG signals.
    
    Architecture:
    - Input: (Batch, 1, 2400) [30-second window at 80 Hz]
    - Block 1: Conv1D (32 filters, kernel=7, padding=3) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.3)
    - Block 2: Conv1D (64 filters, kernel=5, padding=2) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.3)
    - Block 3: Conv1D (128 filters, kernel=3, padding=1) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.3)
    - Flatten: 128 * 300 = 38400
    - Dense 1: Linear(38400, 128) -> ReLU -> Dropout(0.5)
    - Dense 2: Linear(128, 1) -> Output Logit
    """
    def __init__(self, input_samples: int = 2400, in_channels: int = 1):
        super().__init__()
        self.input_samples = input_samples
        self.in_channels = in_channels
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.3)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.3)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=0.3)
        
        # Fully Connected Block
        flatten_dim = 128 * (input_samples // 8)
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (Batch, 1, 2400) or (Batch, 2400)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.drop1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = x.flatten(start_dim=1)
        x = self.drop4(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

