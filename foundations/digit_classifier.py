import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture:
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        torch.manual_seed(0)
        x = self.model(images)
        return torch.round(x * 10000) / 10000
        pass
