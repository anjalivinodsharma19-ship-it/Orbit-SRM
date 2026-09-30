import torch
from torch import nn

class SRCNN(nn.Module):
    """Small residual CNN; checkpoint weights are required for trained inference."""
    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(inplace=True), nn.Conv2d(32, channels, 5, padding=2))
    def forward(self, x):
        return (x + self.net(x)).clamp(0, 1)
