import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("LocalizationSegmenter")
class LocalizationSegmenter(nn.Module):
    """
    A simple FCN/U-Net like model for supervised anomaly localization 
    based on aggregated diffusion residuals.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(LocalizationSegmenter, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        
        # Output
        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        
        x_bottle = F.relu(self.bottleneck(self.pool(x2)))
        
        x_up1 = self.upconv1(x_bottle)
        x_up1 = F.relu(self.conv3(x_up1))
        
        x_up2 = self.upconv2(x_up1)
        
        logits = self.conv_out(x_up2) 
        return logits