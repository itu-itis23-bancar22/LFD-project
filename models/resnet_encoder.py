# This module defines a ResNet18-based encoder for feature extraction.
# The final fully connected layer is removed and replaced by global average pooling.

import torch.nn as nn
import torchvision.models as models

class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Encoder, self).__init__()

        # Load pretrained ResNet18 model from torchvision
        resnet18 = models.resnet18(pretrained=pretrained)

        # Remove the last two layers (avgpool and fc), keep convolutional backbone
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-2])

        # Add adaptive average pooling to reduce feature maps to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract convolutional features
        x = self.pool(x)               # Apply global average pooling
        return x.view(x.size(0), -1)   # Flatten the output to (batch_size, embedding_dim)