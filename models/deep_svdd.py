# This module defines the Deep SVDD model used for one-class anomaly detection.
# It includes a forward method to encode inputs and a custom loss function based on distance to a center point.

import torch
import torch.nn as nn

class DeepSVDD(nn.Module):
    def __init__(self, encoder, device, objective='one-class'):
        super(DeepSVDD, self).__init__()
        self.encoder = encoder              # Feature extractor (e.g., ResNet18)
        self.device = device                # Computation device (CPU or CUDA)
        self.objective = objective          # Training objective: 'one-class' or 'soft-boundary'
        self.c = None                       # Hypersphere center vector
        self.R = torch.tensor(0.0, device=device, requires_grad=True)  # Radius for soft-boundary SVDD
        self.embedding_dim = 512            # Dimension of latent space (hardcoded)

    def forward(self, x):
        # Forward pass through encoder network
        return self.encoder(x)

    def compute_loss(self, outputs):
        # Compute squared distance from center 'c'
        dist = torch.sum((outputs - self.c) ** 2, dim=1)

        if self.objective == 'soft-boundary':
            # Soft-boundary SVDD loss calculation
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            # Standard one-class SVDD loss (mean distance to center)
            loss = torch.mean(dist)

        return loss, dist