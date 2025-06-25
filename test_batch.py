# This script validates core components of the Deep SVDD pipeline on a single batch from CIFAR-10.
# It ensures that data loading, feature extraction, center initialization, and SVDD loss computation function correctly.

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.dataset import CIFAR10OneClass               # Custom CIFAR-10 dataset wrapper for one-class setting
from models.resnet_encoder import ResNet18Encoder      # ResNet18 feature extractor
from models.deep_svdd import DeepSVDD                  # Deep SVDD model definition
from models.init_center import init_center_c           # Function to compute center of hypersphere

# Select computation device: use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define the normal class for one-class training
normal_class = 'airplane'
batch_size = 128

# Define input transformations: convert to float and normalize per channel
transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

# Initialize training dataset and dataloader
train_dataset = CIFAR10OneClass(root='data', normal_class=normal_class, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the ResNet18 encoder model with pretrained weights
encoder = ResNet18Encoder(pretrained=True)
encoder = encoder.to(device)

# Wrap the encoder inside the Deep SVDD model
svdd_model = DeepSVDD(encoder, device)
svdd_model = svdd_model.to(device)

# Compute the initial center of the hypersphere based on training data
print("[*] Initializing hypersphere center c...")
c = init_center_c(svdd_model, train_loader, device)
svdd_model.c = c  # Assign center to model

# Switch model to evaluation mode
svdd_model.eval()

# Retrieve a single batch from the dataloader for testing
for data in train_loader:
    inputs, _ = data
    break  # Only the first batch is needed

inputs = inputs.to(device)

# Forward pass through the model with no gradient computation
with torch.no_grad():
    outputs = svdd_model(inputs)                   # Get latent representations
    loss, dists = svdd_model.compute_loss(outputs) # Compute SVDD loss and distances

# Output shape and evaluation metrics
print(f"[*] Outputs shape: {outputs.shape}")                      # Shape should be [batch_size, feature_dim]
print(f"[*] Loss: {loss.item():.4f}")                            # Mean squared error loss
print(f"[*] Mean distance to center: {dists.mean().item():.4f}") # Average distance from center