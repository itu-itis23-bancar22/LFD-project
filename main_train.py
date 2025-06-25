# This script trains a Deep SVDD model on a single normal class from the CIFAR-10 dataset.
# It covers data loading, model initialization, center calculation, and training execution.

from training.train import train_svdd                       # Training function for Deep SVDD
from models.resnet_encoder import ResNet18Encoder           # ResNet18-based feature extractor
from models.deep_svdd import DeepSVDD                       # Deep SVDD model definition
from models.init_center import init_center_c                # Center initialization function
from data.dataset import CIFAR10OneClass                    # One-class CIFAR-10 dataset wrapper

from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# Set device for training: use GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the normal class for one-class training scenario
normal_class = 'airplane'

# Define input preprocessing steps: convert to float and normalize RGB channels
transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset consisting only of samples from the normal class
train_dataset = CIFAR10OneClass(root='data', normal_class=normal_class, train=True, transform=transform)

# Create a DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize a ResNet18 encoder with pretrained weights
encoder = ResNet18Encoder(pretrained=True)

# Wrap the encoder into a Deep SVDD model
svdd_model = DeepSVDD(encoder, device)

# Compute the initial center of the hypersphere before training
c = init_center_c(svdd_model, train_loader, device)

# Train the SVDD model using the specified number of epochs
train_svdd(svdd_model, train_loader, device, center_c=c, num_epochs=10)