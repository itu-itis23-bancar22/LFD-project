# This script evaluates a trained Deep SVDD model on the test set of CIFAR-10.
# It loads the saved model checkpoint, runs predictions, computes evaluation metrics,
# and visualizes both performance curves and latent feature distributions.

from models.resnet_encoder import ResNet18Encoder           # ResNet18 encoder for feature extraction
from models.deep_svdd import DeepSVDD                       # Deep SVDD model definition
from data.dataset import CIFAR10OneClass                    # Custom CIFAR-10 one-class dataset
from evaluation.evaluate import evaluate_svdd               # Evaluation function for SVDD

from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# --- Setup device, class label, and batch size ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
normal_class = 'airplane'
batch_size = 128

# --- Define preprocessing pipeline ---
transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

# --- Load test dataset containing both normal and anomaly samples ---
test_dataset = CIFAR10OneClass(root='data', normal_class=normal_class, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Reconstruct the model and load the saved weights ---
encoder = ResNet18Encoder(pretrained=False)
model = DeepSVDD(encoder, device)

checkpoint = torch.load('checkpoints/deep_svdd.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # Load model parameters
model.c = checkpoint['center_c']                       # Load hypersphere center

# --- Evaluate model performance on test set ---
from evaluation.evaluate import plot_curves
from evaluation.feature_viz import plot_latent_space

auc, precision, recall, f1, labels, scores = evaluate_svdd(model, test_loader, device)

# --- Plot ROC and Precision-Recall curves ---
plot_curves(labels, scores, title='Deep SVDD')

# --- Extract feature embeddings from encoder for visualization ---
from traditional_models.baseline_eval import extract_features
test_feats, test_labels = extract_features(model.encoder, test_loader, device)

# --- Visualize feature space using dimensionality reduction (e.g., t-SNE) ---
plot_latent_space(test_feats, test_labels, method='tsne', title='Deep SVDD Latent Space')