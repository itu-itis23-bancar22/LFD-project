# This script benchmarks traditional anomaly detection models (One-Class SVM, Isolation Forest)
# on top of deep features extracted by a pretrained ResNet18 encoder.
# It uses the same encoder trained with Deep SVDD to ensure fair comparison.

from models.resnet_encoder import ResNet18Encoder                  # Feature extractor (ResNet18)
from data.dataset import CIFAR10OneClass                          # CIFAR-10 dataset wrapper for one-class setup
from traditional_models.baseline_eval import extract_features, evaluate_baselines  # Traditional methods

from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# --- Configuration: device, class, and batch size ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
normal_class = 'airplane'
batch_size = 128

# --- Define data preprocessing pipeline ---
transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Correct normalization for RGB inputs
])

# --- Load one-class CIFAR-10 training and test sets ---
train_dataset = CIFAR10OneClass(root='data', normal_class=normal_class, train=True, transform=transform)
test_dataset = CIFAR10OneClass(root='data', normal_class=normal_class, train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Load pretrained ResNet18 encoder used in Deep SVDD ---
encoder = ResNet18Encoder(pretrained=False).to(device)
checkpoint = torch.load('checkpoints/deep_svdd.pth', map_location=device)
encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Load only matching layers

# --- Extract latent feature representations for both train and test data ---
train_feats, _ = extract_features(encoder, train_loader, device)
test_feats, test_labels = extract_features(encoder, test_loader, device)

# --- Optionally visualize feature space with t-SNE or PCA ---
from evaluation.feature_viz import plot_latent_space
# plot_latent_space(test_feats, test_labels, method='tsne', title='ResNet18 Latent Space')

# --- Evaluate traditional anomaly detection models on extracted features ---
results = evaluate_baselines(train_feats, test_feats, test_labels)

# --- Print performance metrics for each method ---
print("\nBaseline Results:")
for name, (auc, precision, recall, f1) in results.items():
    print(f"{name} → AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")