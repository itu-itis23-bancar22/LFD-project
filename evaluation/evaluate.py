# This script provides evaluation utilities for Deep SVDD models, including scoring and plotting performance curves.

import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

# Evaluate Deep SVDD performance on test data
# Computes distance-based anomaly scores and calculates standard metrics

def evaluate_svdd(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    scores = []   # Store computed anomaly scores
    labels = []   # Store true class labels

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)  # Forward pass
            dists = torch.sum((outputs - model.c) ** 2, dim=1)  # Compute squared distances to center
            scores.extend(dists.cpu().numpy())  # Save distances as anomaly scores
            labels.extend(targets.cpu().numpy())

    # Compute evaluation metrics
    auc = roc_auc_score(labels, scores)  # Area under ROC curve
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, [s > 0.5 for s in scores], average='binary')  # Binarize scores for metrics

    # Print metrics to console
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return auc, precision, recall, f1, labels, scores

# --- Visualization ---

from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# Plot ROC and Precision-Recall curves using score distributions
def plot_curves(labels, scores, title='Deep SVDD'):
    fpr, tpr, _ = roc_curve(labels, scores)  # Compute ROC curve
    precision, recall, _ = precision_recall_curve(labels, scores)  # Compute PR curve

    plt.figure(figsize=(12, 5))

    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'{title} ROC Curve')
    plt.grid()
    plt.legend()

    # Plot PR Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label='PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title} PR Curve')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()