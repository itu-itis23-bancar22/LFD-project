# This script defines baseline evaluation functions for anomaly detection.
# It includes feature extraction from a trained model and evaluation using One-Class SVM and Isolation Forest.

import torch
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

def extract_features(model, loader, device):
    # Set model to evaluation mode to disable dropout and batchnorm updates
    model.eval()
    features, labels = [], []

    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting Features"):
            inputs = inputs.to(device)
            outputs = model(inputs)  # Extract latent representations
            features.append(outputs.cpu().numpy())  # Move to CPU and convert to numpy
            labels.append(targets.cpu().numpy())

    # Concatenate all batches into a single feature and label array
    return np.concatenate(features), np.concatenate(labels)

def evaluate_baselines(train_feats, test_feats, test_labels):
    results = {}

    # --- One-Class SVM ---
    print("\nTraining One-Class SVM...")
    svm = OneClassSVM(gamma='auto').fit(train_feats)  # Train SVM on training features
    svm_scores = -svm.decision_function(test_feats)  # Invert scores for consistency
    auc = roc_auc_score(test_labels, svm_scores)     # Compute AUC-ROC
    preds = (svm_scores > 0.5).astype(int)            # Threshold-based binary predictions
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average='binary')
    results['OneClassSVM'] = (auc, precision, recall, f1)

    # --- Isolation Forest ---
    print("\nTraining Isolation Forest...")
    iso = IsolationForest(contamination=0.1, random_state=42).fit(train_feats)  # Fit isolation model
    iso_scores = -iso.decision_function(test_feats)  # Consistent score direction
    auc = roc_auc_score(test_labels, iso_scores)     # Evaluate AUC-ROC
    preds = (iso.predict(test_feats) == -1).astype(int)  # Convert anomaly predictions to binary
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average='binary')
    results['IsolationForest'] = (auc, precision, recall, f1)

    return results