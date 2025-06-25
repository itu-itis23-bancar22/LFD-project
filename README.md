# Deep One-Class Classification for Visual Anomaly Detection

This project implements **Deep SVDD (Support Vector Data Description)** integrated with **ResNet18** for anomaly detection on the CIFAR-10 dataset. It trains a one-class classifier using only normal class images and detects anomalies among unseen classes.

## 🔍 Problem

In many real-world applications, only data from normal conditions is available (e.g., working machines or healthy patients). The goal is to detect anomalies without having seen them during training. This project explores one-class classification as a solution.

## 📦 Dataset

We use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consisting of 10 classes (e.g., airplane, car, cat, etc.).  
- **Normal class:** Airplane  
- **Anomalous classes:** The other 9 classes  
- **Training:** Only airplane images  
- **Testing:** All classes (binary labels: normal vs anomaly)

## 🧠 Method

- **Feature extractor:** Pretrained **ResNet18** (final layer removed)
- **One-class model:** **Deep SVDD** which minimizes the volume of a hypersphere around normal features in latent space
- **Baselines:** One-Class SVM and Isolation Forest (on extracted features)


