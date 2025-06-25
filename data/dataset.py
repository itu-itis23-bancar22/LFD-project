# This dataset class loads CIFAR-10 and filters it for one-class classification tasks.
# During training, it only includes one normal class; during testing, it marks all other classes as anomalies.

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch

class CIFAR10OneClass(Dataset):
    def __init__(self, root, normal_class='airplane', train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        self.label_names = self._load_label_names()  # List of CIFAR-10 class names
        self.normal_class = normal_class
        self.normal_class_idx = self.label_names.index(normal_class)  # Index of the normal class

        if train:
            # Load only training data of the normal class
            self.data, self.targets = self._load_data_batches(1, 5)
            mask = np.array(self.targets) == self.normal_class_idx
            self.data = self.data[mask]
            self.targets = np.zeros(len(self.data), dtype=int)  # All normal samples labeled as 0
        else:
            # Load test data and label normal as 0, anomalies as 1
            self.data, self.targets = self._load_data_batches('test_batch')
            self.targets = np.array([0 if y == self.normal_class_idx else 1 for y in self.targets])

    def _load_data_batches(self, start, end=None):
        data_list, label_list = [], []

        if isinstance(start, str):
            batches = [start]  # Single test batch
        else:
            batches = [f"data_batch_{i}" for i in range(start, end + 1)]  # Training batches

        for batch_name in batches:
            file_path = os.path.join(self.root, 'cifar-10-batches-py', batch_name)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data = batch[b'data']  # Raw image data
                labels = batch[b'labels']  # Corresponding labels

                data = data.reshape(-1, 3, 32, 32)  # Reshape to NCHW format
                data = data.astype(np.uint8)

                data_list.append(data)
                label_list.extend(labels)

        return np.concatenate(data_list), label_list

    def _load_label_names(self):
        # Load human-readable label names from meta file
        meta_path = os.path.join(self.root, 'cifar-10-batches-py', 'batches.meta')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            label_names = meta[b'label_names']
            return [name.decode('utf-8') for name in label_names]

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.targets[idx]
        img = torch.tensor(img)  # Convert to tensor

        if self.transform:
            img = self.transform(img)  # Apply data transformations

        return img, label

    def __len__(self):
        return len(self.data)  # Return dataset size