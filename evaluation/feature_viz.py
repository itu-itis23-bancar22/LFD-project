# This script visualizes high-dimensional feature representations in 2D using t-SNE or PCA.
# Useful for exploring the structure of learned embeddings in Deep SVDD or similar models.

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

def plot_latent_space(features, labels, method='tsne', title='Latent Space'):
    # Choose dimensionality reduction method: t-SNE or PCA
    reducer = TSNE(n_components=2, perplexity=30) if method == 'tsne' else PCA(n_components=2)

    # Reduce feature dimensionality to 2D
    reduced = reducer.fit_transform(features)

    # Plot the 2D scatter plot of features
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=8)

    # Add legend using scatter plot color mapping
    plt.legend(*scatter.legend_elements(), title="Label")
    plt.title(title)
    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()