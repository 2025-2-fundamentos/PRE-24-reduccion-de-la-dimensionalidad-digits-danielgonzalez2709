"""Dimensionality reduction visualizations for digits dataset."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def generate_visualizations():
    """Generate PCA, t-SNE and UMAP visualizations of digits dataset."""
    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10)
    plt.colorbar(label='Digit')
    plt.title('PCA - Digits Dataset')
    plt.savefig('digits_pca.png', dpi=150)
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
    plt.colorbar(label='Digit')
    plt.title('t-SNE - Digits Dataset')
    plt.savefig('digits_tsne.png', dpi=150)
    plt.close()

    # UMAP
    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', s=10)
    plt.colorbar(label='Digit')
    plt.title('UMAP - Digits Dataset')
    plt.savefig('digits_umap.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    generate_visualizations()
