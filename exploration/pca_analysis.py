"""
Quick intrinsic dimensionality check via PCA on methylation beta-VAE training data.

Usage:
    python pca_variance_analysis.py /path/to/data.h5ad [--layer LAYER] [--n_components N] [--out OUT.png]
"""

import argparse
import numpy as np
import anndata as ad
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to anndata (.h5ad) file")
    parser.add_argument("--layer", type=str, default=None,
                         help="adata.layers key to use instead of adata.X")
    parser.add_argument("--n_components", type=int, default=500,
                         help="Max PCA components to compute (default 500)")
    parser.add_argument("--out", type=str, default="pca_variance.png",
                         help="Output figure path")
    args = parser.parse_args()

    print(f"Loading {args.data_path} ...")
    adata = ad.read_h5ad(args.data_path)

    X = adata.layers[args.layer] if args.layer else adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    n_comp = min(args.n_components, X.shape[0] - 1, X.shape[1])
    print(f"Running PCA with n_components={n_comp} on data shape {X.shape} ...")

    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X)

    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Components needed for variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    n_needed = {}
    for t in thresholds:
        idx = np.searchsorted(cum_var, t)
        n_needed[t] = idx + 1 if idx < len(cum_var) else None

    print("\nComponents needed to explain variance:")
    for t in thresholds:
        val = n_needed[t]
        print(f"  {int(t*100)}%: {val if val is not None else f'>{n_comp} (not reached)'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(np.arange(1, n_comp + 1), cum_var, lw=2)
    for t in thresholds:
        axes[0].axhline(t, color="gray", linestyle="--", lw=0.8)
    axes[0].set_xlabel("Number of PCs")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].set_title("Cumulative Explained Variance")
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.arange(1, min(50, n_comp) + 1),
                 pca.explained_variance_ratio_[:50], lw=2)
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained variance ratio")
    axes[1].set_title("Scree Plot (first 50 PCs)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nFigure saved to {args.out}")


if __name__ == "__main__":
    main()