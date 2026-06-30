"""
Exploratory analysis of per-probe marginal distributions for methylation data.

Checks whether probes are bimodal (typical of raw beta-values / hypo-hypermethylation)
even after Z-score normalization, which would indicate a Gaussian reconstruction
likelihood is mismatched to the data.

Usage:
    python marginal_distribution_analysis.py /path/to/data.h5ad [--layer LAYER] [--n_probes N] [--out OUT.png]
"""

import argparse
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scipy import stats


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient. >0.555 suggests bimodality (for finite samples)."""
    n = len(x)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=False)  # Pearson kurtosis (normal=3)
    bc = (skew ** 2 + 1) / (kurt + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))
    return bc


def dip_test_pval_proxy(x, n_bins=50):
    """
    Lightweight proxy for multimodality: count local maxima in the smoothed histogram.
    Not a substitute for Hartigan's dip test but needs no extra dependency.
    """
    hist, _ = np.histogram(x, bins=n_bins)
    # simple 3-point smoothing
    smoothed = np.convolve(hist, [1, 2, 1], mode="same") / 4
    peaks = 0
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks += 1
    return peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to anndata (.h5ad) file")
    parser.add_argument("--layer", type=str, default=None,
                         help="adata.layers key to use instead of adata.X")
    parser.add_argument("--n_probes", type=int, default=12,
                         help="Number of probes to sample for histogram grid (default 12)")
    parser.add_argument("--out", type=str, default="marginal_distributions.png",
                         help="Output figure path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.data_path} ...")
    adata = ad.read_h5ad(args.data_path)

    X = adata.layers[args.layer] if args.layer else adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_probes_total = X.shape
    print(f"Data shape: {X.shape}")

    # --- Bimodality coefficient across ALL probes ---
    print("Computing bimodality coefficient across all probes (this may take a moment)...")
    bcs = np.array([bimodality_coefficient(X[:, j]) for j in range(n_probes_total)])
    frac_bimodal = np.mean(bcs > 0.555)
    print(f"Fraction of probes with bimodality coefficient > 0.555: {frac_bimodal:.3f}")
    print(f"Mean BC: {bcs.mean():.3f}, Median BC: {np.median(bcs):.3f}")

    # --- Peak count proxy across all probes ---
    print("Estimating local-maxima peak counts (multimodality proxy)...")
    peak_counts = np.array([dip_test_pval_proxy(X[:, j]) for j in range(n_probes_total)])
    frac_multipeak = np.mean(peak_counts >= 2)
    print(f"Fraction of probes with >=2 histogram peaks: {frac_multipeak:.3f}")

    # --- Sample probes for visual inspection ---
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_probes_total, size=min(args.n_probes, n_probes_total), replace=False)

    n_cols = 4
    n_rows = int(np.ceil(len(sample_idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, sample_idx):
        vals = X[:, idx]
        ax.hist(vals, bins=40, color="steelblue", alpha=0.8)
        bc = bimodality_coefficient(vals)
        ax.set_title(f"Probe {idx} | BC={bc:.2f}", fontsize=10)
        ax.set_xlabel("Z-score")

    for ax in axes[len(sample_idx):]:
        ax.axis("off")

    fig.suptitle(
        f"Marginal probe distributions (n={n_samples} samples)\n"
        f"Bimodal fraction (BC>0.555): {frac_bimodal:.2%} | "
        f"Multi-peak fraction (>=2 peaks): {frac_multipeak:.2%}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=150)
    print(f"\nFigure saved to {args.out}")

    # --- Summary histogram of BC across all probes ---
    bc_hist_path = args.out.replace(".png", "_bc_distribution.png")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(bcs, bins=50, color="darkorange", alpha=0.8)
    ax2.axvline(0.555, color="black", linestyle="--", label="BC=0.555 threshold")
    ax2.set_xlabel("Bimodality coefficient")
    ax2.set_ylabel("Number of probes")
    ax2.set_title("Distribution of Bimodality Coefficients Across All Probes")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(bc_hist_path, dpi=150)
    print(f"BC distribution figure saved to {bc_hist_path}")


if __name__ == "__main__":
    main()