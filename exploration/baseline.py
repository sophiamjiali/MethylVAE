"""
Simple plain autoencoder baseline (no KL term) to check how much val R^2 is being
lost to posterior collapse / beta regularization in the beta-VAE.

Usage:
    python plain_ae_baseline.py /path/to/data.h5ad [--layer LAYER] [--latent_dim 128] [--epochs 100]
"""

import argparse
import numpy as np
import anndata as ad
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096), nn.ReLU(),
            nn.Linear(4096, 1024), nn.ReLU(),
            nn.Linear(1024, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 4096), nn.ReLU(),
            nn.Linear(4096, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def main():
    print("Entered main")
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.data_path} ...")
    adata = ad.read_h5ad(args.data_path)
    X = adata.layers[args.layer] if args.layer else adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    print(f"Data shape: {X.shape}")

    X_train, X_val = train_test_split(X, test_size=0.15, random_state=args.seed)
    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)

    model = AE(input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n_train = X_train_t.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = X_train_t[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.shape[0]
        epoch_loss /= n_train

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_recon = model(X_val_t).cpu().numpy()
            val_r2 = r2_score(X_val, val_recon)
            print(f"Epoch {epoch+1}/{args.epochs} | train_loss={epoch_loss:.4f} | val_r2={val_r2:.4f}")

    model.eval()
    with torch.no_grad():
        val_recon = model(X_val_t).cpu().numpy()
    final_r2 = r2_score(X_val, val_recon)
    print(f"\nFinal plain-AE val R^2: {final_r2:.4f}")
    print("Compare against beta-VAE val_r2 ceiling (~0.62) to estimate capacity lost to KL/beta.")


if __name__ == "__main__":
    main()