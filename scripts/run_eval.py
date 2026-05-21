#!/usr/bin/env python3
# ==============================================================================
# Script:           run_eval.py
# Purpose:          Generate latent embeddings from a trained BetaVAE checkpoint
# ==============================================================================

import argparse
from pathlib import Path
from datetime import datetime

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from methylvae.models.betaVAE import BetaVAE


# ==============================================================================
# Args
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/embeddings")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--split_projects", action="store_true")
    parser.add_argument("--name", type=str, default="pancancer")

    return parser.parse_args()


# ==============================================================================
# Embedding computation
# ==============================================================================

@torch.inference_mode()
def project_batches(model, dataloader, device):
    model.eval()
    model.to(device)

    embeddings = []

    for batch in tqdm(dataloader, desc="Encoding"):
        x = batch[0].to(device)
        z_mu, _, _ = model.encode(x)
        embeddings.append(z_mu.cpu().numpy())

    return np.vstack(embeddings)


def create_latent_adata(source_adata, embeddings, checkpoint_name):
    new_adata = ad.AnnData(obs=source_adata.obs.copy())
    new_adata.uns = source_adata.uns.copy()

    new_adata.obsm["X_embeddings"] = embeddings

    new_adata.uns["embedding_info"] = {
        "model_type": "BetaVAE",
        "checkpoint_source": checkpoint_name,
        "projection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latent_dim": embeddings.shape[1],
    }

    return new_adata


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ----------------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------------
    adata = ad.read_h5ad(args.data_path)

    X = np.asarray(adata.X)
    X = torch.from_numpy(X).float()

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------
    model = BetaVAE.load_from_checkpoint(
        args.checkpoint,
        map_location=device
    )

    # ----------------------------------------------------------------------
    # Embedding
    # ----------------------------------------------------------------------
    embeddings = project_batches(model, dataloader, device)

    checkpoint_name = Path(args.checkpoint).name

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------
    if args.split_projects:
        project_dir = Path(args.output_dir) / args.name
        project_dir.mkdir(parents=True, exist_ok=True)

        for project_id in adata.obs["project_id"].unique():
            mask = (adata.obs["project_id"] == project_id)

            latent = create_latent_adata(
                adata[mask].copy(),
                embeddings[mask.values],
                checkpoint_name
            )

            save_path = project_dir / f"{project_id}_embeddings.h5ad"
            latent.write_h5ad(save_path)

    else:
        latent = create_latent_adata(adata, embeddings, checkpoint_name)

        save_path = output_base / f"{args.name}_embeddings.h5ad"
        latent.write_h5ad(save_path)


if __name__ == "__main__":
    main()