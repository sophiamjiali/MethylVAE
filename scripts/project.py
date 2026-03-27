#!/usr/bin/env python3
# ==============================================================================
# Script:           project.py
# Purpose:          Generate embeddings of input data using a checkpoint.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage:
#   python scripts/generate_embeddings.py \
#       --checkpoint <path_to_checkpoint> \
#       --data_path <path_to_anndata>
# ==============================================================================

import os
import argparse
import sys
from pathlib import Path
from datetime import datetime

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = Path.cwd().parent
sys.path.append(str(project_root / "src"))

from MethylCDM.models.betaVAE import BetaVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Beta-VAE latent-only AnnData objects.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw .h5ad file.")
    parser.add_argument("--output_dir", type=str, default="data/embeddings", help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for projection.")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu.")
    
    # Saving Logic
    parser.add_argument("--split_projects", action="store_true", 
                        help="Save separate .h5ad files for each TCGA project_id.")
    parser.add_argument("--name", type=str, default="pancancer", 
                        help="Custom name for the full cohort output file.")
    
    return parser.parse_args()


@torch.inference_mode()
def project_batches(model, dataloader, device):
    """Computes all embeddings in batches via the encoder."""
    model.eval()
    model.to(device)
    
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Generating Latent Vectors"):
        x = batch[0].to(device)
        z_mu, _, _ = model.encode(x)
        all_embeddings.append(z_mu.cpu().numpy())
    
    return np.vstack(all_embeddings)


def create_latent_adata(source_adata, embeddings, checkpoint_name):
    """
    Constructs a minimalist AnnData object preserving obs and uns.
    """
    # Preserve original metadata
    new_adata = ad.AnnData(obs=source_adata.obs.copy())
    new_adata.uns = source_adata.uns.copy()
    
    # Store latent vectors in the standard slot
    new_adata.obsm['X_embeddings'] = embeddings
    
    # Append new provenance info to .uns without overwriting existing data
    new_adata.uns['embedding_info'] = {
        'model_type': 'BetaVAE',
        'checkpoint_source': checkpoint_name,
        'projection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'latent_dim': embeddings.shape[1]
    }
    return new_adata


def main():
    
    print("~~~~~| Beginning to generate embeddings |~~~~~")
    
    args = parse_args()
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 1. Device Selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 2. Load Data
    print(f"Reading raw data from {args.data_path}...")
    adata = ad.read_h5ad(args.data_path)
    
    # Efficiently prepare data for the model
    X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    dataset = TensorDataset(torch.from_numpy(X_data).float())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Load Model
    print(f"Loading checkpoint: {Path(args.checkpoint).name}")
    model = BetaVAE.load_from_checkpoint(args.checkpoint, map_location=device)

    # 4. Generate Embeddings
    embeddings = project_batches(model, dataloader, device)

    # 5. Saving Logic
    checkpoint_name = Path(args.checkpoint).name

    if args.split_projects:
        print("Splitting latent objects by project_id...")

        # Create a sub-directory in the output direct with the specified name
        project_dir = os.path.join(args.output_dir, args.name)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        unique_projects = adata.obs['project_id'].unique()
        for project_id in unique_projects:

            # Generate embeddings for the given project
            mask = (adata.obs['project_id'] == project_id)
            proj_subset = adata[mask].copy()
            latent_proj = create_latent_adata(proj_subset, embeddings[mask.values], 
                                              checkpoint_name)
            
            save_path = project_dir / f"{project_id}_embeddings.h5ad"
            latent_proj.write_h5ad(save_path)
            print(f" Saved: {save_path} | Samples: {latent_proj.n_obs}")
            
    else:
        # Save the full cohort as a single AnnData object
        full_latent = create_latent_adata(adata, embeddings, checkpoint_name)
        save_path = output_base / f"{args.name}_embeddings.h5ad"
        
        full_latent.write_h5ad(save_path)
        print(f"Saved full latent-only AnnData: {save_path}")

    print("~~~~~| Finished generating embeddings |~~~~~")g


def generate_embeddings(model, sample_tensor):
    """
    Computes beta-VAE latente embeddings on a single sample tensor.
    """
    with torch.no_grad():
        z_mu, _, _ = model.encode(sample_tensor)
    return z_mu.squeeze(0).cpu().numpy()

if __name__ == "__main__":
    main()

# [END]