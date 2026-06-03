#!/usr/bin/env python3
# ==============================================================================
# Script:           run_train.py
# Purpose:          Entry-point for a single BetaVAE training run.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage:
#   python run_train.py \
#     --config_data    config/data.yaml    \
#     --config_train   config/training.yaml \
#     --config_loss    config/loss.yaml
# ==============================================================================

import argparse
from datetime import datetime

from methylvae.utils.config import load_config, merge_configs
from methylvae.training.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Single BetaVAE training run.")
    parser.add_argument("--name", type = str, required=True,
                        help="Name of the project")
    parser.add_argument("--config_dir",     type=str, required=True,
                        help="Path to configurations directory")

    # Fixed single-run hyperparameters not covered by the YAMLs
    parser.add_argument("--latent_dim",      type=int,   default=128)
    parser.add_argument("--encoder_dims",    type=int,   nargs="+",
                        default=[2048, 512, 128],
                        help="Encoder hidden dims, e.g. --encoder_dims 2048 512 128")
    parser.add_argument("--input_dropout",   type=float, default=0.1)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--verbose",         action="store_true", default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve the configuration paths
    config = merge_configs(
        load_config(f"{args.config_dir}/base.yaml"),
        load_config(f"{args.config_dir}/data.yaml"),
        load_config(f"{args.config_dir}/base.yaml"),
        load_config(f"{args.config_dir}/train.yaml"),
    )

    # Inject single-run architectural choices (swept in sweeps, fixed here)
    config["latent_dim"]    = args.latent_dim
    config["encoder_dims"]  = args.encoder_dims
    config["input_dropout"] = args.input_dropout
    config["seed"]          = args.seed

    run_name = f"train_{args.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.verbose:
        print("=" * 60)
        print("BetaVAE Training Run")
        print("=" * 60)
        print(f"Run:          {run_name}")
        print(f"Seed:         {config['seed']}")
        print(f"Epochs:       {config['max_epochs']}")
        print(f"Latent dim:   {config['latent_dim']}")
        print(f"Encoder dims: {config['encoder_dims']}")
        print(f"Beta:         {config['beta']}")
        print(f"Free bits:    {config['free_bits']}")
        print("=" * 60)

    metrics = train(config = config, run_name = run_name, seed=config["seed"])

    if args.verbose:
        print("=" * 60)
        print("Training complete")
        print(f"Val loss: {metrics.get('val_loss')}")
        print("=" * 60)


if __name__ == "__main__":
    main()

# [END]