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

    # CHANGED: three config flags replacing the previous two (config_pipeline +
    # config_train). The pipeline config concept is dissolved — data, training,
    # and loss configs are loaded and merged explicitly.
    parser.add_argument("--config_data",     type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--config_train",    type=str, required=True,
                        help="Path to training.yaml")
    parser.add_argument("--config_loss",     type=str, required=True,
                        help="Path to loss.yaml")

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

    # CHANGED: configs are loaded and merged into a single flat dict.
    # Previously each file was loaded separately and only one was passed
    # to train(), leaving input_dim, paths, free_bits etc. missing.
    config = merge_configs(
        load_config(args.config_data),
        load_config(args.config_train),
        load_config(args.config_loss),
    )

    # Inject single-run architectural choices (swept in sweeps, fixed here)
    config["latent_dim"]    = args.latent_dim
    config["encoder_dims"]  = args.encoder_dims
    config["input_dropout"] = args.input_dropout
    config["seed"]          = args.seed

    run_name = f"betavae_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    metrics = train(config=config, run_name=run_name, seed=config["seed"])

    if args.verbose:
        print("=" * 60)
        print("Training complete")
        print(f"Val loss: {metrics.get('val_loss')}")
        print("=" * 60)


if __name__ == "__main__":
    main()

# [END]