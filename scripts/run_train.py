#!/usr/bin/env python3
# ==============================================================================
# Script:           train_betaVAE.py
# Purpose:          Entry-point for a single BetaVAE training run.
#                   Trains the model using fixed parameters from betaVAE.yaml.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# ==============================================================================

import argparse
from datetime import datetime

from methylvae.utils.seed import init_environment
from methylvae.utils.config import load_config
from methylvae.training.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Single BetaVAE training run.")

    parser.add_argument(
        "--config_pipeline",
        type=str,
        required=True,
        help="Path to pipeline.yaml"
    )

    parser.add_argument(
        "--config_train",
        type=str,
        required=True,
        help="Path to betaVAE training config"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ----------------------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------------------
    pipeline_cfg = load_config(args.config_pipeline)
    train_cfg = load_config(args.config_train)

    # ----------------------------------------------------------------------
    # Environment setup
    # ----------------------------------------------------------------------
    init_environment(pipeline_cfg)

    # ----------------------------------------------------------------------
    # Run identity
    # ----------------------------------------------------------------------
    run_name = f"betavae_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.verbose:
        print("=" * 60)
        print("BetaVAE Training Run")
        print("=" * 60)
        print(f"Run:  {run_name}")
        print(f"Seed: {args.seed}")
        print(f"Epochs: {train_cfg.get('max_epochs')}")
        print("=" * 60)

    # ----------------------------------------------------------------------
    # Execute training
    # ----------------------------------------------------------------------
    val_loss = train(
        config = train_cfg,
        run_name = run_name,
        seed = args.seed
    )

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------
    if args.verbose:
        print("=" * 60)
        print("Training complete")
        print(f"Validation loss: {val_loss}")
        print("=" * 60)


if __name__ == "__main__":
    main()

# [END]