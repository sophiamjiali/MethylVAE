#!/usr/bin/env python3
# ==============================================================================
# Script:           train_betaVAE.py
# Purpose:          Entry-point for a single BetaVAE training run.
#                   Trains the model using fixed parameters from betaVAE.yaml.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
#
# Usage:
#   python scripts/train_betaVAE.py \
#       --config_pipeline pipeline.yaml \
#       --config_train    betaVAE_train.yaml \
#       --seed            42 \
#       --verbose         True
# ==============================================================================

import argparse
from datetime import datetime
from pathlib import Path

from MethylCDM.utils.utils import init_environment, load_config, resolve_path
from MethylCDM.training.train import run_training
from MethylCDM.constants import BETAVAE_CHECKPOINT_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Single training run for BetaVAE."
    )
    parser.add_argument(
        "--config_pipeline", type=str, required=True,
        help="Path to pipeline.yaml."
    )
    parser.add_argument(
        "--config_train", type=str, required=True,
        help="Path to betaVAE.yaml."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--verbose", type=bool, default=True,
        help="Print training progress."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configs
    pipeline_cfg = load_config(args.config_pipeline)
    train_cfg    = load_config(args.config_train)

    # Initialise environment
    init_environment(pipeline_cfg)

    # Generate a unique run name
    run_name = f"betaVAE_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.verbose:
        print("=" * 60)
        print(f"~~~~~| BetaVAE Single Training Run")
        print("=" * 60)
        print(f"  Run Name:  {run_name}")
        print(f"  Seed:      {args.seed}")
        print(f"  Max Epochs: {train_cfg.get('max_epochs', 'N/A')}")
        print("-" * 60)

    # Execute the training
    # Note: We pass the seed explicitly to override any trial-based logic
    val_loss = run_training(
        config=train_cfg, 
        run_name=run_name, 
        seed=args.seed
    )

    if args.verbose:
        print("-" * 60)
        print(f"  Training Completed.")
        print(f"  Final Validation Loss: {val_loss:.5f}" if val_loss else "  Final Loss: N/A")
        print("=" * 60)

if __name__ == "__main__":
    main()