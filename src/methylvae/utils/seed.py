# ==============================================================================
# Script:           seed.py
# Purpose:          Utility functions for environment initialization
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             11/18/2025
# ==============================================================================

import torch
import random

import numpy as np

# =====| Configuration & Environment |==========================================

def init_environment(config):
    """
    Initializes the current runtime environment for reproducibility.
    
    Parameters
    ----------
    config : a configuration object containing:
        - seed (int): integer value for reproducibility
    """

    # Fetch all relevant values from the configurations object
    seed = config.get('seed', -1)

    # Set the seed for all appropriate packages of the pipeline
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# [END]