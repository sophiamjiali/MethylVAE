# ==============================================================================
# Script:           encoder.py
# Purpose:          Defines the Beta-VAE encoder architecture.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
# ==============================================================================

import torch.nn as nn

# =====| Encoder Class |========================================================

class MethylEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list,
                 input_dropout: float = 0.1):
        """
        Parameters
        ----------
        input_dim     : Number of CpG probes (e.g. 211580 for train)
        latent_dim    : Dimension of the bottleneck layer fed into z_mu/z_logvar
        hidden_dims   : List of hidden layer widths for gradual compression.
                        Recommended ranges to sweep:
                          [1024, 256, 128]
                          [2048, 512, 128]
                          [2048, 1024, 256, 128]
        input_dropout : Dropout applied to the raw input only.
                        Sweep: [0.1, 0.2, 0.3]. Default 0.1.
        """

        super(MethylEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.input_dropout = input_dropout

        # Build the encoder architecture
        modules: list[nn.Module] = [nn.Dropout(p = input_dropout)]

        curr_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(curr_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Dropout(p = input_dropout)
                )
            )
            curr_dim = h_dim

        # Linear layer directly into latent dimension
        modules.append(nn.Linear(curr_dim, self.latent_dim))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)
    
# [END]