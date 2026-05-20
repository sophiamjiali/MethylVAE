# ==============================================================================
# Script:           decoder.py
# Purpose:          Defines the Beta-VAE decoder architecture.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
# ==============================================================================

import torch.nn as nn

# =====| Decoder Class |========================================================

class MethylDecoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list):
        """
        Parameters
        ----------
        input_dim   : Number of CpG probes (reconstruction target)
        latent_dim  : Dimensionality of latent z
        hidden_dims : Hidden layer widths (mirror of encoder, reversed).
                      E.g. if encoder_dims=[1024,256,128], pass decoder_dims=[256,1024]
        """
    
        super(MethylDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Build the decoder architecture, mirroring the encoder
        modules = []
        curr_ch = self.latent_dim
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(curr_ch, h_dim),      
                    nn.BatchNorm1d(h_dim),    
                    nn.GELU()
                )
            )
            curr_ch = h_dim

        # Last layer w/o activation, M-values are unbounded
        modules.append(nn.Linear(curr_ch, self.input_dim))
        
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        return self.decoder(z)
    
# [END]