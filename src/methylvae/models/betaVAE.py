# ==============================================================================
# Script:           betaVAE.py
# Purpose:          Defines Beta-VAE model architecture.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
#
# Configurations:   betaVAE.yaml
#
# Notes:            Compatible with PyTorch Lightning
# ==============================================================================

import math
import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from warmup_scheduler import GradualWarmupScheduler
from torch import Tensor

from methylvae.models.encoder import MethylEncoder
from methylvae.models.decoder import MethylDecoder


# =====| LightningModule Class |================================================

class BetaVAE(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_dims: list,
                 decoder_dims: list,
                 beta: float = 0.005,
                 input_dropout: float = 0.1,
                 num_cycles: int = 4,
                 lr: float = 3e-3):
        """
        Parameters
        ----------
        input_dim     : Number of CpG probes
        latent_dim    : Latent space dimensionality.
                        Sweep: [64, 128, 256]. Default 128.
        encoder_dims  : Hidden layer widths for encoder.
                        Sweep: [1024,256,128] | [2048,512,128] | [2048,1024,256,128]
        decoder_dims  : Hidden layer widths for decoder (typically reversed encoder).
                        E.g. encoder [2048,512,128] → decoder [512,2048]
        beta          : Maximum KL weight for cyclical annealing.
                        Sweep: [0.001, 0.005, 0.01]. Default 0.005.
        input_dropout : Dropout on raw CpG input.
                        Sweep: [0.1, 0.2, 0.3]. Default 0.1.
        num_cycles    : Number of cyclical annealing cycles over full training.
                        Sweep: [2, 4]. Default 4.
        lr            : Adam learning rate. Default 3e-3.
        """
        
        super(BetaVAE, self).__init__()

        # Save hyperparameters for reproducibility and logging
        self.save_hyperparameters()

        # Initialize model components
        self.encoder = MethylEncoder(input_dim, latent_dim, encoder_dims, input_dropout)
        self.z_mu = nn.Linear(latent_dim, latent_dim)
        self.z_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = MethylDecoder(input_dim, latent_dim, decoder_dims)

        # Initialize the linear layers using Xavier Initialization
        self.apply(self._init_weights)

    # -------------------------------------------------------------------------

    def encode(self, x):
        z = self.encoder(x)
        z_mu = self.z_mu(z)
        z_log_var = self.z_logvar(z)
        return z_mu, z_log_var, z
    

    def decode(self, z):
        return self.decoder(z)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -------------------------------------------------------------------------
    
    def get_beta(self) -> float:
        """
        Cyclical annealing schedule (Fu et al., 2019).

        Within each cycle, beta ramps linearly from 0 → beta_max over the
        first half of the cycle, then holds at beta_max for the second half.
        This prevents posterior collapse by periodically releasing KL pressure,
        ensuring all latent dimensions remain active — important for a
        downstream diffusion model that relies on a well-utilised latent space.

        Sweep num_cycles ∈ [2, 4], beta_max ∈ [0.001, 0.005, 0.01].
        """
        
        total_steps = self.trainer.estimated_stepping_batches
        cycle_len   = total_steps / self.hparams.num_cycles
        
        # Position within the current cycle [0, 1)
        cycle_pos   = (self.global_step % math.ceil(cycle_len)) / cycle_len
        
        # Linear ramp for first half of cycle, hold at max for second half
        annealed    = min(1.0, cycle_pos * 2.0)
        
        return self.hparams.beta * annealed

    # -------------------------------------------------------------------------
    
    def forward(self, x):
        z_mu, z_log_var, _ = self.encode(x)
        z = self.reparameterize(z_mu, z_log_var)
        x_hat = self.decoder(z)

        return x_hat, z_mu, z_log_var

    
    def compute_loss(self, x, x_hat, mu, logvar):
        """
        MSE reconstruction loss — correct for unbounded continuous M-values.
        Assumes a Gaussian decoder p(x|z), consistent with the linear
        (no activation) decoder output layer.
        """
        
        recon_loss = F.mse_loss(x_hat, x, reduction = "mean")
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1),
            dim = 0
        )
        
        beta = self.get_beta()
        total_loss = recon_loss + beta * kld_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kld_loss,
            'beta': beta
        }

    # -------------------------------------------------------------------------
    
    def training_step(self, batch, batch_idx):
        x = batch['methylation_data']
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)

        self.log('train_loss',         losses['total_loss'], prog_bar = True)
        self.log('train_recon',        losses['reconstruction_loss'])
        self.log('train_kl',           losses['kl_loss'])
        self.log('train_kl_per_dim',   losses['kl_loss'] / self.hparams.latent_dim)
        self.log('beta',               losses['beta'])
        self.log('train_post_var',     logvar.exp().mean())

        return losses['total_loss']
    

    def validation_step(self, batch, batch_idx):
        x = batch['methylation_data']
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)

        self.log('val_loss',       losses['total_loss'], prog_bar = True)
        self.log('val_recon',      losses['reconstruction_loss'])
        self.log('val_kl',         losses['kl_loss'])
        self.log('val_kl_per_dim', losses['kl_loss'] / self.hparams.latent_dim)
        self.log('mean_posterior_var', logvar.exp().mean(), prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x = batch["methylation_data"]
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)
        self.log('test_loss', losses['total_loss'])
        self.log('test_post_var',      logvar.exp().mean())

    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        """
        Adam optimizer with linear warm-up + cosine annealing LR schedule.
        Warm-up covers the first 5% of training steps.
        """

        # Initialize the Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr)

        # Initialize the Cosine Annealing Scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = self.trainer.estimated_stepping_batches
        )

        # Initialize the Warm-up Scheduler
        warmup_steps = max(1, int(0.05 * self.trainer.estimated_stepping_batches))
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier = 1, total_epoch = warmup_steps, 
            after_scheduler = cosine_scheduler
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "val_loss"
            }
        }

    # -------------------------------------------------------------------------
        
    def sample(self,
               num_samples: int,
               current_device: int,
               interpolation: Tensor = None,
               alpha: float = 1.0) -> Tensor:
        """
        Samples from the latent space and returns reconstructed M-value profiles.

        Parameters
        ----------
        num_samples    : Number of samples to generate
        current_device : Device index
        interpolation  : Direction vector to shift samples in latent space
        alpha          : Step size for latent interpolation

        Returns
        -------
        Tensor : Reconstructed M-value profiles of shape (num_samples, input_dim)
        """
        z = torch.randn(num_samples, self.hparams.latent_dim).to(current_device)

        if interpolation is not None:
            z = z + torch.from_numpy(alpha * interpolation).float().to(current_device)

        return self.decoder(z)

    # -------------------------------------------------------------------------
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
