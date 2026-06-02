# ==============================================================================
# Script:           betaVAE.py
# Purpose:          Defines Beta-VAE model architecture.
# Author:           Sophia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             12/31/2025
# ==============================================================================

import math
import torch
import wandb

import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F

from warmup_scheduler import GradualWarmupScheduler
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from typing import cast

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
                 free_bits: float = 0.5,
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
                        Sweep: [1024,256,128] | [2048,512,128] | [2048,1024,256,
                        128]
        decoder_dims  : Hidden layer widths for decoder (typically 
                        reversed encoder).
                        E.g. encoder [2048,512,128] → decoder [512,2048]
        beta          : Maximum KL weight for cyclical annealing.
        free_bits     : Minimum KL per latent dimension (nats), applied before
                        beta scaling. Prevents posterior collapse by ensuring
                        each dimension encodes at least `free_bits` nats.
                        Kingma et al. (2016). Default 0.5.
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
        self.encoder  = MethylEncoder(input_dim, latent_dim, 
                                      encoder_dims, input_dropout)
        self.z_mu     = nn.Linear(latent_dim, latent_dim)
        self.z_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder  = MethylDecoder(input_dim, latent_dim, decoder_dims)


        # Initialize the linear layers using Xavier Initialization
        self.apply(self._init_weights)

        # Initialize a buffer to store logging metrics
        self.val_step_outputs = []

    # -------------------------------------------------------------------------

    def encode(self, x):
        h       = self.encoder(x)
        z_mu    = self.z_mu(h)
        z_logvar = self.z_logvar(h)
        return z_mu, z_logvar

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
        
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        cycle_len   = total_steps / self.hparams['num_cycles']
        
        # Position within the current cycle [0, 1)
        cycle_pos   = (self.global_step % math.ceil(cycle_len)) / cycle_len
        
        # Linear ramp for first half of cycle, hold at max for second half
        annealed    = min(1.0, cycle_pos * 2.0)
        
        return self.hparams['beta'] * annealed

    # -------------------------------------------------------------------------
    
    def forward(self, x):
        z_mu, z_log_var = self.encode(x)
        z = self.reparameterize(z_mu, z_log_var)
        x_hat = self.decoder(z)
        return x_hat, z_mu, z_log_var
    

    def _compute_kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Per-dimension free-bits KL divergence (Kingma et al., 2016).
 
        KL is computed per sample per dimension, then clamped from below at
        `free_bits` per dimension before averaging. This guarantees each latent
        dimension encodes a minimum of `free_bits` nats regardless of beta,
        preventing posterior collapse even during the beta=0 phase of cyclical
        annealing.
 
        Returns the scalar KL term (mean over batch, sum over dimensions),
        after free-bits clamping.
        """
 
        # Per-sample, per-dimension KL: shape (B, D)
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
 
        # Free-bits threshold applied per dimension before aggregation.
        kl_per_dim = torch.clamp(kl_per_dim, min = self.hparams['free_bits'])
 
        # Mean over batch, sum over dimensions — standard ELBO convention
        return kl_per_dim.mean(dim=0).sum()

    
    def compute_loss(self, x: Tensor, x_hat: Tensor,
                     mu: Tensor, logvar: Tensor) -> dict:
        """
        ELBO = MSE reconstruction + beta * KL(q||p).
 
        MSE is the correct reconstruction term for unbounded continuous M-values
        under a Gaussian decoder assumption, consistent with the linear (no
        activation) decoder output layer.
        """
 
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss    = self._compute_kl_loss(mu, logvar)
        beta       = self.get_beta()
        total_loss = recon_loss + beta * kl_loss
 
        return {
            "total_loss":          total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss":             kl_loss,
            "beta":                beta,
        }

    # -------------------------------------------------------------------------
    
    def training_step(self, batch, batch_idx):
        x = batch['methylation_data']
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)

        self.log('train_loss',       losses['total_loss'], prog_bar = True)
        self.log('train_recon',      losses['reconstruction_loss'])
        self.log('train_kl',         losses['kl_loss'])
        self.log('train_kl_per_dim', losses['kl_loss'] 
                 / self.hparams['latent_dim'])
        self.log('beta',             losses['beta'])
        self.log('train_post_var',   logvar.exp().mean())

        return losses['total_loss']
    

    def validation_step(self, batch, batch_idx):
        x = batch['methylation_data']
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)

        val_loss = losses['total_loss'].detach().mean()
        
        self.log('val_loss',       val_loss, prog_bar = True)
        self.log('val_recon',      losses['reconstruction_loss'])
        self.log('val_kl',         losses['kl_loss'])
        self.log('val_kl_per_dim', losses['kl_loss'] 
                 / self.hparams['latent_dim'])
        self.log('val_post_var',   logvar.exp().mean(), prog_bar = True)

        # Save the latent variables for epoch-end diagnostics
        self.val_step_outputs.append({
            'mu': mu.detach(),
            'logvar': logvar.detach()
        })

        return losses['total_loss'].detach()
    

    def test_step(self, batch, batch_idx):
        x = batch["methylation_data"]
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)

        self.log('test_loss',     losses['total_loss'])
        self.log('test_post_var', logvar.exp().mean())

        return losses['total_loss']
    
    # -------------------------------------------------------------------------

    def on_validation_epoch_end(self):
        
        # 1. Aggregate from list (using the buffer strategy we discussed)
        all_mu = torch.cat([x['mu'] for x in self.val_step_outputs], dim=0)
        all_logvar = torch.cat([x['logvar'] for x in 
                                self.val_step_outputs], dim = 0)

        # 2. Latent Space Health Check
        mu_variance = torch.var(all_mu, dim = 0).mean()
        self.log("diag/latent_mu_variance", mu_variance)

        # 3. Log Distributions to WandB
        if self.logger:
            # Tell the IDE: "Treat self.logger as a WandbLogger"
            wandb_logger = cast(WandbLogger, self.logger)
            
            # Access the experiment attribute safely
            if wandb_logger.experiment:
                wandb_logger.experiment.log({
                    "diag/mu_hist": wandb.Histogram(all_mu.cpu().numpy()),
                    "diag/logvar_hist": wandb.Histogram(all_logvar.cpu().numpy())
                })

        # 4. Clear the buffer for the next epoch
        self.val_step_outputs.clear()

    # -------------------------------------------------------------------------

    def configure_optimizers(self): # type: ignore
        """
        Adam optimizer with linear warm-up + cosine annealing LR schedule.
        Warm-up covers the first 5% of training steps.
        """

        # Initialize the Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams['lr'])

        # Initialize the Cosine Annealing Scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = 10000
        )

        # Initialize the Warm-up Scheduler
        scheduler = GradualWarmupScheduler(
            optimizer, 
            multiplier      = 1, 
            total_epoch     = 10, 
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
               interpolation: Tensor,
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
        z = torch.randn(num_samples, 
                        self.hparams['latent_dim']).to(current_device)

        if interpolation is not None:
            z = z + torch.from_numpy(alpha * interpolation).float().to(current_device)

        return self.decoder(z)

    # -------------------------------------------------------------------------
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
