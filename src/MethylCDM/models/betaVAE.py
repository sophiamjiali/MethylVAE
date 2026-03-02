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

from torch import Tensor
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler

# =====| Encoder Class |========================================================

class MethylEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list):
        super(MethylEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build the encoder architecture
        modules = [nn.Sequential(nn.Dropout())]
        curr_ch = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(curr_ch, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            curr_ch = h_dim

        # Last layer interfaces with the latent dimension
        modules.append(nn.Sequential(
            nn.Linear(curr_ch, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU()
        ))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

# =====| Decoder Class |========================================================

class MethylDecoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list):
        super(MethylDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Build the decoder architecture, mirroring the encoder
        modules = []
        curr_ch = self.latent_dim
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(nn.Linear(curr_ch, h_dim),
                              nn.BatchNorm1d(h_dim),
                              nn.LeakyReLU())
            )
            curr_ch = h_dim

        # Last layer interfaces with the latent dimension
        modules.append(nn.Sequential(
            nn.Linear(curr_ch, self.input_dim), 
            nn.Sigmoid()
        ))
        
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        return self.decoder(z)


# =====| LightningModule Class |================================================

class BetaVAE(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_dims: list,
                 decoder_dims: list,
                 beta: float = 0.005,
                 lr: float = 3e-3):
        super(BetaVAE, self).__init__()

        # Save hyperparameters for reproducibility and logging
        self.save_hyperparameters()

        # Initialize model components
        self.encoder = MethylEncoder(input_dim, latent_dim, encoder_dims)
        self.z_mu = nn.Linear(latent_dim, latent_dim)
        self.z_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = MethylDecoder(input_dim, latent_dim, decoder_dims)

        # Initialize the linear layers using Xavier Initialization
        self.apply(self._init_weights)

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
    
    
    def sample(self,
               num_samples: int,
               current_device: int,
               interpolation: Tensor = None,
               alpha: float = 1.0) -> Tensor:
        """
        Samples from the latent space and returns the corresponding gene 
        expression.

        Parameters
        ----------
        num_samples (int): number of samples
        current_device (int): device to run the model
        interpolation (Tensor): difference to move samples in the latent space
        alpha (float): weight for moving in the latent space

        Returns
        -------
        (Tensor): gene expression corresponding to the latent space
        """

        z = torch.randn(num_samples, self.hparams.latent_dim)
        z = z.to(current_device)

        if interpolation is not None:
            z = (z + torch.from_numpy(alpha * interpolation)
                    .float().to(current_device))
            
        samples = self.decoder(z)
        return samples
    

    def forward(self, x):
        z_mu, z_log_var, z = self.encode(x)
        z = self.reparameterize(z_mu, z_log_var)
        x_hat = self.decoder(z)

        return x_hat, z_mu, z_log_var
    

    def compute_loss(self, x, x_hat, mu, logvar):
        recon_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1),
            dim = 0
        )
        total_loss = recon_loss + self.hparams.beta * kld_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kld_loss
        }


    def training_step(self, batch, batch_idx):
        x = batch['methylation_data']
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)

        self.log('train_loss', losses['total_loss'], prog_bar = True)
        self.log('train_recon', losses['reconstruction_loss'])
        self.log('train_kl', losses['kl_loss'])
        self.log('train_kl_per_dim', 
                 losses['kl_loss'] / self.hparams.latent_dim)

        return losses['total_loss']
    

    def validation_step(self, batch, batch_idx):
        x = batch['methylation_data']
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)

        self.log('val_loss', losses['total_loss'], prog_bar = True)
        self.log('val_recon', losses['reconstruction_loss'])
        self.log('val_kl', losses['kl_loss'])
        self.log('val_kl_per_dim', 
                 losses['kl_loss'] / self.hparams.latent_dim)

    def test_step(self, batch, batch_idx):
        x = batch["methylation_data"]
        x_hat, mu, logvar = self(x)
        losses = self.compute_loss(x, x_hat, mu, logvar)
        self.log('test_loss', losses['total_loss'])
    

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Uses the Adam optimizer with learning rate defined by `self.hparams.lr`.
        Implements a linear warm-up for the first few steps followed by a cosine annealing schedule for the remainder of training.

        Returns
        -------
        (dict): Dictionary containing the optimizer, LR scheduler, and 
                scheduling for PyTorch Lightning Compatibility
        """

        # Initialize the Adam Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr)

        # Initialize the Cosine Annealing Scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = self.trainer.estimated_stepping_batches
        )

        # Initialize the Warm-up Scheduler
        warmup_steps = int(0.05 * self.trainer.estimated_stepping_batches)
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
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
