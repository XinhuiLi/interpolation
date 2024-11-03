import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, slope=0.5, seed=None):
        super(VAE, self).__init__()
        
        self.seed = seed
        self.set_seed()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(nn.LeakyReLU(negative_slope=slope))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            decoder_layers.append(nn.LeakyReLU(negative_slope=slope))
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def loss(self, xr, x, mean, logvar):
        reconstruction_loss = F.mse_loss(xr, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kl_loss
    
    def set_seed(self, seed_torch = True):
        if self.seed is None:
            self.seed = np.random.choice(2 ** 32)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if seed_torch:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True