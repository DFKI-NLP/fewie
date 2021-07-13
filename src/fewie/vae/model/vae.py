import torch.nn as nn
import torch
import torch.utils
import torch.distributions
from fewie.vae.model.encoders.simple import *
from fewie.vae.model.decoders.simple import*

class VAE(nn.Module):
    def __init__(self, dims, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(dims, device)
        self.decoder = Decoder(dims)
        self.float()
    def forward(self, x):
        z=self.encoder(x)
        y=self.decoder(z)
        return y
    def predict_latent(self, x):
        return self.encoder(x)
    def use_latent(self,z):
        return self.decoder(z)