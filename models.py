
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_channels):
        #Todo make work for not 32x32 images: change self.mu and self.logvar input dims
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        layers = []
        for channel in hidden_channels:
            layers.append(nn.Conv2d(in_channels, channel, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.ReLU())
            in_channels = channel

        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_channels[-1], latent_dim)
        self.logvar = nn.Linear(hidden_channels[-1], latent_dim)

    def forward(self, x, training=True):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_channels):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        H, W, C = self.output_dim
        factor = 2 ** len(self.hidden_channels)
        assert (
            H % factor == W % factor == 0
        ), f"output_dim must be a multiple of {factor}"
        H, W = H // factor, W // factor

        self.fc = nn.Linear(latent_dim, H * W * self.hidden_channels[-1])

        self.reshape = lambda x: x.view(-1, self.hidden_channels[-1], H, W)

        layers = []
        in_channels = hidden_channels[-1]
        for channel in reversed(hidden_channels[:-1]):
            #CHANGE Double check the padding and output padding
            layers.append(nn.ConvTranspose2d(in_channels, channel, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.ReLU())
            in_channels = channel
        layers.append(nn.ConvTranspose2d(in_channels, output_dim[2], kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Sigmoid())
        
        self.convolutions = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.relu(x)
        x = self.reshape(x)
        x = self.convolutions(x)
        return x

class VAE(nn.Module):
    #reimplements the VAE from https://github.com/probml/probml-utils/blob/main/probml_utils/conv_vae_flax_utils.py in pytorch
    def __init__(self, in_channels, latent_dim, output_dim, hidden_channels, variational=True):
        super(VAE, self).__init__()
        self.variational = variational
        self.encoder = Encoder(in_channels, latent_dim, hidden_channels)
        self.decoder = Decoder(latent_dim, output_dim, hidden_channels)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, training=True):
        mean, logvar = self.encoder(x)
        if self.variational:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        recon = self.decoder(z)
        return recon, mean, logvar

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

