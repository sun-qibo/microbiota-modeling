import torch
import torch.nn as nn


# todo  add noise (e.g., zero out random values) to the input during training and train the network to reconstruct the original data. This encourages the model to learn robust features despite the sparse noise.


class ShallowAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ShallowAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
            
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, latent_dim),
            nn.LeakyReLU()
        )        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class DeepShallowAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DeepShallowAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//8),
            nn.LeakyReLU(),
            nn.Linear(input_dim//8, latent_dim),
            nn.LeakyReLU()
        )        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class DeepShallowerAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DeepShallowerAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//8),
            nn.LeakyReLU(),
            nn.Linear(input_dim//8, latent_dim),
            nn.LeakyReLU()
        )        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class ShallowVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ShallowVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
        )
        self.mu = nn.Linear( input_dim//2, latent_dim)
        self.logvar = nn.Linear( input_dim//2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        encoded = self.reparameterize(mu, logvar)
        decoded = self.decoder(encoded)
        return encoded, decoded, mu, logvar


class DeepVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DeepVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
        )
        self.mu = nn.Linear( input_dim//4, latent_dim)
        self.logvar = nn.Linear( input_dim//4, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        encoded = self.reparameterize(mu, logvar)
        decoded = self.decoder(encoded)
        return encoded, decoded, mu, logvar




