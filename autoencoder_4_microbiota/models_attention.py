import torch
import torch.nn as nn


class AttentionAEend(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128, num_heads=4):
        super(AttentionAEend, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, hidden_dim),
            nn.LeakyReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded = self.encoder_out(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AttentionAEmid(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128*2, num_heads=4):
        super(AttentionAEmid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, hidden_dim),
            nn.LeakyReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim//8),
            nn.LeakyReLU(),
            nn.Linear(input_dim//8, latent_dim),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded = self.encoder_out(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

class AttentionAEbegin(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128*4, num_heads=8):
        super(AttentionAEbegin, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//8),
            nn.LeakyReLU(),
            nn.Linear(input_dim//8, latent_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded = self.encoder_out(x)
        decoded = self.decoder(encoded)
        return encoded, decoded