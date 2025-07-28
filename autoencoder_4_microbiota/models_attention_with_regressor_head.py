"""
1. separated decoders for presence and for nonzero value prediction
2. deeper decoder for nonzero value prediction
3. added a regressor 

"""


import torch
import torch.nn as nn


class AttentionAEend(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0, hidden_dim=128, num_heads=4):
        super(AttentionAEend, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, hidden_dim),
            nn.LeakyReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU()
        )
        self.shallowdecoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.deepdecoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim//2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded = self.encoder_out(x)
        decoded1 = self.shallowdecoder(encoded)
        decoded2 = self.deepdecoder(encoded)
        pred = self.regressor(encoded)
        return encoded, decoded1, decoded2, pred


class AttentionAEmid(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0, hidden_dim=128*2, num_heads=4):
        super(AttentionAEmid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, hidden_dim),
            nn.LeakyReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim//8),
            nn.LeakyReLU(),
            nn.Linear(input_dim//8, latent_dim),
            nn.LeakyReLU(),
        )
        self.shallowdecoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.deepdecoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim//2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded = self.encoder_out(x)
        decoded1 = self.shallowdecoder(encoded)
        decoded2 = self.deepdecoder(encoded)
        pred = self.regressor(encoded)
        return encoded, decoded1, decoded2, pred



class AttentionAEend(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0, hidden_dim=128, num_heads=4):
        super(AttentionAEend, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, hidden_dim),
            nn.LeakyReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU()
        )
        self.shallowdecoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.deepdecoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim//2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded = self.encoder_out(x)
        decoded1 = self.shallowdecoder(encoded)
        decoded2 = self.deepdecoder(encoded)
        pred = self.regressor(encoded)
        return encoded, decoded1, decoded2, pred


class AttentionAEmidconcat(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0, hidden_dim=128*2, num_heads=4):
        super(AttentionAEmidconcat, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, latent_dim//2),
            nn.LeakyReLU()
        )        
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, hidden_dim),
            nn.LeakyReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim//8),
            nn.LeakyReLU(),
            nn.Linear(input_dim//8, latent_dim//2),
            nn.LeakyReLU(),
        )
        self.shallowdecoder = nn.Sequential(
            nn.Linear(latent_dim//2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.deepdecoder = nn.Sequential(
            nn.Linear(latent_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim//2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded1 = self.encoder(x)
        x = self.encoder2(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded2 = self.encoder_out(x)
        encoded = torch.cat((encoded1, encoded2), dim=1)
        decoded1 = self.shallowdecoder(encoded1)
        decoded2 = self.deepdecoder(encoded2)
        pred = self.regressor(encoded)
        return encoded, decoded1, decoded2, pred

    

class AttentionAEbegin(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0, hidden_dim=128*4, num_heads=8):
        super(AttentionAEbegin, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.encoder_out = nn.Sequential(
            nn.Linear(hidden_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//8),
            nn.LeakyReLU(),
            nn.Linear(input_dim//8, latent_dim),
            nn.LeakyReLU()
        )
        self.shallowdecoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.deepdecoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim//2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        encoded = self.encoder_out(x)
        decoded1 = self.shallowdecoder(encoded)
        decoded2 = self.deepdecoder(encoded)
        pred = self.regressor(encoded)
        return encoded, decoded1, decoded2, pred



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
        self.shallowdecoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )
        self.deepdecoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim//2, 1),
            nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded1 = self.shallowdecoder(encoded)
        decoded2 = self.deepdecoder(encoded)
        pred = self.regressor(encoded)
        return encoded, decoded1, decoded2, pred

