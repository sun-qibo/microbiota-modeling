import torch
import torch.nn as nn
from sklearn.metrics import f1_score


# def masked_mse_loss(reconstructed, X_true):
#     mask = (X_true != 0).float()
#     return (nn.MSELoss(reduction='none')(reconstructed, X_true) * mask).sum() / mask.sum() 


# def masked_kl_loss(mu, logvar, X_true):
#     mask = (X_true !=0).float()
#     kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#     non_zero_ratio = mask.mean(dim=1)
#     kl_weight = 0.1 + 0.9 * (1-non_zero_ratio.unsqueeze(1))
#     return torch.mean(kl_weight * kl_per_dim)


def softly_masked_mse_loss(X_pred_value, X_true, mask, mask_x_true = False):
    if mask_x_true:
        X_true = X_true * mask
    X_pred = X_pred_value * mask
    return (nn.MSELoss(reduction='sum')(X_pred, X_true)) / mask.sum()


def get_f1_score(X_pred_logits, X_true):
    X_pred_presence = (torch.sigmoid(X_pred_logits).cpu().detach().numpy().flatten()> 0.5).astype(int)
    X_true_presence = (X_true.cpu().numpy().flatten() > 0).astype(int)
    return f1_score(X_true_presence, X_pred_presence)