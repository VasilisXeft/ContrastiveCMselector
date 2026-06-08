import torch
import torch.nn as nn


def reliability_loss(r):
    """
    r: Tensor [B, M]
    """
    # Mean reliability scores
    r_bar = torch.mean(r, dim=1, keepdim=True)  # [B, 1]

    # Loss computation
    loss = torch.mean((r - r_bar) ** 2)
    return loss