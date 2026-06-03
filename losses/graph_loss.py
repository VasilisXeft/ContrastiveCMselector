import torch
import torch.nn.functional as F

def graph_loss(z, scores):
    """
    z: [M, D]
    scores: [M, M]
    """

    # ensure non-negative weights
    W = F.relu(scores)

    z_i = z.unsqueeze(1)   # [M, 1, D]
    z_j = z.unsqueeze(0)   # [1, M, D]

    diff = (z_i - z_j) ** 2   # [M, M, D]

    dist = diff.sum(dim=-1)   # [M, M]

    loss = (W * dist).sum() / (W.sum() + 1e-8)

    return loss

def graph_loss_batch(z, scores):
    """
    z: [B, M, D]
    scores: [M, M]
    """

    B = len(z)

    loss = 0.0
    for b in range(B):
        loss += graph_loss(z[b], scores)

    return loss / B