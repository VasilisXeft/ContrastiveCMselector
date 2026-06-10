import torch
import torch.nn.functional as F

def reliability_entropy_loss(r):
    eps = 1e-6
    entropy = - (r * torch.log(r + eps) +
                 (1 - r) * torch.log(1 - r + eps))
    return entropy.mean() * 0.001

def reliability_loss(r, penalty):
    """
    r: [B, M] reliability (0-1)
    penalty: [B, M] (0-1 normalized, higher = worse)
    """

    # pairwise differences within modality dimension
    B, M = r.shape

    loss = 0.0
    count = 0

    for i in range(M):
        for j in range(M):
            if i == j:
                continue

            # if penalty_i > penalty_j, then r_i < r_j
            target = torch.sign(penalty[:, j] - penalty[:, i])

            diff = r[:, i] - r[:, j]

            loss += F.relu(diff * target + 0.1).mean()  # margin
            count += 1

    return loss / (count + 1e-6) + reliability_entropy_loss(r)