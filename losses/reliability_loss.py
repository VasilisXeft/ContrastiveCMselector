import torch
import torch.nn as nn


class ReliabilityLoss(nn.Module):
    def __init__(self):
        super(ReliabilityLoss, self).__init__()

    def forward(self, r):
        """
        r: Tensor [B, M]
        """
        # Mean reliability scores
        r_bar = torch.mean(r, dim=1, keepdim=True)  # [B, 1]

        # Loss computation
        loss = torch.mean((r - r_bar) ** 2)
        return loss