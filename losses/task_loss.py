import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        alpha_t = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha
        )

        loss = alpha_t * (1 - pt) ** self.gamma * bce

        return loss.mean()

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.binary_cross_entropy_loss = nn.BCELoss(reduction="none")

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        bce = self.binary_cross_entropy_loss(probs, targets)

        return bce.mean()