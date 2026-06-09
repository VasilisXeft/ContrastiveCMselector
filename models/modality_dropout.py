import torch
import torch.nn as nn


class ModalityDropout(nn.Module):

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, embeddings):

        if not self.training:
            return embeddings

        B, M, D = embeddings.shape

        mask = torch.bernoulli(
            torch.full((B, M), 1 - self.p,
                       device=embeddings.device)
        )

        # να μην σβήσουν όλα
        all_zero = mask.sum(dim=1) == 0

        if all_zero.any():
            idx = torch.randint(0, M, (all_zero.sum(),), device=embeddings.device)

            mask[all_zero] = 0
            mask[all_zero, idx] = 1

        return embeddings * mask.unsqueeze(-1)