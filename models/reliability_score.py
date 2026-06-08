import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityGating(nn.Module):

    def __init__(self, emb_dim, num_modalities, init_lambda=0.1):
        super().__init__()

        self.num_modalities = num_modalities

        self.norm = nn.LayerNorm(emb_dim)

        self.scorers = nn.ModuleList([
            nn.Linear(emb_dim, 1)
            for _ in range(num_modalities)
        ])

        self.lambda_param = nn.Parameter(
            torch.tensor(init_lambda)
        )

    def forward(self, embeddings, signal_quality):

        # [B,M,D]
        embeddings = self.norm(embeddings)

        scores = []

        for m in range(self.num_modalities):

            s = self.scorers[m](
                embeddings[:, m]
            )

            scores.append(s)

        proj = torch.cat(scores, dim=1)

        lambda_param = F.softplus(
            self.lambda_param
        )

        r = torch.sigmoid(
            proj - lambda_param * signal_quality
        )

        gated_embeddings = (
            embeddings *
            r.unsqueeze(-1)
        )

        return gated_embeddings, r