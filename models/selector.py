import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedContrastiveSelector(nn.Module):
    """
    Learns a directed sparse modality graph:
    i -> j edges are asymmetric and learned.
    """

    def __init__(self, dim=64, proj_dim=64, hidden=128, temperature=0.1, top_k=2):
        super().__init__()

        self.top_k = top_k
        self.temperature = temperature

        # shared projection space
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # directed edge scorer: i -> j
        self.edge_mlp = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, modality_feats):
        """
        modality_feats: list of [B, T, D] or [B, D]

        Returns:
            directed_edges: list of (i, j, score)
            score_matrix: [M, M]
        """

        M = len(modality_feats)

        # ---- pool ----
        pooled = []
        for x in modality_feats:
            if x.dim() == 3:
                x = x.mean(dim=1)
            pooled.append(x)

        x = torch.stack(pooled)  # [B, M, D]

        # ---- projection ----
        z = self.proj(x)  # [B, M, P]
        z = F.normalize(z, dim=-1)

        B, M, P = z.shape

        # ---- build all directed pairs ----
        zi = z.unsqueeze(2).expand(B, M, M, P)
        zj = z.unsqueeze(1).expand(B, M, M, P)

        edge_input = torch.cat([
            zi,
            zj,
            zi - zj,
            zi * zj
        ], dim=-1)  # [B, M, M, 4P]

        # ---- score directed edges ----
        scores = self.edge_mlp(edge_input).squeeze(-1)  # [B, M, M]

        mean_scores = scores.mean(dim=0)

        mask = torch.eye(M, device=scores.device).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

        # ---- top-k outgoing edges per node ----
        directed_edges = []

        for i in range(M):
            topk_vals, topk_idx = torch.topk(mean_scores[i], self.top_k)

            for k, j in enumerate(topk_idx.tolist()):
                directed_edges.append((i, j, topk_vals[k].item()))
