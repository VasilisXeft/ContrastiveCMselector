import torch
import torch.nn as nn


class CrossModalBlock(nn.Module):

    def __init__(self, dim=64, num_heads=8, dropout=0.1):
        super().__init__()

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_a, feat_b, mask_a=None, mask_b=None):


        # -------------------------
        # Cross Attention
        # A attends to B
        # -------------------------

        a_ca, attn_a = self.cross_attn(
            query=feat_a,
            key=feat_b,
            value=feat_b,
            key_padding_mask=mask_b
        )

        feat_a = self.norm(feat_a + a_ca)

        return feat_a, attn_a