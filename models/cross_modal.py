import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    """Single attention head"""

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        # Initialising weights
        self.wk = nn.Linear(embedding_dim, attention_dim, bias=False)  # [batch_size, num_words, emdedding_dim]
        self.wq = nn.Linear(embedding_dim, attention_dim, bias=False)  # [batch_size, num_words, emdedding_dim]
        self.wv = nn.Linear(embedding_dim, attention_dim, bias=False)  # [batch_size, num_words, emdedding_dim]

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # calculating Query, Key and Value
        q = self.wq(embedded)  # [num_sentence, num_words, attn_dim]
        k = self.wk(embedded)  # [num_sentence, num_words, attn_dim]
        v = self.wv(embedded)  # [num_sentence, num_words, attn_dim]

        # calculating attention scores
        attn_score = q @ torch.transpose(k, 1, 2) / (k.shape[-1] ** 0.5)  # [batch_size, num_words, num_words]

        # below 2 lines is for masking in decoder block, comment it for encoder block
        upper_triangular = torch.triu(attn_score, diagonal=1).bool()
        attn_score[upper_triangular] = float("-inf")

        # applying softmax
        attn_score_softmax = nn.functional.softmax(attn_score, dim=-1)  # [batch_size, num_words, num_words]

        # getting weighted values by multiplying softmax of attention score with values
        weighted_values = attn_score_softmax @ v  # [batch_size, num_words, attention_dim]

        return weighted_values  # [batch_size, num_words, attention_dim]

class CrossModalBlock(nn.Module):

    def __init__(self, dim=64, num_heads=8, dropout=0.1):
        super().__init__()

        # Self attention
        self.self_attn = SingleHeadAttention(dim, dim)

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
        # Self Attention before fusion
        # -------------------------

        feat_a = self.self_attn(feat_a)
        feat_b = self.self_attn(feat_b)

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

        a_ca = self.self_attn(a_ca)

        feat_a = self.norm(feat_a + a_ca)

        return feat_a, attn_a