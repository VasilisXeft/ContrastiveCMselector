import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class FaceMobileNetEncoder(nn.Module):
    def __init__(self, embed_dim=128, pretrained=True):
        super().__init__()

        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # remove classifier head
        self.features = backbone.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        """
        x: [B, 3, H, W]
        """

        with torch.no_grad():
            x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)

        emb = self.proj(x)

        return emb

class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, mask=None):
        """
        x: [B, T, D]
        mask: [B, T] (optional)
        """

        # attention scores
        scores = self.attn(x).squeeze(-1)  # [B, T]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)  # [B, T]

        # weighted sum
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, D]

        return out, weights


class VisualEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.frame_encoder = FaceMobileNetEncoder(emb_dim)
        for p in self.frame_encoder.features.parameters():
            p.requires_grad = False
        for p in self.frame_encoder.features[-2:].parameters():
            p.requires_grad = True

        self.temp_attn = TemporalAttentionPooling(emb_dim)
    def forward(self, x, mask=None):
        B, T, C, H, W = x.shape

        # -------------------------
        # 1. reshape to frame batch
        # -------------------------
        x = x.view(B * T, C, H, W)

        # -------------------------
        # 2. per-frame encoding
        # -------------------------
        frame_emb = self.frame_encoder(x)  # [B*T, D]

        # -------------------------
        # 3. restore temporal structure
        # -------------------------
        frame_emb = frame_emb.view(B, T, -1)  # [B, T, D]

        # -------------------------
        # 4. temporal pooling
        # -------------------------
        video_emb, attn_weights = self.temp_attn(frame_emb, mask)

        return video_emb

class EEGNetEncoder(nn.Module):

    def __init__(self, n_chans=32, emb_dim=128):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.GroupNorm(4, 8),
            nn.ELU()
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (n_chans, 1), groups=8, bias=False),
            nn.GroupNorm(8, 16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 32, (1, 1), bias=False),
            nn.GroupNorm(8, 32),
            nn.ELU(),
            nn.AvgPool2d((1, 8))
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(32, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)

        x = x.squeeze(2)  # [B, F, T']
        x = self.pool(x)  # [B, F, 1]
        x = x.squeeze(-1)  # [B, F]

        x = self.proj(x)  # [B, D]

        return x

class PhysioEncoder(nn.Module):
    def __init__(self, in_ch=1, emb_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Linear(64, emb_dim)

    def forward(self, x):
        if x.dim() == 2: # x: [B, T]
            x = x.unsqueeze(1)

        x = self.net(x)
        x = self.pool(x)
        x = x.squeeze(-1)

        return self.proj(x)  # [B, D]