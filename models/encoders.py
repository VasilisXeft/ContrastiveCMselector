import torch
import torch.nn as nn
import torch.nn.functional as F
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
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + 1e-8)

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
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + 1e-8)

        x = self.net(x)
        x = self.pool(x)
        x = x.squeeze(-1)

        return self.proj(x)  # [B, D]

class MultiScalePhysioEncoder(nn.Module):
    """
    Ένας πανίσχυρος, Inception-style 1D Encoder για βιοσήματα.
    Χρησιμοποιεί πολλαπλά μεγέθη φίλτρων ταυτόχρονα για να "βλέπει"
    τόσο τις γρήγορες αιχμές (ECG) όσο και τις αργές αλλαγές (EDA/TMP).
    """

    def __init__(self, in_ch=1, emb_dim=64):
        super().__init__()

        # 1. Multi-Scale Feature Extraction (Παράλληλα φίλτρα)
        # Χρησιμοποιούμε padding='same' ώστε τα αποτελέσματα να έχουν ακριβώς το ίδιο μήκος
        self.branch_fast = nn.Conv1d(in_ch, 16, kernel_size=5, padding='same')
        self.branch_mid = nn.Conv1d(in_ch, 16, kernel_size=15, padding='same')
        self.branch_slow = nn.Conv1d(in_ch, 16, kernel_size=31, padding='same')

        # 2. Μίξη των χαρακτηριστικών
        # Έχουμε 16+16+16 = 48 channels από το concatenation
        self.mixer = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),  # Η GELU είναι πιο σταθερή από τη ReLU για συνεχή φυσιολογικά σήματα
            nn.MaxPool1d(2)  # Μικρή μείωση του χρόνου στο μισό
        )

        # 3. Deep Feature Extraction
        self.deep = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        # 4. ΣΥΝΤΗΡΗΣΗ ΤΟΥ ΧΡΟΝΟΥ! (Το αντίδοτο στο Mean Trap)
        # Κρατάμε τις 4 πιο δυνατές "στιγμές" του παραθύρου.
        self.pool = nn.AdaptiveMaxPool1d(4)

        # 5. Τελική προβολή στο embedding dimension (64)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, emb_dim),
            nn.LayerNorm(emb_dim)  # Εξασφαλίζει ότι το Fusion δεν θα δει gradients να εκρήγνυνται
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Εσωτερικό On-the-fly Z-Score Normalization
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + 1e-8)

        # Multi-Scale περάσματα
        x_fast = self.branch_fast(x)
        x_mid = self.branch_mid(x)
        x_slow = self.branch_slow(x)

        # Συνένωση (Concatenation) στον άξονα των channels [B, 48, T]
        x_cat = torch.cat([x_fast, x_mid, x_slow], dim=1)

        # Επεξεργασία
        x_mix = self.mixer(x_cat)
        x_feat = self.deep(x_mix)

        # Pooling & Projection
        x_pooled = self.pool(x_feat)  # [Batch, 128, 4]
        out = self.proj(x_pooled)  # [Batch, 64]

        return out

class ECGEncoder(nn.Module):
    """
    Αποκλειστικός Encoder για Ηλεκτροκαρδιογράφημα (ECG) με 3 κανάλια.
    Σχεδιασμένος για να διατηρεί τα γεωμετρικά peaks (QRS) χωρίς να τα καταστρέφει το Pooling.
    """

    def __init__(self, in_ch=3, emb_dim=64):
        super().__init__()

        # Εξαγωγή χαρακτηριστικών: Σταδιακή μείωση του χρόνου με Strided Convolutions
        self.features = nn.Sequential(
            # Μεγάλο kernel (15) για να πιάσει το αργό κύμα T
            nn.Conv1d(in_ch, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Μικρότερο kernel (7) για τη γεωμετρία
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Focus στις αιχμές
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 🌟 ΤΟ ΜΥΣΤΙΚΟ ΟΠΛΟ: Αντί για 1 σημείο, κρατάμε 4 χρονικές ζώνες!
        # Το MaxPool εξασφαλίζει ότι το QRS peak ΔΕΝ θα χαθεί.
        self.pool = nn.AdaptiveMaxPool1d(4)

        # Μετατρέπουμε το [Batch, 64, 4] σε ένα επίπεδο διάνυσμα [Batch, 256]
        # και μετά το ρίχνουμε στο επιθυμητό emb_dim (64).
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, x):
        # 1. Z-Score Normalization (On-the-fly)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + 1e-8)

        # 2. Εξαγωγή & Pooling
        x = self.features(x)  # Βγάζει π.χ. [B, 64, 32]
        x = self.pool(x)  # Το κάνει ΑΚΡΙΒΩΣ [B, 64, 4]

        # 3. Τελικό Projection
        x = self.proj(x)  # Βγάζει το τέλειο [B, 64]

        return x