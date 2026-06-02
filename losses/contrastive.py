# losses/contrastive.py
import torch
import torch.nn.functional as F


def info_nce(z, z_pos, temperature=0.1):

    z = F.normalize(z, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)

    logits = torch.mm(z, z_pos.T) / temperature

    labels = torch.arange(z.shape[0]).to(z.device)

    return F.cross_entropy(logits, labels)