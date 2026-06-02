import torch.nn as nn
from losses.contrastive import info_nce


def build_losses(cfg):

    loss_cfg = cfg["loss"]

    task_losses = {
        "valence": nn.CrossEntropyLoss(),
        "arousal": nn.CrossEntropyLoss()
    }

    contrastive_loss = info_nce

    def graph_loss(edges, fused=None, embeddings=None):
        # placeholder (θα το βελτιώσουμε μετά)
        return 0.0

    return task_losses, contrastive_loss, graph_loss, loss_cfg