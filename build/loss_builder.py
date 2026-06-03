import torch.nn as nn
from losses.contrastive import info_nce
from losses.graph_loss import graph_loss_batch


def build_losses(cfg):

    loss_cfg = cfg["loss"]

    task_losses = {
        "valence": nn.BCEWithLogitsLoss(),
        "arousal": nn.BCEWithLogitsLoss()
    }

    contrastive_loss = info_nce

    loss_graph = graph_loss_batch

    return task_losses, contrastive_loss, loss_graph, loss_cfg