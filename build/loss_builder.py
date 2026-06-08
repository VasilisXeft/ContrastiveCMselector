from losses.contrastive import info_nce
from losses.graph_loss import graph_loss_batch
from losses.task_loss import FocalLoss
from losses.reliability_loss import reliability_loss


def build_losses(cfg):

    loss_cfg = cfg["loss"]

    task_losses = {
        "valence": FocalLoss(alpha=0.5, gamma=2.0),
        "arousal": FocalLoss(alpha=0.5, gamma=2.0)
    }

    contrastive_loss = info_nce

    loss_graph = graph_loss_batch

    loss_rel = reliability_loss

    return task_losses, contrastive_loss, loss_graph, loss_rel, loss_cfg