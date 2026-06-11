import torch


class LossRouter:

    def __init__(
        self,
        task_losses: dict,
        contrastive_loss,
        graph_loss,
        reliability_loss,
        lambda_cfg: dict
    ):

        """
        task_losses:
            {
                "valence": loss_fn,
                "arousal": loss_fn,
                "activity": loss_fn
            }

        lambda_cfg:
            {
                "task": 1.0,
                "contrastive": 0.5,
                "graph": 0.1
            }
        """

        self.task_losses = task_losses
        self.contrastive_loss = contrastive_loss
        self.graph_loss = graph_loss
        self.reliability_loss = reliability_loss

        self.lambda_cfg = lambda_cfg

    def compute(self, model_out: dict, batch: dict):

        total_loss = 0.0
        logs = {}

        # =====================================================
        # 1. TASK LOSS (multi-head support)
        # =====================================================
        task_loss = 0.0

        preds = model_out["pred"]
        targets = batch["targets"]

        for task_name, pred in preds.items():

            if task_name not in self.task_losses:
                continue

            loss_fn = self.task_losses[task_name]

            loss = loss_fn(pred.squeeze(-1), targets[task_name])

            task_loss += loss

            # logs[f"loss_{task_name}"] = loss.item()

        total_loss += self.lambda_cfg["task"] * task_loss
        logs["task_loss"] = task_loss.item()

        # =====================================================
        # 2. CONTRASTIVE LOSS
        # =====================================================
        try:
            contrastive_loss = self.contrastive_loss(
                model_out["graph_emb"],
                batch["graph_emb_pos"]
            )

            try:
                logs["contrastive_loss"] = contrastive_loss.item()
            except:
                logs["contrastive_loss"] = contrastive_loss

        except:
            contrastive_loss = 0.0
        total_loss += self.lambda_cfg["contrastive"] * contrastive_loss

        # =====================================================
        # 3. GRAPH LOSS (selector structure regularization)
        # =====================================================
        graph_loss = self.graph_loss(
            model_out["fused"],
            model_out["scores"]
        )

        total_loss += self.lambda_cfg["graph"] * graph_loss

        try:
            logs["graph_loss"] = graph_loss.item()
        except:
            logs["graph_loss"] = graph_loss

        # =====================================================
        # 4. RELIABILITY LOSS
        # =====================================================

        reliability_loss = self.reliability_loss(model_out["reliability_score"], batch["signal_quality"])
        total_loss += self.lambda_cfg["reliability"] * reliability_loss

        try:
            logs["reliability_loss"] = reliability_loss.item()
        except:
            logs["reliability_loss"] = reliability_loss

        # =====================================================
        # FINAL
        # =====================================================
        logs["total_loss"] = total_loss.item()

        return total_loss, logs