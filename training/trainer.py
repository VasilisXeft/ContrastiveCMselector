import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch

from training.train_step import train_step, move_batch_to_device


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_router,
        train_loader,
        val_loader=None,
        device="cuda"
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_router = loss_router

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.model.to(self.device)

    def fit(self, epochs):

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_preds, train_targets = self.train_epoch()
            train_metrics = self.compute_epoch_metrics(train_preds, train_targets)

            print("Train:", train_loss, train_metrics)

            if self.val_loader is not None:
                val_loss, val_preds, val_targets = self.validate()
                val_metrics = self.compute_epoch_metrics(val_preds, val_targets)

                print("Val:", val_loss, val_metrics)


    def train_epoch(self):

        self.model.train()

        epoch_logs = []

        all_preds = {"valence": [], "arousal": []}
        all_targets = {"valence": [], "arousal": []}

        for batch in tqdm(self.train_loader):
            if batch is None:
                continue

            logs, preds, targets = train_step(
                batch,
                self.model,
                self.optimizer,
                self.loss_router,
                self.device
            )

            epoch_logs.append(logs)

            # collect predictions
            for task in all_preds.keys():
                logits = preds[task].squeeze(-1)  # [B]

                pred_labels = (torch.sigmoid(logits) > 0.5).detach().cpu().long()

                all_preds[task].append(pred_labels)
                all_targets[task].append(batch["targets"][task].detach().cpu().view(-1))

        return self.aggregate(epoch_logs), all_preds, all_targets

    def validate(self):

        self.model.eval()

        epoch_logs = []

        all_preds = {"valence": [], "arousal": []}
        all_targets = {"valence": [], "arousal": []}

        with torch.no_grad():

            for batch in self.val_loader:

                batch = move_batch_to_device(batch, self.device)

                outputs = self.model(batch)

                loss, logs = self.loss_router.compute(outputs, batch)

                logs["val_loss"] = loss.item()

                epoch_logs.append(logs)

                for task in all_preds.keys():
                    logits = outputs["pred"][task].squeeze(-1)  # [B]

                    pred_labels = (torch.sigmoid(logits) > 0.5).detach().cpu().long()

                    all_preds[task].append(pred_labels)
                    all_targets[task].append(batch["targets"][task].detach().cpu().view(-1))

        return self.aggregate(epoch_logs), all_preds, all_targets

    def compute_epoch_metrics(self, all_preds, all_targets):

        metrics = {}

        for task in all_preds.keys():
            y_pred = torch.cat(all_preds[task]).cpu().numpy()
            y_true = torch.cat(all_targets[task]).cpu().numpy()

            metrics[f"{task}_bal_acc"] = balanced_accuracy_score(y_true, y_pred)
            metrics[f"{task}_f1_score"] = f1_score(y_true, y_pred)

        return metrics

    def aggregate(self, logs_list):

        agg = {}

        for logs in logs_list:

            for k, v in logs.items():

                if k not in agg:
                    agg[k] = []

                agg[k].append(v)

        return {
            k: sum(v) / len(v)
            for k, v in agg.items()
        }