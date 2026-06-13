import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch

from training.train_step import train_step, move_batch_to_device


def aggregate(logs_list):

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


def compute_epoch_metrics(all_preds, all_targets):

    metrics = {}

    for task in all_preds.keys():
        y_pred = torch.cat(all_preds[task]).cpu().numpy()
        y_true = torch.cat(all_targets[task]).cpu().numpy()

        metrics[f"{task}_acc"] = accuracy_score(y_true, y_pred)
        metrics[f"{task}_f1"] = f1_score(y_true, y_pred)

    return metrics


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_router,
        train_loader,
        val_loader=None,
        scheduler=None,
        device="cuda"
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_router = loss_router
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "lr": []
        }

        self.model.to(self.device)

    def save_history(self, path="training_log.json"):
        import json

        def to_serializable(x):
            if isinstance(x, dict):
                return {k: to_serializable(v) for k, v in x.items()}
            if isinstance(x, list):
                return [to_serializable(v) for v in x]
            try:
                return float(x)
            except:
                return str(x)

        with open(path, "w") as f:
            json.dump(to_serializable(self.history), f, indent=2)

    def fit(self, epochs, log_pth=None):

        val_loss_val = None

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_preds, train_targets = self.train_epoch()
            train_metrics = compute_epoch_metrics(train_preds, train_targets)

            print(
                f"Train | "
                f"loss={train_loss['total_loss']:.4f} | "
                f"task={train_loss['task_loss']:.4f} | "
                f"graph={train_loss['graph_loss']:.4f} | "
                f"rel={train_loss['reliability_loss']:.4f} | "
                f"acc_val={train_metrics['valence_acc']:.4f} | "
                f"f1_val={train_metrics['valence_f1']:.4f} | "
                f"acc_aro={train_metrics['arousal_acc']:.4f} | "
                f"f1_aro={train_metrics['arousal_f1']:.4f} | "
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_metrics"].append(train_metrics)

            if self.val_loader is not None:
                val_loss, val_preds, val_targets = self.validate()
                val_metrics = compute_epoch_metrics(val_preds, val_targets)

                print(
                    f"Val   | "
                    f"loss={val_loss['total_loss']:.4f} | "
                    f"task={val_loss['task_loss']:.4f} | "
                    f"graph={val_loss['graph_loss']:.4f} | "
                    f"rel={val_loss['reliability_loss']:.4f} | "
                    f"acc_val={val_metrics['valence_acc']:.4f} | "
                    f"f1_val={val_metrics['valence_f1']:.4f} | "
                    f"acc_aro={val_metrics['arousal_acc']:.4f} | "
                    f"f1_aro={val_metrics['arousal_f1']:.4f} | "
                    f"lr={self.optimizer.param_groups[0]['lr']}"
                )

                self.history["val_loss"].append(val_loss)
                self.history["val_metrics"].append(val_metrics)

                if isinstance(val_loss, dict) and "val_loss" in val_loss:
                    val_loss_val = val_loss["val_loss"]
                elif isinstance(val_loss, dict):
                    val_loss_val = next(iter(val_loss.values()))

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss_val is not None:
                        self.scheduler.step(val_loss_val)
                    else:
                        train_loss_val = next(iter(train_loss.values())) if isinstance(train_loss, dict) else train_loss
                        self.scheduler.step(train_loss_val)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["lr"].append(current_lr)

        self.save_history(log_pth)


    def train_epoch(self):

        self.model.train()

        epoch_logs = []

        all_preds = {"valence":[], "arousal": []}
        all_targets = {"valence":[], "arousal": []}

        for batch in tqdm(self.train_loader, leave=False):
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

        return aggregate(epoch_logs), all_preds, all_targets

    def validate(self):

        self.model.eval()

        epoch_logs = []

        all_preds = {"valence":[], "arousal": []}
        all_targets = {"valence":[], "arousal": []}

        with torch.no_grad():

            for batch in self.val_loader:

                batch = move_batch_to_device(batch, self.device)

                outputs = self.model(batch)

                loss, logs = self.loss_router.compute(outputs, batch)

                epoch_logs.append(logs)

                for task in all_preds.keys():
                    logits = outputs["pred"][task].squeeze(-1)  # [B]

                    pred_labels = (torch.sigmoid(logits) > 0.5).detach().cpu().long()

                    all_preds[task].append(pred_labels)
                    all_targets[task].append(batch["targets"][task].detach().cpu().view(-1))

        return aggregate(epoch_logs), all_preds, all_targets

