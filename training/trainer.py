import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import json

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


# 🌟 1. Dynamic Metric Calculation
def compute_epoch_metrics(all_preds, all_targets, tasks):
    metrics = {}
    for task in tasks:
        # Check if we have predictions for this task (protect against empty sets)
        if len(all_preds[task]) > 0 and len(all_targets[task]) > 0:
            y_pred = torch.cat(all_preds[task]).cpu().numpy()
            y_true = torch.cat(all_targets[task]).cpu().numpy()

            metrics[f"{task}_acc"] = accuracy_score(y_true, y_pred)
            metrics[f"{task}_f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            metrics[f"{task}_acc"] = 0.0
            metrics[f"{task}_f1"] = 0.0

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
            device="cuda",
            tasks=["valence", "arousal"]  # 🌟 2. Accept tasks dynamically
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_router = loss_router
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.tasks = tasks

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "lr": []
        }
        self.model.to(self.device)

    def save_history(self, path="training_log.json"):
        def to_serializable(x):
            if isinstance(x, dict):
                return {k: to_serializable(v) for k, v in x.items()}
            if isinstance(x, list):
                return [to_serializable(v) for v in x]
            try:
                return float(x)
            except:
                return str(x)

        if path:
            with open(path, "w") as f:
                json.dump(to_serializable(self.history), f, indent=2)

    def fit(self, epochs, log_pth=None, save_path="best_model.pth", patience=12):
        best_val_f1 = 0.0
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # --- TRAIN ---
            train_loss, train_preds, train_targets = self.train_epoch()
            train_metrics = compute_epoch_metrics(train_preds, train_targets, self.tasks)

            # 3. Dynamic Printing for Train
            train_print = (
                f"Train | "
                f"loss={train_loss.get('total_loss', 0):.4f} | "
                f"task={train_loss.get('task_loss', 0):.4f} | "
                f"graph={train_loss.get('graph_loss', 0):.4f} | "
                f"rel={train_loss.get('reliability_loss', 0):.4f} | "
            )
            for t in self.tasks:
                train_print += f"acc_{t[:3]}={train_metrics[f'{t}_acc']:.4f} | f1_{t[:3]}={train_metrics[f'{t}_f1']:.4f} | "
            print(train_print)

            self.history["train_loss"].append(train_loss)
            self.history["train_metrics"].append(train_metrics)

            # --- VALIDATION ---
            if self.val_loader is not None:
                val_loss, val_preds, val_targets = self.validate()
                val_metrics = compute_epoch_metrics(val_preds, val_targets, self.tasks)

                # 4. Dynamic Printing for Val
                val_print = (
                    f"Val   | "
                    f"loss={val_loss.get('total_loss', 0):.4f} | "
                    f"task={val_loss.get('task_loss', 0):.4f} | "
                    f"graph={val_loss.get('graph_loss', 0):.4f} | "
                    f"rel={val_loss.get('reliability_loss', 0):.4f} | "
                )
                for t in self.tasks:
                    val_print += f"acc_{t[:3]}={val_metrics[f'{t}_acc']:.4f} | f1_{t[:3]}={val_metrics[f'{t}_f1']:.4f} | "
                val_print += f"lr={self.optimizer.param_groups[0]['lr']}"
                print(val_print)

                self.history["val_loss"].append(val_loss)
                self.history["val_metrics"].append(val_metrics)

                # 5. Dynamic Mean F1 for Early Stopping
                mean_f1 = sum([val_metrics[f"{t}_acc"] for t in self.tasks]) / len(self.tasks)

                if mean_f1 > best_val_f1:
                    best_val_f1 = mean_f1
                    best_epoch = epoch + 1
                    epochs_no_improve = 0

                    torch.save(self.model.state_dict(), save_path)
                    print(f"New Record Mean Acc: {best_val_f1:.4f}! Saved to '{save_path}'")
                else:
                    epochs_no_improve += 1
                    print(
                        f"No Acc improvement for {epochs_no_improve} epochs (Best: {best_val_f1:.4f} @ Epoch {best_epoch})")

                val_loss_val = val_loss.get("total_loss", next(iter(val_loss.values())))
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss_val)
                    else:
                        self.scheduler.step()

                if epochs_no_improve >= patience:
                    print(f"\nEARLY STOPPING triggered! No improvement for {patience} consecutive epochs.")
                    break

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["lr"].append(current_lr)

        print(f"Finished! Best model was at Epoch {best_epoch} with Mean Val Acc: {best_val_f1:.4f}")
        self.save_history(log_pth)

    def train_epoch(self):
        self.model.train()
        epoch_logs = []

        # 🌟 6. Dynamic Dict Initialization
        all_preds = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}

        for batch in tqdm(self.train_loader, leave=False):
            if batch is None or len(batch) == 0 or "targets" not in batch:
                continue

            try:
                logs, preds, targets = train_step(
                    batch, self.model, self.optimizer, self.loss_router, self.device
                )
                epoch_logs.append(logs)

                for task in self.tasks:
                    if task in preds and task in batch["targets"]:
                        logits = preds[task].squeeze(-1)
                        pred_labels = (torch.sigmoid(logits) > 0.5).detach().cpu().long()
                        all_preds[task].append(pred_labels)
                        all_targets[task].append(batch["targets"][task].detach().cpu().view(-1))
            except RuntimeError as e:
                if "stack expects a non-empty TensorList" in str(e):
                    continue
                raise e

        return aggregate(epoch_logs), all_preds, all_targets

    def validate(self):
        self.model.eval()
        epoch_logs = []

        # 7. Dynamic Dict Initialization
        all_preds = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None or len(batch) == 0 or "targets" not in batch:
                    continue

                batch = move_batch_to_device(batch, self.device)

                try:
                    outputs = self.model(batch)
                except RuntimeError as e:
                    if "stack expects a non-empty TensorList" in str(e):
                        continue
                    raise e

                loss, logs = self.loss_router.compute(outputs, batch)
                epoch_logs.append(logs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                for task in self.tasks:
                    if task in outputs.get("pred", {}) and task in batch["targets"]:
                        logits = outputs["pred"][task].squeeze(-1)
                        pred_labels = (torch.sigmoid(logits) > 0.5).detach().cpu().long()
                        all_preds[task].append(pred_labels)
                        all_targets[task].append(batch["targets"][task].detach().cpu().view(-1))

        return aggregate(epoch_logs), all_preds, all_targets