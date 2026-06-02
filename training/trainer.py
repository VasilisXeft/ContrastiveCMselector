import torch
from tqdm import tqdm


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

    def fit(self, epochs):

        self.model.to(self.device)

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1}/{epochs}")

            train_logs = self.train_epoch()

            print("Train:", train_logs)

            if self.val_loader is not None:

                val_logs = self.validate()

                print("Val:", val_logs)

    def train_epoch(self):

        epoch_logs = []

        for batch in tqdm(self.train_loader):

            logs = train_step(
                batch,
                self.model,
                self.optimizer,
                self.loss_router,
                self.device
            )

            epoch_logs.append(logs)

        return self.aggregate(epoch_logs)

    def validate(self):

        self.model.eval()

        epoch_logs = []

        with torch.no_grad():

            for batch in self.val_loader:

                batch = move_batch_to_device(batch, self.device)

                outputs = self.model(batch)

                loss, logs = self.loss_router.compute(outputs, batch)

                logs["val_loss"] = loss.item()

                epoch_logs.append(logs)

        return self.aggregate(epoch_logs)

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