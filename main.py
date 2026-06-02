import torch
from torch.utils.data import DataLoader

from build.model_builder import build_model
from build.loss_builder import build_losses
from losses.loss_router import LossRouter
from training.trainer import Trainer
from data.dataset import MultimodalDataset


def main():

    model, cfg = build_model("configs/config.yaml")

    task_losses, contrastive_loss, graph_loss, loss_cfg = build_losses(cfg)

    loss_router = LossRouter(
        task_losses=task_losses,
        contrastive_loss=contrastive_loss,
        graph_loss=graph_loss,
        lambda_cfg=loss_cfg
    )

    train_dataset = MultimodalDataset(...)
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_router=loss_router,
        train_loader=train_loader,
        device="cuda"
    )

    trainer.fit(cfg["training"]["epochs"])


if __name__ == "__main__":
    main()