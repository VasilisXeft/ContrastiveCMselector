import os

import torch
from torch.utils.data import DataLoader

from build.model_builder import build_model
from build.loss_builder import build_losses
from losses.loss_router import LossRouter
from training.trainer import Trainer
from data.dataset import MultimodalDataset
from data.split import get_loso_splits

DATA_PATH = r"C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/data_preprocessed_python/data_preprocessed_python"

VIDEO_PATH = r"C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/face_video"

def get_subjects(data_path):
    return [
        f.split(".")[0]
        for f in os.listdir(data_path)
        if f.endswith(".avi")
    ]

def main():

    subjects = [f"s{subj:02d}" for subj in range(1, 23)]

    splits = get_loso_splits(subjects)

    for fold, (train_subs, test_subs) in enumerate(splits):

        print(f"\n===== FOLD {fold} =====")

        model, cfg = build_model("configs/config.yaml")

        task_losses, contrastive_loss, graph_loss, loss_cfg = build_losses(cfg)

        loss_router = LossRouter(
            task_losses=task_losses,
            contrastive_loss=contrastive_loss,
            graph_loss=graph_loss,
            lambda_cfg=loss_cfg
        )

        train_dataset = MultimodalDataset(
            data_path=DATA_PATH,
            video_path=VIDEO_PATH,
            subject_list=train_subs
        )

        test_dataset = MultimodalDataset(
            data_path=DATA_PATH,
            video_path=VIDEO_PATH,
            subject_list=test_subs
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False
        )

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
            val_loader=test_loader,
            device="cuda"
        )

        trainer.fit(cfg["training"]["epochs"])


if __name__ == "__main__":
    main()