import os
import argparse

import torch
from torch.utils.data import DataLoader

from build.model_builder import build_model
from build.loss_builder import build_losses
from losses.loss_router import LossRouter
from training.trainer import Trainer
from data.dataset import MultimodalDataset
from data.split import get_loso_splits, get_group_kfold_splits, get_subject_dependent_splits
from data.collate import collate_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau

DATA_PATH = r"C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/data_preprocessed_python/data_preprocessed_python"
VIDEO_PATH = r"C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/face_video"


def get_subjects(data_path):
    return [
        f.split(".")[0]
        for f in os.listdir(data_path)
        if f.endswith(".avi")
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Contrastive Selector based Cross-Modal fusion with customizable splits."
    )

    parser.add_argument(
        "split_type",
        nargs="?",
        choices=["loso", "kfold", "subject"],
        help="Type of data split strategy to use: 'loso', 'kfold', 'subject'"
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to use if 'kfold' is selected (default: 5)"
    )

    args = parser.parse_args()

    if not args.split_type:
        print("\n--- Data Split Selection ---")
        print("1. Leave-One-Subject-Out (loso)")
        print("2. Group K-Fold (kfold)")
        print("3. Subject-dependent Splitting")

        while True:
            choice = input("Select split type (1, 2, 3, loso, kfold, subject): ").strip().lower()
            if choice in ["1", "loso"]:
                args.split_type = "loso"
                break
            elif choice in ["2", "kfold"]:
                args.split_type = "kfold"
                break
            elif choice in ["3", "subject"]:
                args.split_type = "subject"
                break
            print("Invalid selection. Please try again.")

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = [f"s{subj:02d}" for subj in range(1, 23)]

    if args.split_type == "loso":
        splits = get_loso_splits(subjects)
    elif args.split_type == "kfold":
        splits = get_group_kfold_splits(subjects, n_splits=args.folds)
    elif args.split_type == "subject":
        splits = get_subject_dependent_splits()

    if args.split_type == "subject":
        for subject in subjects:
            for fold, (train_trials, test_trials) in enumerate(splits):
                print(f"\n===== SUBJECT {subject} | FOLD {fold} =====")
                log_pth = f"logs/subject_{subject}_fold{fold}.json"
                train_dataset = MultimodalDataset(
                    data_path=DATA_PATH,
                    video_path=VIDEO_PATH,
                    subject_list=[subject],
                    trial_indices=train_trials
                )

                test_dataset = MultimodalDataset(
                    data_path=DATA_PATH,
                    video_path=VIDEO_PATH,
                    subject_list=[subject],
                    trial_indices=test_trials
                )

                model, cfg = build_model("configs/config.yaml")
                model = model.to(device)

                task_losses, contrastive_loss, graph_loss, rel_loss, loss_cfg = build_losses(cfg)

                loss_router = LossRouter(
                    task_losses=task_losses,
                    contrastive_loss=contrastive_loss,
                    graph_loss=graph_loss,
                    reliability_loss=rel_loss,
                    lambda_cfg=loss_cfg
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=cfg["training"]["batch_size"],
                    shuffle=True,
                    collate_fn=collate_fn,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=cfg["training"]["batch_size"],
                    shuffle=False,
                    collate_fn=collate_fn,
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg["training"]["lr"],
                    weight_decay=cfg["training"]["weight_decay"]
                )

                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.1,
                    patience=2
                )

                trainer = Trainer(
                    model=model,
                    optimizer=optimizer,
                    loss_router=loss_router,
                    train_loader=train_loader,
                    val_loader=test_loader,
                    scheduler=scheduler,
                    device="cuda"
                )

                trainer.fit(cfg["training"]["epochs"], log_pth=log_pth)
    else:
        for fold, (train_subs, test_subs) in enumerate(splits):
            print(f"\n===== FOLD {fold} =====")
            log_pth = "logs/" + args.split_type + f"_fold{fold}.json"

            model, cfg = build_model("configs/config.yaml")
            model = model.to(device)

            task_losses, contrastive_loss, graph_loss, reliability_loss, loss_cfg = build_losses(cfg)

            loss_router = LossRouter(
                task_losses=task_losses,
                contrastive_loss=contrastive_loss,
                graph_loss=graph_loss,
                reliability_loss=reliability_loss,
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
                shuffle=True,
                collate_fn=collate_fn,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg["training"]["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"]["weight_decay"]
            )

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=2
            )

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_router=loss_router,
                train_loader=train_loader,
                val_loader=test_loader,
                scheduler=scheduler,
                device="cuda"
            )

            trainer.fit(cfg["training"]["epochs"], log_pth=log_pth)


if __name__ == "__main__":
    main()