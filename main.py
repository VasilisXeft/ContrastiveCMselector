import os
import glob
import argparse
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from build.model_builder import build_model
from build.loss_builder import build_losses
from losses.loss_router import LossRouter
from training.trainer import Trainer

from data.dataset import (
    MultimodalDataset,
    MAHNOBMultimodalDataset
)

from data.split import get_loso_splits
from data.collate import collate_fn_deap, collate_fn_mahnob


DEAP_DATA_PATH = (
    r"C:/Users/vxefteris/Desktop/D/MindSpaces/"
    r"DEAP Dataset/data_preprocessed_python/data_preprocessed_python"
)

DEAP_VIDEO_PATH = (
    r"C:/Users/vxefteris/Desktop/D/MindSpaces/"
    r"DEAP Dataset/face_video"
)

MAHNOB_PATH = (
    r"C:/Users/vxefteris/Desktop/D/SUN/Data/"
    r"MAHNOB_HCI_Tagging/Sessions"
)


def get_deap_subjects():
    return [f"s{subj:02d}" for subj in range(1, 23)]


def get_mahnob_subjects(base_path):
    subjects = set()

    xml_files = glob.glob(
        os.path.join(base_path, "*", "session.xml")
    )

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()

            sid = int(
                root.find("subject").attrib["id"]
            )

            subjects.add(sid)

        except Exception:
            continue

    return sorted(list(subjects))


def parse_args():

    parser = argparse.ArgumentParser(
        description="LOSO training for DEAP or MAHNOB."
    )

    parser.add_argument(
        "--dataset",
        choices=["deap", "mahnob"],
        default="deap",
        help="Dataset to use."
    )

    return parser.parse_args()


def build_datasets(dataset_name,
                   train_subjects,
                   test_subjects):

    if dataset_name == "deap":

        train_dataset = MultimodalDataset(
            data_path=DEAP_DATA_PATH,
            video_path=DEAP_VIDEO_PATH,
            subject_list=train_subjects
        )

        test_dataset = MultimodalDataset(
            data_path=DEAP_DATA_PATH,
            video_path=DEAP_VIDEO_PATH,
            subject_list=test_subjects
        )

    elif dataset_name == "mahnob":

        train_dataset = MAHNOBMultimodalDataset(
            base_path=MAHNOB_PATH,
            subjects_to_keep=train_subjects
        )

        test_dataset = MAHNOBMultimodalDataset(
            base_path=MAHNOB_PATH,
            subjects_to_keep=test_subjects
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    return train_dataset, test_dataset


def calculate_pos_weights(dataloader, device):
    """
    Υπολογίζει δυναμικά το pos_weight για Valence και Arousal
    από τα δεδομένα του τρέχοντος Dataloader.
    """
    val_pos, val_neg = 0, 0
    ars_pos, ars_neg = 0, 0

    for batch in dataloader:
        val_labels = batch['targets']['valence']
        ars_labels = batch['targets']['arousal']

        val_pos += (val_labels == 1).sum().item()
        val_neg += (val_labels == 0).sum().item()

        ars_pos += (ars_labels == 1).sum().item()
        ars_neg += (ars_labels == 0).sum().item()

    # Προστασία από διαίρεση με το μηδέν
    val_weight = val_neg / (val_pos + 1e-5)
    ars_weight = ars_neg / (ars_pos + 1e-5)

    print(f"📊 Class Weights - Valence: {val_weight:.2f}, Arousal: {ars_weight:.2f}")

    return {
        "valence": torch.tensor([val_weight], dtype=torch.float).to(device),
        "arousal": torch.tensor([ars_weight], dtype=torch.float).to(device)
    }

def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing dataset: {args.dataset}")
    print(f"Device: {device}")

    ###################################
    # SUBJECTS
    ###################################

    if args.dataset == "deap":

        subjects = get_deap_subjects()

    else:

        subjects = get_mahnob_subjects(
            MAHNOB_PATH
        )

    ###################################
    # LOSO SPLITS
    ###################################

    splits = get_loso_splits(subjects)

    ###################################
    # TRAINING
    ###################################

    for fold, (train_subs, test_subs) in enumerate(splits):

        print("\n" + "=" * 50)
        print(f"FOLD {fold + 1}/{len(splits)}")
        print(f"Train: {train_subs}")
        print(f"Test : {test_subs}")
        print("=" * 50)

        log_pth = (
            f"logs/"
            f"{args.dataset}_loso_fold{fold}.json"
        )

        ###################################
        # MODEL
        ###################################

        model, cfg = build_model(
            "configs/config.yaml"
        )

        model = model.to(device)




        ###################################
        # DATASETS
        ###################################

        train_dataset, test_dataset = build_datasets(
            args.dataset,
            train_subs,
            test_subs
        )

        ###################################
        # LOADERS
        ###################################

        if args.dataset == "deap":
            collate_fn = collate_fn_deap
        else:
            collate_fn = collate_fn_mahnob

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn
        )

        ###################################
        # OPTIMIZER
        ###################################

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"]
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.05,
            patience=3
        )

        ###################################
        # LOSSES
        ###################################

        pos_weights = calculate_pos_weights(train_loader, device)

        (
            task_losses,
            contrastive_loss,
            graph_loss,
            reliability_loss,
            loss_cfg
        ) = build_losses(cfg, pos_weights)

        loss_router = LossRouter(
            task_losses=task_losses,
            contrastive_loss=contrastive_loss,
            graph_loss=graph_loss,
            reliability_loss=reliability_loss,
            lambda_cfg=loss_cfg
        )

        ###################################
        # TRAINER
        ###################################

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_router=loss_router,
            train_loader=train_loader,
            val_loader=test_loader,
            scheduler=scheduler,
            device=device
        )

        trainer.fit(
            cfg["training"]["epochs"],
            log_pth=log_pth
        )


if __name__ == "__main__":
    main()