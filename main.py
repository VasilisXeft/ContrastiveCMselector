import os
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np

import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

from build.model_builder import build_model
from build.loss_builder import build_losses
from losses.loss_router import LossRouter

from models.encoders import EEGNetEncoder, MultiScalePhysioEncoder, PhysioEncoder, ECGEncoder
from training.trainer import Trainer
from training.pretrainer import pretrain_encoder

from data.dataset import (
    MultimodalDataset,
    MAHNOBMultimodalDataset
)

from data.split import get_loso_splits
from data.collate import collate_fn_deap, collate_fn_mahnob

DEAP_DATA_PATH = r"C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/data_preprocessed_python/data_preprocessed_python"
DEAP_VIDEO_PATH = r"C:/Users/vxefteris/Desktop/D/MindSpaces/DEAP Dataset/face_video"
MAHNOB_PATH = r"C:/Users/vxefteris/Desktop/D/SUN/Data/MAHNOB_HCI_Tagging/Sessions"


def get_deap_subjects():
    return [f"s{subj:02d}" for subj in range(1, 23)]


def get_mahnob_subjects(base_path):
    subjects = set()
    xml_files = glob.glob(os.path.join(base_path, "*", "session.xml"))
    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
            sid = int(root.find("subject").attrib["id"])
            subjects.add(sid)
        except Exception:
            continue
    return sorted(list(subjects))


def parse_args():
    parser = argparse.ArgumentParser(description="LOSO training for DEAP or MAHNOB.")
    parser.add_argument("--dataset", choices=["deap", "mahnob"], default="deap", help="Dataset to use.")
    parser.add_argument("--do_pretrain", choices=[True, False], default=True, help="Pretrain.")
    return parser.parse_args()


def build_datasets(dataset_name, train_subjects, val_subjects, test_subjects):
    if dataset_name == "deap":
        train_dataset = MultimodalDataset(data_path=DEAP_DATA_PATH, video_path=DEAP_VIDEO_PATH,
                                          subject_list=train_subjects)
        val_dataset = MultimodalDataset(data_path=DEAP_DATA_PATH, video_path=DEAP_VIDEO_PATH, subject_list=val_subjects)
        test_dataset = MultimodalDataset(data_path=DEAP_DATA_PATH, video_path=DEAP_VIDEO_PATH,
                                         subject_list=test_subjects)
    elif dataset_name == "mahnob":
        train_dataset = MAHNOBMultimodalDataset(base_path=MAHNOB_PATH, subjects_to_keep=train_subjects)
        val_dataset = MAHNOBMultimodalDataset(base_path=MAHNOB_PATH, subjects_to_keep=val_subjects)
        test_dataset = MAHNOBMultimodalDataset(base_path=MAHNOB_PATH, subjects_to_keep=test_subjects)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, val_dataset, test_dataset


def calculate_pos_weights(dataloader, device):
    val_pos, val_neg = 0, 0
    ars_pos, ars_neg = 0, 0
    for batch in dataloader:
        val_labels = batch['targets']['valence']
        ars_labels = batch['targets']['arousal']
        val_pos += (val_labels == 1).sum().item()
        val_neg += (val_labels == 0).sum().item()
        ars_pos += (ars_labels == 1).sum().item()
        ars_neg += (ars_labels == 0).sum().item()
    val_weight = val_neg / (val_pos + 1e-5)
    ars_weight = ars_neg / (ars_pos + 1e-5)
    print(f"📊 Class Weights - Valence: {val_weight:.2f}, Arousal: {ars_weight:.2f}")
    return {
        "valence": torch.tensor([val_weight], dtype=torch.float).to(device),
        "arousal": torch.tensor([ars_weight], dtype=torch.float).to(device)
    }


def evaluate_on_test(model, test_loader, device, tasks):
    model.eval()

    all_preds = {task: [] for task in tasks}
    all_targs = {task: [] for task in tasks}

    valid_batches_counted = 0
    from training.train_step import move_batch_to_device

    with torch.no_grad():
        for batch in test_loader:

            if batch is None or len(batch) == 0 or "targets" not in batch:
                continue

            batch = move_batch_to_device(batch, device)

            try:
                outputs = model(batch)
            except RuntimeError as e:
                if "stack expects a non-empty TensorList" in str(e):
                    continue
                raise e

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            for task in tasks:
                if task in batch["targets"] and task in outputs["pred"]:
                    targets = batch["targets"][task].detach().cpu().view(-1)
                    logits = outputs["pred"][task].squeeze(-1)

                    preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()

                    all_preds[task].extend(preds)
                    all_targs[task].extend(targets.numpy())

            valid_batches_counted += 1

    if valid_batches_counted == 0:
        print("No valid biosignals for user. Return Acc: 0.0")
        return {task: 0.0 for task in tasks}

    f1_scores = {}
    for task in tasks:
        if len(all_targs[task]) > 0:
            f1_scores[task] = accuracy_score(all_targs[task], all_preds[task])
        else:
            f1_scores[task] = 0.0

    return f1_scores


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing dataset: {args.dataset}")
    print(f"Device: {device}")

    subjects = get_deap_subjects() if args.dataset == "deap" else get_mahnob_subjects(MAHNOB_PATH)

    splits = get_loso_splits(subjects, val_ratio=0.15)

    all_test_results = []

    for fold, (train_subs, val_subs, test_subs) in enumerate(splits):
        print("\n" + "=" * 60)
        print(f"FOLD {fold + 1}/{len(splits)}")
        print(f"Train : {train_subs}")
        print(f"Val   : {val_subs}")
        print(f"Test  : {test_subs}")
        print("=" * 60)

        log_pth = f"logs/{args.dataset}_loso_fold{fold}.json"

        with open("configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        active_tasks = cfg["model"]["task_head"]["outputs"]

        train_dataset, val_dataset, test_dataset = build_datasets(
            args.dataset, train_subs, val_subs, test_subs
        )

        collate_fn = collate_fn_deap if args.dataset == "deap" else collate_fn_mahnob

        train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True,
                                  collate_fn=collate_fn)

        val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
                                collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
                                 collate_fn=collate_fn)

        # =========================
        # PRETRAIN
        # =========================
        if args.do_pretrain:
            if args.dataset == "deap":
                encoders_dict = {
                    "eeg": EEGNetEncoder(emb_dim=64),
                    "ppg": MultiScalePhysioEncoder(emb_dim=64, in_ch=1),
                    "eda": MultiScalePhysioEncoder(emb_dim=64, in_ch=1),
                    "tmp": MultiScalePhysioEncoder(emb_dim=64, in_ch=1)
                }
                pretrain_config = {
                    "ppg": {"mode": "supervised", "lr": 1e-4, "epochs": 25},
                    "eda": {"mode": "supervised", "lr": 1e-4, "epochs": 25},
                    "tmp": {"mode": "supervised", "lr": 1e-4, "epochs": 25},
                    "eeg": {"mode": "supervised", "lr": 1e-3, "epochs": 25},
                }
            else:
                encoders_dict = {
                    "eeg": EEGNetEncoder(emb_dim=64),
                    "ecg": ECGEncoder(emb_dim=64, in_ch=3),
                    "eda": MultiScalePhysioEncoder(emb_dim=64, in_ch=1),
                    "tmp": MultiScalePhysioEncoder(emb_dim=64, in_ch=1),
                    "rsp": MultiScalePhysioEncoder(emb_dim=64, in_ch=1),
                    "eye": PhysioEncoder(emb_dim=64, in_ch=3)
                }
                pretrain_config = {
                    "ecg": {"mode": "supervised", "lr": 1e-4, "epochs": 25},
                    "eeg": {"mode": "supervised", "lr": 1e-3, "epochs": 25},
                    "eda": {"mode": "supervised", "lr": 1e-4, "epochs": 25},
                    "tmp": {"mode": "supervised", "lr": 1e-4, "epochs": 25},
                    "rsp": {"mode": "supervised", "lr": 1e-4, "epochs": 25},
                    "eye": {"mode": "supervised", "lr": 1e-3, "epochs": 25},
                }

            paths_for_this_fold = {}
            for mod, config in pretrain_config.items():
                best_path = pretrain_encoder(
                    modality_name=mod, encoder=encoders_dict[mod], train_loader=train_loader,
                    val_loader=val_loader, fold=fold, device=device, mode=config["mode"],
                    lr=config["lr"], epochs=config["epochs"],
                    tasks=active_tasks
                )
                paths_for_this_fold[mod] = best_path
                encoders_dict[mod].load_state_dict(torch.load(best_path))

        # =========================
        # MODEL FUSION
        # =========================
        model, _ = build_model("configs/config.yaml", pretrained_weights=paths_for_this_fold, freeze_encoders=True)
        model = model.to(device)


        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-5)

        pos_weights = calculate_pos_weights(train_loader, device)
        task_losses, contrastive_loss, graph_loss, reliability_loss, loss_cfg = build_losses(cfg, pos_weights)
        loss_router = LossRouter(task_losses=task_losses, contrastive_loss=contrastive_loss, graph_loss=graph_loss,
                                 reliability_loss=reliability_loss, lambda_cfg=loss_cfg)

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_router=loss_router,
            train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, device=device,
            tasks=active_tasks
        )

        best_fusion_path = f"pretrained_weights/best_fusion_fold{fold}.pth"

        trainer.fit(
            cfg["training"]["epochs"],
            log_pth=log_pth,
            save_path=best_fusion_path,
            patience=12)

        # =========================
        # FINAL TEST EVAUATION
        # =========================
        print("\nLoading best fusion model!")

        model.load_state_dict(torch.load(best_fusion_path))

        try:
            test_f1_dict = evaluate_on_test(model, test_loader, device, active_tasks)

            res_string = " | ".join([f"{t.capitalize()} F1: {v:.4f}" for t, v in test_f1_dict.items()])
            print(f"Results FOLD {fold + 1} (Subj: {test_subs[0]}): {res_string}")

            result_entry = {"fold": fold + 1, "test_subject": test_subs[0]}
            result_entry.update(test_f1_dict)
            all_test_results.append(result_entry)

        except Exception as e:
            print(f"Test evaluation failed due to error: {e}")

        # =========================
        # FINAL REPORT
        # =========================
        print("\n" + "=" * 30)
        print("      FINAL TEST SET RESULTS      ")
        print("=" * 30)

        scores = {task: [] for task in active_tasks}

        for res in all_test_results:
            res_str = " | ".join([f"{t.capitalize()}: {res[t]:.4f}" for t in active_tasks])
            print(f"Fold {res['fold']:02d} (Subj {res['test_subject']}): {res_str}")
            for t in active_tasks:
                scores[t].append(res[t])

        print("-" * 50)
        for task in active_tasks:
            mean_score = np.mean(scores[task])
            std_score = np.std(scores[task])
            print(f"AVERAGE {task.upper()} ACC: {mean_score:.4f} ± {std_score:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()