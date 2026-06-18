import os
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np

import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

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
    parser.add_argument("--dataset", choices=["deap", "mahnob"], default="mahnob", help="Dataset to use.")
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


def evaluate_on_test(model, test_loader, device):
    model.eval()
    val_preds, val_targs = [], []
    aro_preds, aro_targs = [], []

    from training.train_step import move_batch_to_device

    valid_batches_counted = 0

    with torch.no_grad():
        for batch in test_loader:
            # 1. Προστασία από εντελώς κενά batches που μπορεί να στείλει ο collate_fn
            if batch is None or len(batch) == 0:
                continue
            if "targets" not in batch:
                continue

            batch = move_batch_to_device(batch, device)

            # 2. Προστασία στον Γράφο: Αν λείπουν ΟΛΑ τα modalities, το torch.stack θα σκάσει.
            try:
                outputs = model(batch)
            except RuntimeError as e:
                if "stack expects a non-empty TensorList" in str(e):
                    # Το batch δεν έχει ούτε ένα modality! Το προσπερνάμε αθόρυβα.
                    continue
                else:
                    # Αν είναι άλλο σοβαρό error, το πετάμε κανονικά
                    raise e

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # 3. Εξαγωγή Targets
            targets_val = batch["targets"]["valence"].detach().cpu().view(-1)
            targets_aro = batch["targets"]["arousal"].detach().cpu().view(-1)

            # 4. Εξαγωγή Predictions
            logits_v = outputs["pred"]["valence"].squeeze(-1)
            logits_a = outputs["pred"]["arousal"].squeeze(-1)

            preds_v = (torch.sigmoid(logits_v) > 0.5).int().cpu().numpy()
            preds_a = (torch.sigmoid(logits_a) > 0.5).int().cpu().numpy()

            val_preds.extend(preds_v)
            val_targs.extend(targets_val.numpy())
            aro_preds.extend(preds_a)
            aro_targs.extend(targets_aro.numpy())

            valid_batches_counted += 1

    # 5. Προστασία: Τι γίνεται αν ο χρήστης (π.χ. Subject 1) δεν είχε ΟΥΤΕ ΕΝΑ σωστό παράθυρο;
    if valid_batches_counted == 0 or len(val_preds) == 0:
        print("   ⚠️ Ο συγκεκριμένος χρήστης δεν έχει κανένα έγκυρο βιοσήμα! Επιστροφή F1: 0.0")
        return 0.0, 0.0

    f1_v = f1_score(val_targs, val_preds, average='macro', zero_division=0)
    f1_a = f1_score(aro_targs, aro_preds, average='macro', zero_division=0)

    return f1_v, f1_a


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
        print(f"🚀 FOLD {fold + 1}/{len(splits)}")
        print(f"Train : {train_subs}")
        print(f"Val   : {val_subs}")
        print(f"Test  : {test_subs}")
        print("=" * 60)

        log_pth = f"logs/{args.dataset}_loso_fold{fold}.json"

        with open("configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

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
        # PRETRAIN (ΟΛΑ SUPERVISED!)
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
                    lr=config["lr"], epochs=config["epochs"]
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
            train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, device=device
        )

        best_fusion_path = f"pretrained_weights/best_fusion_fold{fold}.pth"

        trainer.fit(
            cfg["training"]["epochs"],
            log_pth=log_pth,
            save_path=best_fusion_path,   # <--- Στέλνουμε το path!
            patience=12)

        # =========================
        # 🏆 ΤΕΛΙΚΗ ΑΞΙΟΛΟΓΗΣΗ TEST
        # =========================
        print("\n⏳ Φόρτωση του Καλύτερου Μοντέλου για αξιολόγηση στον Άγνωστο Χρήστη (Test Set)...")
        # Ανάλογα πού σώζει ο Trainer σου τα βάρη. Βάλε το σωστό path!
        model.load_state_dict(torch.load(best_fusion_path))

        try:
            test_f1_v, test_f1_a = evaluate_on_test(model, test_loader, device)
            print(
                f"🎯 Αποτελέσματα FOLD {fold + 1} (Subj: {test_subs[0]}): Valence F1: {test_f1_v:.4f} | Arousal F1: {test_f1_a:.4f}")

            all_test_results.append({
                "fold": fold + 1,
                "test_subject": test_subs[0],
                "valence_f1": test_f1_v,
                "arousal_f1": test_f1_a
            })
        except Exception as e:
            print(f"⚠️ Προσοχή: Απέτυχε το Test Evaluation. Σφάλμα: {e}")

    # =========================
    # FINAL REPORT
    # =========================
    print("\n" + "=" * 30)
    print("      TEST SET RESULTS      ")
    print("=" * 30)

    val_scores = []
    aro_scores = []

    for res in all_test_results:
        print(
            f"Fold {res['fold']:02d} (Subj {res['test_subject']}): Valence = {res['valence_f1']:.4f} | Arousal = {res['arousal_f1']:.4f}")
        val_scores.append(res["valence_f1"])
        aro_scores.append(res["arousal_f1"])

    print("-" * 50)
    print(f"AVERAGE VALENCE F1: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    print(f"AVERAGE AROUSAL F1: {np.mean(aro_scores):.4f} ± {np.std(aro_scores):.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()