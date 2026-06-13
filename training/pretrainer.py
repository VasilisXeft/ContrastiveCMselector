import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
import os


class UnimodalPretrainer(nn.Module):
    """
    Ένας 'περίβλημα' (wrapper) που παίρνει έναν οποιοδήποτε encoder
    και προσθέτει classifiers για Valence και Arousal.
    """

    def __init__(self, encoder, emb_dim=64):
        super().__init__()
        self.encoder = encoder

        # Απλά MLP για τα tasks
        self.val_head = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.aro_head = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        emb = self.encoder(x)
        val = self.val_head(emb)
        aro = self.aro_head(emb)
        return {"valence": val, "arousal": aro}


def pretrain_single_modality(modality_name, encoder, train_loader, val_loader, fold, device, epochs=15):
    """
    Εκπαιδεύει έναν encoder για ένα συγκεκριμένο modality και αποθηκεύει τα καλύτερα βάρη.
    """
    print(f"\n🚀 Ξεκινάει Unimodal Pretraining για: {modality_name.upper()} (Fold {fold})")

    model = UnimodalPretrainer(encoder, emb_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    save_dir = "pretrained_weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{modality_name}_fold{fold}.pth")

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        # Λίστες για την αποθήκευση των train metrics
        train_preds_val, train_preds_aro = [], []
        train_targs_val, train_targs_aro = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            x = batch[modality_name].to(device)
            targets_val = batch["targets"]["valence"].to(device).float().view(-1)
            targets_aro = batch["targets"]["arousal"].to(device).float().view(-1)

            optimizer.zero_grad()
            outputs = model(x)

            loss_val = criterion(outputs["valence"].view(-1), targets_val)
            loss_aro = criterion(outputs["arousal"].view(-1), targets_aro)
            loss = loss_val + loss_aro

            loss.backward()

            # Προαιρετικά: Κόφτης για ασφάλεια στα σήματα
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # --- Συλλογή δεδομένων για τα Train Metrics ---
            # To .detach() είναι ΑΠΟΛΥΤΩΣ απαραίτητο εδώ για την αποφυγή Memory Leak!
            preds_v = (torch.sigmoid(outputs["valence"].detach()) > 0.5).int().cpu().numpy()
            preds_a = (torch.sigmoid(outputs["arousal"].detach()) > 0.5).int().cpu().numpy()

            train_preds_val.extend(preds_v)
            train_preds_aro.extend(preds_a)
            train_targs_val.extend(targets_val.cpu().numpy())
            train_targs_aro.extend(targets_aro.cpu().numpy())

        # Υπολογισμός Train Metrics
        train_f1_val = f1_score(train_targs_val, train_preds_val, average='macro', zero_division=0)
        train_f1_aro = f1_score(train_targs_aro, train_preds_aro, average='macro', zero_division=0)

        # --- VALIDATION ---
        model.eval()
        val_preds_val, val_preds_aro = [], []
        val_targs_val, val_targs_aro = [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch[modality_name].to(device)
                outputs = model(x)

                preds_v = (torch.sigmoid(outputs["valence"]) > 0.5).int().cpu().numpy()
                preds_a = (torch.sigmoid(outputs["arousal"]) > 0.5).int().cpu().numpy()

                val_preds_val.extend(preds_v)
                val_preds_aro.extend(preds_a)
                val_targs_val.extend(batch["targets"]["valence"].numpy())
                val_targs_aro.extend(batch["targets"]["arousal"].numpy())

        # Υπολογισμός Validation Metrics
        val_f1_val = f1_score(val_targs_val, val_preds_val, average='macro', zero_division=0)
        val_f1_aro = f1_score(val_targs_aro, val_preds_aro, average='macro', zero_division=0)
        mean_val_f1 = (val_f1_val + val_f1_aro) / 2

        # --- ΕΚΤΥΠΩΣΗ ---
        print(
            f"   Epoch {epoch + 1}/{epochs} | Loss: {train_loss / len(train_loader):.4f} | "
            f"Train F1 (V/A): {train_f1_val:.4f}/{train_f1_aro:.4f} | "
            f"Val F1 (V/A): {val_f1_val:.4f}/{val_f1_aro:.4f} (Mean: {mean_val_f1:.4f})"
        )

        # Αποθήκευση καλύτερου μοντέλου με βάση το Validation Mean F1
        if mean_val_f1 > best_f1:
            best_f1 = mean_val_f1
            torch.save(model.encoder.state_dict(), save_path)

    print(f"✅ Ολοκληρώθηκε! Το καλύτερο μοντέλο ({best_f1:.4f}) αποθηκεύτηκε στο {save_path}\n")
    return save_path