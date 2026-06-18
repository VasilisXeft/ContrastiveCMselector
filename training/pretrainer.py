import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import os


# =====================================================================
# 1. WRAPPERS ΓΙΑ SUPERVISED PRETRAINING (VALENCE / AROUSAL)
# =====================================================================
class SupervisedWrapper(nn.Module):
    def __init__(self, encoder, emb_dim=64):
        super().__init__()
        self.encoder = encoder
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

# =====================================================================
# 2. WRAPPERS ΓΙΑ UNSUPERVISED PRETRAINING (AUTOENCODERS)
# =====================================================================
class PhysioAutoencoder(nn.Module):
    def __init__(self, encoder, emb_dim=64, in_channels=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(emb_dim, 32, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Παίρνουμε ΠΑΝΤΑ την τελευταία διάσταση ως χρόνο (T),
        # ανεξάρτητα αν είναι [B, C, T] ή απλά [B, T]
        T = x.shape[-1]

        z = self.encoder(x)
        out = self.decoder(z.unsqueeze(-1))
        out = F.interpolate(out, size=T, mode='linear', align_corners=False)

        # Αν το σήμα που μπήκε ήταν 2D [Batch, Time],
        # πρέπει και η έξοδος να είναι 2D για να ταιριάζει στο MSELoss
        if x.dim() == 2:
            out = out.squeeze(1)

        return out


class EEGAutoencoder(nn.Module):
    def __init__(self, encoder, emb_dim=64, num_channels=32):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(emb_dim, 64, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(64, num_channels, kernel_size=8, stride=2, padding=3)
        )

    def forward(self, x):
        T = x.shape[-1]

        z = self.encoder(x)
        out = self.decoder(z.unsqueeze(-1))
        out = F.interpolate(out, size=T, mode='linear', align_corners=False)

        if x.dim() == 2:
            out = out.squeeze(1)

        return out


# =====================================================================
# 3. Η ΚΕΝΤΡΙΚΗ ΣΥΝΑΡΤΗΣΗ PRETRAINING
# =====================================================================
def pretrain_encoder(modality_name, encoder, train_loader, val_loader, fold, device, mode="supervised", epochs=15,
                     lr=1e-3):
    """
    Ενοποιημένη ρουτίνα pretraining.
    mode: "supervised" (για F1-score σε Valence/Arousal) ή "autoencoder" (για MSE Reconstruction)
    """
    print(f"\n🚀 Pretraining: {modality_name.upper()} | Mode: {mode.upper()} | Fold: {fold}")

    save_dir = "pretrained_weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{modality_name}_fold{fold}.pth")

    # --- SETUP ΒΑΣΕΙ ΤΟΥ MODE ---
    if mode == "supervised":
        model = SupervisedWrapper(encoder, emb_dim=64).to(device)
        criterion = nn.BCEWithLogitsLoss()
        best_metric = 0.0  # Θέλουμε το F1 να ΜΕΓΑΛΩΣΕΙ

    elif mode == "autoencoder":
        if modality_name == "eeg":
            model = EEGAutoencoder(encoder, emb_dim=64, num_channels=32).to(device)
        elif modality_name == "ecg":
            model = EEGAutoencoder(encoder, emb_dim=64, num_channels=3).to(device)
        else:
            model = PhysioAutoencoder(encoder, emb_dim=64, in_channels=1).to(device)
        criterion = nn.MSELoss()
        best_metric = float('inf')  # Θέλουμε το MSE να ΜΙΚΡΥΝΕΙ

    else:
        raise ValueError(f"Άγνωστο mode: {mode}. Επίλεξε 'supervised' ή 'autoencoder'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # --- ΚΕΝΤΡΙΚΗ ΛΟΥΠΑ ΕΚΠΑΙΔΕΥΣΗΣ ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Λίστες μόνο για supervised
        if mode == "supervised":
            train_preds_v, train_preds_a, train_targs_v, train_targs_a = [], [], [], []

        # --- TRAIN LOOP ---
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            x = batch[modality_name].to(device)
            optimizer.zero_grad()

            if mode == "supervised":
                targets_val = batch["targets"]["valence"].to(device).float().view(-1)
                targets_aro = batch["targets"]["arousal"].to(device).float().view(-1)

                outputs = model(x)
                loss = criterion(outputs["valence"].view(-1), targets_val) + \
                       criterion(outputs["arousal"].view(-1), targets_aro)

                # Συλλογή metrics
                preds_v = (torch.sigmoid(outputs["valence"].detach()) > 0.5).int().cpu().numpy()
                preds_a = (torch.sigmoid(outputs["arousal"].detach()) > 0.5).int().cpu().numpy()
                train_preds_v.extend(preds_v)
                train_preds_a.extend(preds_a)
                train_targs_v.extend(targets_val.cpu().numpy())
                train_targs_a.extend(targets_aro.cpu().numpy())

            elif mode == "autoencoder":
                mean = x.mean(dim=-1, keepdim=True)
                std = x.std(dim=-1, keepdim=True)
                target_x = (x - mean) / (std + 1e-8)

                reconstructed_x = model(x)
                loss = criterion(reconstructed_x, target_x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- VAL LOOP ---
        model.eval()
        val_loss = 0.0

        if mode == "supervised":
            val_preds_v, val_preds_a, val_targs_v, val_targs_a = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch[modality_name].to(device)

                if mode == "supervised":
                    targets_val = batch["targets"]["valence"].to(device).float().view(-1)
                    targets_aro = batch["targets"]["arousal"].to(device).float().view(-1)

                    outputs = model(x)
                    loss = criterion(outputs["valence"].view(-1), targets_val) + \
                           criterion(outputs["arousal"].view(-1), targets_aro)

                    preds_v = (torch.sigmoid(outputs["valence"]) > 0.5).int().cpu().numpy()
                    preds_a = (torch.sigmoid(outputs["arousal"]) > 0.5).int().cpu().numpy()
                    val_preds_v.extend(preds_v)
                    val_preds_a.extend(preds_a)
                    val_targs_v.extend(targets_val.cpu().numpy())
                    val_targs_a.extend(targets_aro.cpu().numpy())

                elif mode == "autoencoder":
                    mean = x.mean(dim=-1, keepdim=True)
                    std = x.std(dim=-1, keepdim=True)
                    target_x = (x - mean) / (std + 1e-8)

                    reconstructed_x = model(x)
                    loss = criterion(reconstructed_x, target_x)

                val_loss += loss.item()

        # --- METRICS & PRINTS ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if mode == "supervised":
            val_f1_v = f1_score(val_targs_v, val_preds_v, average='macro', zero_division=0)
            val_f1_a = f1_score(val_targs_a, val_preds_a, average='macro', zero_division=0)
            mean_val_f1 = (val_f1_v + val_f1_a) / 2

            print(
                f"   Epoch {epoch + 1}/{epochs} | Loss: {avg_train_loss:.4f} | Val F1 (V/A): {val_f1_v:.4f}/{val_f1_a:.4f} (Mean: {mean_val_f1:.4f})")

            # Αποθήκευση στο supervised: max F1
            if mean_val_f1 > best_metric:
                best_metric = mean_val_f1
                torch.save(model.encoder.state_dict(), save_path)

        elif mode == "autoencoder":
            print(f"   Epoch {epoch + 1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

            # Αποθήκευση στον autoencoder: min MSE
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                torch.save(model.encoder.state_dict(), save_path)

    print(f"✅ Ολοκληρώθηκε! Το καλύτερο μοντέλο αποθηκεύτηκε στο {save_path}\n")
    return save_path