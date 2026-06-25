import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os


# =====================================================================
# 1. WRAPPERS ΓΙΑ SUPERVISED PRETRAINING (ΔΥΝΑΜΙΚΑ TASKS)
# =====================================================================
class SupervisedWrapper(nn.Module):
    def __init__(self, encoder, emb_dim=64, tasks=["valence", "arousal"]):
        super().__init__()
        self.encoder = encoder
        self.tasks = tasks

        # 🌟 Δυναμική δημιουργία των Task Heads
        self.heads = nn.ModuleDict()
        for task in self.tasks:
            self.heads[task] = nn.Sequential(
                nn.Linear(emb_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward(self, x):
        emb = self.encoder(x)
        preds = {}
        for task in self.tasks:
            preds[task] = self.heads[task](emb)
        return preds


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
        T = x.shape[-1]
        z = self.encoder(x)
        out = self.decoder(z.unsqueeze(-1))
        out = F.interpolate(out, size=T, mode='linear', align_corners=False)
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
                     lr=1e-3, tasks=["valence", "arousal"]):
    print(f"\n🚀 Pretraining: {modality_name.upper()} | Mode: {mode.upper()} | Fold: {fold}")

    save_dir = "pretrained_weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{modality_name}_fold{fold}.pth")

    if mode == "supervised":
        model = SupervisedWrapper(encoder, emb_dim=64, tasks=tasks).to(device)
        criterion = nn.BCEWithLogitsLoss()
        best_metric = 0.0
    elif mode == "autoencoder":
        if modality_name == "eeg":
            model = EEGAutoencoder(encoder, emb_dim=64, num_channels=32).to(device)
        elif modality_name == "ecg":
            model = PhysioAutoencoder(encoder, emb_dim=64, in_channels=3).to(device)
        elif modality_name == "eye":
            model = PhysioAutoencoder(encoder, emb_dim=64, in_channels=3).to(device)
        else:
            model = PhysioAutoencoder(encoder, emb_dim=64, in_channels=1).to(device)
        criterion = nn.MSELoss()
        best_metric = float('inf')  # Θέλουμε min MSE
    else:
        raise ValueError(f"Άγνωστο mode: {mode}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # --- TRAIN LOOP ---
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            if batch is None or len(batch) == 0: continue

            x = batch[modality_name].to(device)
            optimizer.zero_grad()

            if mode == "supervised":
                outputs = model(x)
                loss = 0
                # 🌟 Δυναμικός υπολογισμός Loss για όσα tasks είναι ενεργά
                for task in tasks:
                    targets = batch["targets"][task].to(device).float().view(-1)
                    loss += criterion(outputs[task].view(-1), targets)

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
            val_preds = {t: [] for t in tasks}
            val_targs = {t: [] for t in tasks}

        with torch.no_grad():
            for batch in val_loader:
                if batch is None or len(batch) == 0: continue
                x = batch[modality_name].to(device)

                if mode == "supervised":
                    outputs = model(x)
                    loss = 0
                    for task in tasks:
                        targets = batch["targets"][task].to(device).float().view(-1)
                        loss += criterion(outputs[task].view(-1), targets)

                        # Συλλογή predictions
                        preds = (torch.sigmoid(outputs[task].detach()) > 0.5).int().cpu().numpy()
                        val_preds[task].extend(preds)
                        val_targs[task].extend(targets.cpu().numpy())

                elif mode == "autoencoder":
                    mean = x.mean(dim=-1, keepdim=True)
                    std = x.std(dim=-1, keepdim=True)
                    target_x = (x - mean) / (std + 1e-8)
                    reconstructed_x = model(x)
                    loss = criterion(reconstructed_x, target_x)

                val_loss += loss.item()

        # --- METRICS & PRINTS ---
        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))

        if mode == "supervised":
            # Δυναμικός υπολογισμός F1-Scores
            f1_scores = []
            print_str = f"   Epoch {epoch + 1}/{epochs} | Loss: {avg_train_loss:.4f} | "

            for task in tasks:
                f1 = accuracy_score(val_targs[task], val_preds[task])
                f1_scores.append(f1)
                print_str += f"{task[:3].upper()} Acc: {f1:.4f} | "

            mean_val_f1 = sum(f1_scores) / len(f1_scores)
            print_str += f"(Mean: {mean_val_f1:.4f})"
            print(print_str)

            if mean_val_f1 >= best_metric:
                best_metric = mean_val_f1
                torch.save(model.encoder.state_dict(), save_path)

        elif mode == "autoencoder":
            print(f"   Epoch {epoch + 1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                torch.save(model.encoder.state_dict(), save_path)

    print(f"Best model for {modality_name} with best metric {best_metric} saved at {save_path} \n")
    return save_path