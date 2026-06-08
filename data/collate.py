import torch


def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None
    
    out = {}

    # ======================================================
    # FACE (streaming → tensor)
    # ======================================================
    face_batch = []

    for b in batch:

        frames = list(b["face"])  # consume generator

        if len(frames) == 0:
            raise ValueError("Empty face stream in batch")

        face_batch.append(torch.stack(frames))  # [T,3,H,W]

    out["face"] = torch.stack(face_batch, dim=0)  # [B,T,3,H,W]

    # ----------------------
    # EEG: [B, C, T]
    # ----------------------
    out["eeg"] = torch.stack([b["eeg"] for b in batch], dim=0)

    # ----------------------
    # PPG: [B, ...]
    # ----------------------
    out["ppg"] = torch.stack([b["ppg"] for b in batch], dim=0)

    # ----------------------
    # EDA: [B, ...]
    # ----------------------
    out["eda"] = torch.stack([b["eda"] for b in batch], dim=0)

    # ----------------------
    # TMP: [B, ...]
    # ----------------------
    out["tmp"] = torch.stack([b["tmp"] for b in batch], dim=0)

    # ----------------------
    # Signal quality: [B, ...]
    # ----------------------
    out["signal_quality"] = torch.stack([b["signal_quality"] for b in batch], dim=0)

    # ----------------------
    # TARGETS (DICT OF LISTS)
    # ----------------------
    targets = {}

    for key in batch[0]["targets"].keys():

        targets[key] = torch.stack(
            [b["targets"][key] for b in batch],
            dim=0
        )

    out["targets"] = targets

    return out