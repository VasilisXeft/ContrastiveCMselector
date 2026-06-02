import torch


def collate_fn(batch):

    out = {}

    # ----------------------
    # FACE: [B, T, 3, H, W]
    # ----------------------
    out["face"] = torch.stack([b["face"] for b in batch], dim=0)

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