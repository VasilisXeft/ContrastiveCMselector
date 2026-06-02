from torch.utils.data import DataLoader
from data.collate import collate_fn


def build_dataloader(dataset, batch_size, shuffle=True):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )