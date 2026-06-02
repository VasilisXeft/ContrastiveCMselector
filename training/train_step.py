import torch


def train_step(
    batch,
    model,
    optimizer,
    loss_router,
    device
):

    model.train()
    optimizer.zero_grad()

    # --------------------------
    # 1. MOVE TO DEVICE
    # --------------------------
    batch = move_batch_to_device(batch, device)

    # --------------------------
    # 2. FORWARD PASS
    # --------------------------
    outputs = model(batch)

    # --------------------------
    # 3. LOSS COMPUTATION
    # --------------------------
    loss, logs = loss_router.compute(outputs, batch)

    # --------------------------
    # 4. BACKPROP
    # --------------------------
    loss.backward()

    # (optional stability trick)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
    )

    optimizer.step()

    logs["lr"] = optimizer.param_groups[0]["lr"]

    return logs


# helper
def move_batch_to_device(batch, device):

    if isinstance(batch, dict):

        return {
            k: move_batch_to_device(v, device)
            for k, v in batch.items()
        }

    if torch.is_tensor(batch):
        return batch.to(device)

    return batch