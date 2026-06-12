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
    preds = outputs["pred"]
    targets = batch["targets"]

    # --------------------------
    # 4. BACKPROP
    # --------------------------
    loss.backward()

    # Μετά το backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.abs().sum() == 0:
                print(f"⚠️ ΠΡΟΣΟΧΗ: Το layer {name} έχει μηδενικό gradient!")

    # (optional stability trick)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
    )

    optimizer.step()

    logs["lr"] = optimizer.param_groups[0]["lr"]

    return logs, preds, targets


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