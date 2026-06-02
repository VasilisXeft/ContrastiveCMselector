def train_step(batch, model, optim, losses, device):

    model.train()
    optim.zero_grad()

    # --------------------------
    # 1. forward pass
    # --------------------------
    out = model(batch)

    preds = out["pred"]
    graph_emb = out["graph_emb"]
    fused = out["fused"]
    edges = out["edges"]

    targets = batch["targets"]

    # --------------------------
    # 2. TASK LOSS
    # --------------------------
    task_loss = 0.0

    for name, pred in preds.items():

        if name in losses["task"]:

            task_loss += losses["task"][name](
                pred,
                targets[name]
            )

    # --------------------------
    # 3. CONTRASTIVE LOSS
    # --------------------------
    # assume we use graph embedding for alignment
    contrastive_loss = losses["contrastive"](
        graph_emb,
        batch["graph_emb_pos"]
    )

    # --------------------------
    # 4. GRAPH REGULARIZATION LOSS
    # --------------------------
    graph_loss = losses["graph"](edges)

    # --------------------------
    # 5. TOTAL LOSS
    # --------------------------
    loss = (
        task_loss
        + losses["lambda_task"] * task_loss
        + losses["lambda_contrastive"] * contrastive_loss
        + losses["lambda_graph"] * graph_loss
    )

    loss.backward()
    optim.step()

    return {
        "loss": loss.item(),
        "task": task_loss.item(),
        "contrastive": contrastive_loss.item(),
        "graph": graph_loss.item()
    }
