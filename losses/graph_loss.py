def graph_loss(z, scores):
    """
    z: [M, D]
    scores: [M, M] (directed)
    """

    loss = 0.0
    M = z.size(0)

    for i in range(M):
        for j in range(M):
            weight = scores[i, j]
            loss += weight * F.mse_loss(z[i], z[j])

    return loss