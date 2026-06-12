import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedFusion(nn.Module):

    def __init__(self, cross_modal_block):
        super().__init__()

        self.cross_modal_block = cross_modal_block

    def forward(self, modality_embeddings, scores, edges ):

        # modality_embeddings:
        # list([B,D])

        if torch.is_tensor(modality_embeddings):
            modality_list = torch.unbind(modality_embeddings, dim=1)
        else:
            modality_list = modality_embeddings

        tokens = [m.unsqueeze(1) for m in modality_list]

        M = len(tokens)

        incoming = [[] for _ in range(M)]

        for i, j, score in edges:
            fused, _ = self.cross_modal_block(tokens[i],tokens[j])

            incoming[i].append((fused.squeeze(1), score))

        outputs = []

        for i in range(M):
            if len(incoming[i]) == 0:
                outputs.append(modality_list[i])
                continue

            feats = torch.stack([x for x, _ in incoming[i]],dim=1)

            weights = torch.stack([s for _, s in incoming[i]])

            weights = F.softmax(weights, dim=0)

            fused = (feats * weights.view(1, -1, 1)).sum(dim=1)

            outputs.append(fused)

        return torch.stack(outputs, dim=1)