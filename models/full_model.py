import torch
import torch.nn as nn


class FullModel(nn.Module):

    def __init__(
        self,
        encoders,
        selector,
        fusion,
        graph_embedding,
        task_head
    ):
        super().__init__()

        self.encoders = nn.ModuleDict(encoders)
        self.selector = selector
        self.fusion = fusion
        self.graph_embedding = graph_embedding
        self.task_head = task_head

    def forward(self, batch):

        # ------------------------
        # 1. ENCODERS
        # ------------------------
        embeddings = []

        embeddings_dict = {}

        for name, encoder in self.encoders.items():

            z = encoder(batch[name])

            embeddings.append(z)
            embeddings_dict[name] = z

        embeddings = torch.stack(embeddings, dim=1)

        # ------------------------
        # 2. SELECTOR (GRAPH)
        # ------------------------
        scores, edges = self.selector(embeddings)

        # ------------------------
        # 3. FUSION (DIRECTED)
        # ------------------------
        fused_embeddings, scores, edges = self.fusion(
            embeddings
        )

        # ------------------------
        # 4. GRAPH EMBEDDING (READOUT)
        # ------------------------
        graph_emb = self.graph_embedding(fused_embeddings)

        # ------------------------
        # 5. TASK HEAD
        # ------------------------
        preds = self.task_head(graph_emb)

        # ------------------------
        # OUTPUT CONTRACT
        # ------------------------
        return {
            "pred": preds,
            "graph_emb": graph_emb,
            "fused": fused_embeddings,
            "embeddings": embeddings_dict,
            "edges": edges,
            "scores": scores
        }