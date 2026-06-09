import torch
import torch.nn as nn


class FullModel(nn.Module):

    def __init__(
        self,
        encoders,
        modality_dropout,
        reliability_score,
        selector,
        fusion,
        graph_embedding,
        task_head
    ):
        super().__init__()

        self.encoders = nn.ModuleDict(encoders)
        self.modality_dropout = modality_dropout
        self.reliability_score = reliability_score
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
        # 2. MODALITY DROPOUT
        # ------------------------

        embeddings = self.modality_dropout(embeddings)

        # ------------------------
        # 3. RELIABILITY GATING
        # ------------------------

        embeddings, r = self.reliability_score(embeddings, batch["signal_quality"])

        # ------------------------
        # 4. SELECTOR (GRAPH)
        # ------------------------

        scores, edges = self.selector(embeddings)

        # ------------------------
        # 5. FUSION (DIRECTED)
        # ------------------------

        fused_embeddings = self.fusion(
            embeddings, scores, edges
        )

        # ------------------------
        # 6. GRAPH EMBEDDING (READOUT)
        # ------------------------

        graph_emb = self.graph_embedding(fused_embeddings)

        # ------------------------
        # 7. TASK HEAD
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
            "scores": scores,
            "reliability_score": r
        }