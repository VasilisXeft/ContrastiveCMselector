import torch
import torch.nn as nn
import os


class FullModel(nn.Module):

    def __init__(
            self,
            encoders,
            modality_dropout,
            reliability_score,
            selector,
            fusion,
            graph_embedding,
            task_head,
            pretrained_weights=None,  # NEW: Λεξικό με τα μονοπάτια των βαρών π.χ. {"eeg": "eeg_fold1.pth"}
            freeze_encoders=True  # NEW: Flag για το αν θέλουμε να τα παγώσουμε
    ):
        super().__init__()

        self.encoders = nn.ModuleDict(encoders)
        self.modality_dropout = modality_dropout
        self.reliability_score = reliability_score
        self.selector = selector
        self.fusion = fusion
        self.graph_embedding = graph_embedding
        self.task_head = task_head

        self.freeze_encoders_flag = freeze_encoders

        # ------------------------
        # LOAD & FREEZE ENCODERS
        # ------------------------
        if pretrained_weights is not None:
            self._load_and_freeze_encoders(pretrained_weights, freeze_encoders)

    def _load_and_freeze_encoders(self, pretrained_weights, freeze):
        """
        Εσωτερική μέθοδος που φορτώνει τα βάρη από τα paths και παγώνει τους encoders.
        """
        for name, encoder in self.encoders.items():
            # 1. Φόρτωση Βαρών
            if name in pretrained_weights and pretrained_weights[name] is not None:
                weight_path = pretrained_weights[name]
                if os.path.exists(weight_path):
                    encoder.load_state_dict(torch.load(weight_path))
                    print(f"✅ Φορτώθηκαν τα pre-trained βάρη για το: {name.upper()}")
                else:
                    print(f"⚠️ Προσοχή: Το αρχείο βαρών δεν βρέθηκε -> {weight_path}")

            # 2. Πάγωμα (Freezing)
            if freeze:
                for param in encoder.parameters():
                    param.requires_grad = False
                print(f"❄️ Ο encoder {name.upper()} πάγωσε (Δεν εκπαιδεύεται).")

    def train(self, mode=True):
        """
        OVERRIDE της default train() του PyTorch.
        Είναι ΑΠΑΡΑΙΤΗΤΟ για να παραμείνουν τα BatchNorm/Dropout των encoders
        σε eval() state, ακόμα και όταν το υπόλοιπο μοντέλο μπαίνει σε train().
        """
        super().train(mode)
        if self.freeze_encoders_flag:
            for encoder in self.encoders.values():
                encoder.eval()
        return self

    def forward(self, batch):

        # ------------------------
        # 1. ENCODERS
        # ------------------------
        embeddings = []
        embeddings_dict = {}

        for name, encoder in self.encoders.items():
            try:
                z = encoder(batch[name])
                embeddings.append(z)
                embeddings_dict[name] = z
            except:
                continue

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