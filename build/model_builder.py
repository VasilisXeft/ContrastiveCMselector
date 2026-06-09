import yaml
import torch.nn as nn

from models import modality_dropout
from models.full_model import FullModel
from models.graph_embedding import GraphEmbedding
from models.task_head import TaskHead

# encoders imports
from models.encoders import VisualEncoder
from models.encoders import EEGNetEncoder
from models.encoders import PhysioEncoder

# selector + fusion imports
from models.selector import DirectedContrastiveSelector
from models.reliability_score import ReliabilityGating
from models.fusion import DirectedFusion
from models.cross_modal import CrossModalBlock
from models.modality_dropout import ModalityDropout


def build_model(cfg_path):

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]

    # ----------------------
    # 1. ENCODERS
    # ----------------------
    encoders = {
        "face": VisualEncoder(**model_cfg["encoders"]["face"]),
        "eeg": EEGNetEncoder(**model_cfg["encoders"]["eeg"]),
        "ppg": PhysioEncoder(**model_cfg["encoders"]["ppg"]),
        "eda": PhysioEncoder(**model_cfg["encoders"]["eda"]),
        "tmp": PhysioEncoder(**model_cfg["encoders"]["tmp"])
    }

    # ------------------------
    # 2. MODALITY DROPOUT
    # ------------------------
    modality_dropout = ModalityDropout(**model_cfg["modality_dropout"])

    # ------------------------
    # 3. RELIABILITY GATING
    # ------------------------
    reliability_score = ReliabilityGating(
        emb_dim=model_cfg["reliability_score"]["emb_dim"],
        num_modalities=model_cfg["reliability_score"]["num_modalities"],
        init_lambda=model_cfg["reliability_score"]["init_lambda"]
    )

    # ----------------------
    # 4. SELECTOR
    # ----------------------
    selector = DirectedContrastiveSelector(
        top_k=model_cfg["selector"]["top_k"],
        temperature=model_cfg["selector"]["temperature"]
    )

    # ----------------------
    # 5. FUSION
    # ----------------------
    cross_modal_block = CrossModalBlock()
    fusion = DirectedFusion(cross_modal_block)

    # ----------------------
    # 6. GRAPH EMBEDDING
    # ----------------------
    graph_embedding = GraphEmbedding(
        dim=64,
        method=model_cfg["graph_embedding"]["type"]
    )

    # ----------------------
    # 7. TASK HEAD
    # ----------------------
    task_head = TaskHead(
        input_dim=64,
        task_config=model_cfg["task_head"]
    )

    return FullModel(
        encoders=encoders,
        modality_dropout=modality_dropout,
        reliability_score=reliability_score,
        selector=selector,
        fusion=fusion,
        graph_embedding=graph_embedding,
        task_head=task_head
    ), cfg