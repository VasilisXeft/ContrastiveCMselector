import torch
import torch.nn as nn


class MLPHead(nn.Module):

    def __init__(self, input_dim, output_dim, hidden=128, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class TaskHead(nn.Module):

    def __init__(self, input_dim, task_config):
        super().__init__()

        self.task_type = task_config["type"]

        self.heads = nn.ModuleDict()

        for name, cfg in task_config["outputs"].items():

            self.heads[name] = MLPHead(
                input_dim=input_dim,
                output_dim=cfg["output_dim"]
            )

    def forward(self, x):

        outputs = {}

        for name, head in self.heads.items():

            outputs[name] = head(x)

        return outputs