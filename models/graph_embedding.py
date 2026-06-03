import torch
import torch.nn as nn

class GraphEmbedding(nn.Module):
    def __init__(self, dim, method='attention'):
        super(GraphEmbedding, self).__init__()
        valid_methods = ['attention', 'concat', 'sum', 'mean']

        if method not in valid_methods:
            raise ValueError(
                f"Unknown graph embedding method: {method}"
            )

        if method == 'attention':
            self.scorer = nn.Linear(dim, 1)
        else:
            self.scorer = None
        self.method = method

    def forward(self, x):
        if self.method == 'attention':
            x = torch.stack(x)

            scores = self.scorer(x)
            weights = torch.softmax(scores, dim=1)

            x = (weights * x).sum(dim=1)
        elif self.method == 'concat':
            x = torch.cat(x, dim=-1)
        elif self.method == 'sum':
            x = torch.stack(x, dim=1)
            x = x.sum(dim=1)
        elif self.method == 'mean':
            x = torch.stack(x, dim=1)
            x = x.mean(dim=1)

        return x
