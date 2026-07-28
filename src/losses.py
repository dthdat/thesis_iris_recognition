from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular margin classification head."""

    def __init__(self, embedding_dim: int, num_classes: int, s: float = 64.0, m: float = 0.25):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        output = cosine * (1 - one_hot) + target * one_hot
        return output * self.s

    def get_cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(embeddings, weight)
