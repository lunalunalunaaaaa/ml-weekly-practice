"""24 小时油温的单头缩放点积自注意力。"""
from typing import NamedTuple

import torch
from torch import nn


class AttentionTrace(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    scores: torch.Tensor
    weights: torch.Tensor
    readout: torch.Tensor


def add_position(embedded):
    """在第一个特征上加从 0 到 1 的位置值，批内各窗口共用位置。"""
    position = embedded.new_zeros(embedded.shape[-2:])
    position[:, 0] = torch.linspace(
        0, 1, embedded.size(1), device=embedded.device, dtype=embedded.dtype
    )
    return embedded + position


class OilAttention(nn.Module):
    width = 16

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, self.width)
        self.query = nn.Linear(self.width, self.width)
        self.key = nn.Linear(self.width, self.width)
        self.value = nn.Linear(self.width, self.width)
        self.head = nn.Linear(self.width, 1)

    def attend(self, x, *, with_position=False):
        """输入 (B, 24, 1)，返回同次前向的 Q/K/V、分数、权重和读出。"""
        embedded = self.embed(x)
        if with_position:
            embedded = add_position(embedded)
        query = self.query(embedded)
        key = self.key(embedded)
        value = self.value(embedded)
        scores = (query @ key.transpose(-2, -1)) / self.width**0.5
        weights = scores.softmax(dim=-1)
        readout = weights @ value
        return AttentionTrace(query, key, value, scores, weights, readout)

    def forward(self, x):
        return self.head(self.attend(x).readout[:, -1])
