"""实践四至六：双头注意力、因果遮罩与单层编码器。"""
from typing import NamedTuple

import torch
from torch import nn


class HeadOutput(NamedTuple):
    weights: torch.Tensor
    readout: torch.Tensor


class TwoHeadOutput(NamedTuple):
    head1: HeadOutput
    head2: HeadOutput
    concatenated: torch.Tensor
    output: torch.Tensor


class AttentionHead(nn.Module):
    """两个头都接收完整 X，各自学习三套 16 → 8 的投影。"""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(16, 8)
        self.key = nn.Linear(16, 8)
        self.value = nn.Linear(16, 8)

    def forward(self, x, *, use_causal_mask=False):
        # X: (B, 24, 16)，Q/K/V: (B, 24, 8)。
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # 行是查询小时，列是被读取小时；默认关闭遮罩，兼容实践四。
        scores = (q @ k.transpose(-2, -1)) / q.size(-1) ** 0.5
        if use_causal_mask:
            t = scores.size(-1)
            # True 是未来位置：列下标 > 行下标。对角线保留，允许读取自己。
            future = torch.ones(t, t, dtype=torch.bool, device=scores.device).triu(1)
            # 必须先遮住分数，再 softmax，让历史位置分完全部权重。
            scores = scores.masked_fill(future, float('-inf'))
        weights = torch.softmax(scores, dim=-1)  # (B, 24, 24)
        readout = weights @ v                   # (B, 24, 8)
        return HeadOutput(weights, readout)


class TwoHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 默认独立初始化；复制参数只在实践四第一组对照中进行。
        self.head1 = AttentionHead()
        self.head2 = AttentionHead()
        self.output_projection = nn.Linear(16, 16)

    def forward(self, x, *, use_causal_mask=False):
        # 同一个开关传给两个头，避免其中一个头仍然读取未来。
        first = self.head1(x, use_causal_mask=use_causal_mask)
        second = self.head2(x, use_causal_mask=use_causal_mask)
        # 拼的是特征轴，不是小时轴：两个 8 维读出合为 16 维。
        concatenated = torch.cat([first.readout, second.readout], dim=-1)
        output = self.output_projection(concatenated)
        return TwoHeadOutput(first, second, concatenated, output)


class EncoderTrace(NamedTuple):
    X: torch.Tensor
    M: torch.Tensor
    H: torch.Tensor
    F: torch.Tensor
    E: torch.Tensor
    last: torch.Tensor
    prediction: torch.Tensor


class OilEncoder(nn.Module):
    """实践六：原图顺序的 Post-LN 编码器，末行接油温预测层。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 16)
        self.attention = TwoHeadAttention()  # 两头独立初始化，不复制参数。
        self.norm1 = nn.LayerNorm(16)
        self.feed_forward = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        self.norm2 = nn.LayerNorm(16)
        self.predictor = nn.Linear(16, 1)
        # 简化位置编码：(t - 1) / 23 仅放在第一个特征中。
        position = torch.zeros(1, 24, 16)
        position[0, :, 0] = torch.arange(24) / 23
        self.register_buffer('position', position)

    def encode(self, x):
        if x.ndim != 3 or x.shape[1:] != (24, 1):
            raise ValueError('输入形状必须为 (B, 24, 1)')
        X = self.embed(x) + self.position               # (B, 24, 16)
        # 24 次观测都已取得，预测第 25 小时无需窗口内因果遮罩。
        M = self.attention(X, use_causal_mask=False).output
        H = self.norm1(X + M)                          # 第一处残差
        F = self.feed_forward(H)                       # 逐行 16 → 32 → 16
        E = self.norm2(H + F)                          # 第二处残差
        last = E[:, 23, :]                             # 编码结束后才取末行
        prediction = self.predictor(last)              # (B, 1)
        return EncoderTrace(X, M, H, F, E, last, prediction)

    def forward(self, x):
        return self.encode(x).prediction
