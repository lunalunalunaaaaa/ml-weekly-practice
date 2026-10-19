"""网络定义与前向形状检查。"""
import torch
from torch import nn


def linear_model():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def mlp(hidden_dim=16):
    """10 个输入特征 → 隐藏层 → 非线性激活 → 1 个回归预测值。"""
    # hidden_dim 同时控制第一层输出与第二层输入。
    # ReLU 让模型能够表达非线性关系；输出层不加激活，允许预测负的标准分数。
    return nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))


def forward_shapes(model, x):
    """逐层前向并记录形状，帮助检查相邻 Linear 的输入输出是否匹配。"""
    shapes = [tuple(x.shape)]
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            shapes.append(tuple(x.shape))
    return shapes
