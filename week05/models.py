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
    # hidden_dim 同时控制第一层输出与第二层输入。
    return nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))


def forward_shapes(model, x):
    shapes = [tuple(x.shape)]
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            shapes.append(tuple(x.shape))
    return shapes
