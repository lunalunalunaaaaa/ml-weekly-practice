"""核心：清梯度 → 前向 → 损失 → 反传 → 更新。"""
import copy
import math
import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from evaluation import scaled_mse


def one_step(model, optimizer, x, y):
    """对单输入、单输出的线性模型执一次更新，并返回三个阶段的快照。

    本函数只负责更新和记录；固定样例的手算结果由入口检查，
    因而调用者修改数据或学习率时，不会被旧的预期数值限制。
    """
    loss_fn = nn.MSELoss()
    rows = []

    def record(stage, loss=None):
        rows.append(dict(stage=stage, weight=model.weight.item(), bias=model.bias.item(),
                         weight_grad=None if model.weight.grad is None else model.weight.grad.item(),
                         bias_grad=None if model.bias.grad is None else model.bias.grad.item(), loss=loss))

    # 显式将梯度置为 None，避免沿用上一次 backward 累积的梯度。
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    assert pred.shape == y.shape
    loss = loss_fn(pred, y)
    record('after_zero_grad', loss.item())
    loss.backward()  # 自动求导：相当于手写 dw、db。
    record('after_backward')
    optimizer.step()  # 用梯度更新参数；backward 本身不更新参数。
    # 原来的 pred 是更新前的计算结果，必须重新前向才能观察更新后的损失。
    # 此处只记录数值，不再反传，所以不需要构建新的计算图。
    with torch.no_grad():
        updated_loss = loss_fn(model(x), y).item()
    record('after_step', updated_loss)
    return rows


def fit_mlp(model, optimizer, loader, data, name, lr, epochs=200):
    """训练指定轮数，返回逐轮记录和最佳轮次，并原地恢复最佳模型。

    loader 只提供训练样本；验证集只用于选择参数快照，不参与梯度计算。
    lr 是写入结果表的配置说明，实际更新使用 optimizer 中的学习率。
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if len(loader) == 0:
        raise ValueError('Training loader must contain at least one batch')
    loss_fn = nn.MSELoss()
    history, best_value, best_epoch, best_state = [], float('inf'), None, None
    for epoch in range(1, epochs + 1):
        # 上一轮评价会切换到 eval 模式，每轮训练前需要切回 train 模式。
        # 当前网络没有 Dropout/BatchNorm，但保留此习惯方便以后扩展。
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            assert pred.shape == yb.shape
            loss = loss_fn(pred, yb)  # 本次梯度只来自当前批次。
            loss.backward()
            optimizer.step()
        # 使用整轮结束时的同一模型重新评价完整数据，不使用最后一批 loss。
        train_mse = scaled_mse(model, data['x']['train'], data['y']['train'])
        val_mse = scaled_mse(model, data['x']['validation'], data['y']['validation'])
        if not (math.isfinite(train_mse) and math.isfinite(val_mse)):
            raise RuntimeError('Training produced a non-finite loss')
        history.append(dict(config=name, learning_rate=lr, epoch=epoch,
                            train_mse=train_mse, validation_mse=val_mse,
                            target_scale='standardized'))
        if val_mse < best_value:  # 严格小于：相同时保留较早轮次。
            best_value, best_epoch = val_mse, epoch
            # 必须深拷贝；仅保存引用会让所谓“最佳参数”随后续更新一起变化。
            best_state = copy.deepcopy(model.state_dict())
    # 训练满 epochs 轮后才恢复最佳参数；这不是提前停止训练。
    model.load_state_dict(best_state)
    return history, best_epoch


def fit_baselines(data):
    """仅用训练集拟合两个基线，供 MLP 的误差作参照。"""
    return {'Mean baseline': float(data['raw_y']['train'].mean()),
            'Linear regression': LinearRegression().fit(data['inputs']['train'],
                                                        data['raw_y']['train'])}
