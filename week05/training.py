"""核心：清梯度 → 前向 → 损失 → 反传 → 更新。"""
import copy
import math
from torch import nn
from sklearn.linear_model import LinearRegression
from evaluation import scaled_mse


def one_step(model, optimizer, x, y):
    loss_fn = nn.MSELoss()
    rows = []

    def record(stage, loss=None):
        rows.append(dict(stage=stage, weight=model.weight.item(), bias=model.bias.item(),
                         weight_grad=None if model.weight.grad is None else model.weight.grad.item(),
                         bias_grad=None if model.bias.grad is None else model.bias.grad.item(), loss=loss))

    optimizer.zero_grad()
    pred = model(x)
    assert pred.shape == y.shape
    loss = loss_fn(pred, y)
    record('after_zero_grad', loss.item())
    loss.backward()  # 自动求导：相当于手写 dw、db。
    record('after_backward')
    old_w, old_b = model.weight.item(), model.bias.item()
    dw, db = model.weight.grad.item(), model.bias.grad.item()
    optimizer.step()  # 用梯度更新参数；backward 本身不更新参数。
    updated_loss = loss_fn(model(x), y).item()
    record('after_step', updated_loss)
    assert math.isclose(loss.item(), 10., abs_tol=1e-6)
    assert math.isclose(dw, -10., abs_tol=1e-6)
    assert math.isclose(db, -6., abs_tol=1e-6)
    assert math.isclose(model.weight.item(), old_w - 0.1 * dw, abs_tol=1e-6)
    assert math.isclose(model.bias.item(), old_b - 0.1 * db, abs_tol=1e-6)
    assert math.isclose(updated_loss, 0.26, abs_tol=1e-6)
    return rows


def fit_mlp(model, optimizer, loader, data, name, lr, epochs=200):
    loss_fn = nn.MSELoss()
    history, best_value, best_epoch, best_state = [], float('inf'), None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
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
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_epoch


def fit_baselines(data):
    return {'Mean baseline': float(data['raw_y']['train'].mean()),
            'Linear regression': LinearRegression().fit(data['inputs']['train'],
                                                        data['raw_y']['train'])}
