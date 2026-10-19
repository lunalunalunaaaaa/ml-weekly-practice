"""评价不反传、不更新参数；最终指标回到原始目标单位。"""
import numpy as np
import torch


def predict_scaled(model, x):
    # eval 切换层的行为，no_grad 关闭自动求导；两者职责不同，需要分别调用。
    model.eval()
    with torch.no_grad():
        return model(x)


def scaled_mse(model, x, y):
    # 强制形状一致，防止 (N, 1) 与 (N,) 广播成 (N, N)，得到错误损失。
    pred = predict_scaled(model, x)
    assert pred.shape == y.shape
    return float(torch.mean((pred - y) ** 2).item())


def metrics(y, pred):
    y, pred = np.asarray(y).reshape(-1), np.asarray(pred).reshape(-1)
    assert y.shape == pred.shape
    return {'mae': float(np.mean(np.abs(y - pred))),
            'rmse': float(np.sqrt(np.mean((y - pred) ** 2)))}


def evaluate_mlp(model, data, split):
    # 模型预测的是标准化目标，必须先恢复原始单位，才能与基线公平比较。
    pred = predict_scaled(model, data['x'][split]).numpy().reshape(-1)
    pred_raw = pred * data['y_std'] + data['y_mean']
    return metrics(data['raw_y'][split], pred_raw)


def evaluate_baseline(model, data, split):
    pred = (np.full(len(data['raw_y'][split]), model) if np.isscalar(model)
            else model.predict(data['inputs'][split]))
    return metrics(data['raw_y'][split], pred)
