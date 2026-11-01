"""逐批训练、整集评价、独立复制并恢复最佳参数。"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def predict_scaled(model, x, batch_size=128):
    """按批前向以限制内存；返回一维 float64 数组，内容仍是标准化预测。"""
    model.eval()
    return np.concatenate([model(xb).numpy().ravel().astype(np.float64)
                           for xb in x.split(batch_size)])


def scaled_mse(model, split):
    # 全部目标误差平方和 / 全部目标数；最后一个不足 128 的批次按真实数量计入。
    error = predict_scaled(model, split['x']) - split['y'].numpy().ravel().astype(np.float64)
    return float(np.square(error).sum() / error.size)


def fit_lstm(model, tensors, epochs=40):
    """返回 history、best_epoch；模型原地恢复最佳参数。此接口不接收测试集。

    验证只负责选择快照；完整训练 epochs 轮，不提前停止，不合并重训。
    """
    generator = torch.Generator().manual_seed(42)  # 每组都新建，同一训练样本顺序。
    loader = DataLoader(TensorDataset(tensors['train']['x'], tensors['train']['y']),
                        batch_size=128, shuffle=True, drop_last=False,
                        num_workers=0, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    loss_fn = nn.MSELoss()
    best_value, best_epoch, best_state = math.inf, None, None
    history = []
    name = f'lstm_h{model.hidden_size}'
    for epoch in range(1, epochs+1):
        model.train()  # 上轮评价进入了 eval，本轮恢复训练模式。
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            assert prediction.shape == yb.shape
            loss = loss_fn(prediction, yb)
            loss.backward()    # 末端损失沿 24 步计算关系传播到共享的门控参数。
            optimizer.step()   # 更新参数；门值、h、c 在下次前向时重新计算。
        train_mse = scaled_mse(model, tensors['train'])
        val_mse = scaled_mse(model, tensors['val'])
        if not all(math.isfinite(v) for v in (train_mse, val_mse)):
            raise RuntimeError('训练出现非有限损失')
        history.append(dict(model=name, hidden_size=model.hidden_size, epoch=epoch,
                            train_mse=train_mse, val_mse=val_mse, scale='standardized'))
        if val_mse < best_value:  # 相同不覆盖，保留较早轮次。
            best_value, best_epoch = val_mse, epoch
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f'{name} epoch={epoch:02d} train={train_mse:.6f} val={val_mse:.6f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return history, best_epoch
