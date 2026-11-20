"""实践六：训练整层编码器，按验证 MSE 保存并恢复最佳参数。"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from week07.training import scaled_mse

SEED = 42
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def fit(model, train, val):
    loader = DataLoader(
        TensorDataset(train['x'], train['y']), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    watched = {
        'head1_query': model.attention.head1.query.weight,
        'predictor': model.predictor.weight,
    }
    history, first_batch = [], {}
    best_value, best_epoch, best_state = math.inf, None, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch, (x, y) in enumerate(loader):
            observe = epoch == 1 and batch == 0
            optimizer.zero_grad(set_to_none=True)
            if observe:
                before = {name: parameter.detach().clone() for name, parameter in watched.items()}
            prediction = model(x)
            assert prediction.shape == y.shape
            loss = loss_fn(prediction, y)
            loss.backward()
            if observe:
                # 每个可学习参数都应连接到末端损失；buffer position 不需要梯度。
                for name, parameter in model.named_parameters():
                    assert parameter.grad is not None, f'{name} 没有梯度'
                    assert torch.isfinite(parameter.grad).all(), f'{name} 梯度非有限'
                for name, parameter in watched.items():
                    assert torch.equal(parameter, before[name])  # backward 尚未更新参数
                    first_batch[name] = {
                        'gradient_shape': list(parameter.grad.shape),
                        'gradient_norm': parameter.grad.norm().item(),
                        'max_change_after_backward': (parameter - before[name]).abs().max().item(),
                    }
            optimizer.step()
            if observe:
                for name, parameter in watched.items():
                    change = (parameter - before[name]).abs().max().item()
                    assert change > 0
                    first_batch[name]['max_change_after_step'] = change

        train_mse, val_mse = scaled_mse(model, train), scaled_mse(model, val)
        if not all(math.isfinite(value) for value in (train_mse, val_mse)):
            raise RuntimeError('训练或验证损失出现非有限值')
        history.append({'epoch': epoch, 'train_mse': train_mse, 'val_mse': val_mse})
        if val_mse < best_value:  # 同分不覆盖，保留较早轮次。
            best_value, best_epoch = val_mse, epoch
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f'epoch={epoch:02d} train={train_mse:.6f} val={val_mse:.6f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return history, best_epoch, first_batch
