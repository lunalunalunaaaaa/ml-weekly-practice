"""单头模型训练和首批反向传播核对。"""
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
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    history = []
    first_batch = {}
    best_value = math.inf
    best_epoch = None
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch, (x, y) in enumerate(loader):
            observe = epoch == 1 and batch == 0
            optimizer.zero_grad(set_to_none=True)
            if observe:
                before = model.value.weight.detach().clone()
            prediction = model(x)
            assert prediction.shape == y.shape
            loss = loss_fn(prediction, y)
            loss.backward()
            if observe:
                gradients = {
                    name: getattr(model, name).weight.grad
                    for name in ('query', 'key', 'value')
                }
                assert all(grad is not None and torch.isfinite(grad).all() for grad in gradients.values())
                assert torch.equal(model.value.weight, before)
                value_gradient = gradients['value']
                row, col = divmod(value_gradient.abs().argmax().item(), value_gradient.size(1))
                gradient = value_gradient[row, col].item()
                assert gradient != 0
                first_batch = dict(
                    gradient_norms={name: grad.norm().item() for name, grad in gradients.items()},
                    row=row, col=col, gradient=gradient, before=before[row, col].item(),
                    after_backward=model.value.weight[row, col].item(),
                )
            optimizer.step()
            if observe:
                first_batch['after_step'] = model.value.weight[row, col].item()
                assert first_batch['after_step'] != first_batch['before']

        train_mse = scaled_mse(model, train)
        val_mse = scaled_mse(model, val)
        if not all(math.isfinite(value) for value in (train_mse, val_mse)):
            raise RuntimeError('训练或验证损失出现非有限值')
        history.append(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse))
        if val_mse < best_value:
            best_value, best_epoch = val_mse, epoch
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f'epoch={epoch:02d} train={train_mse:.6f} val={val_mse:.6f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return history, best_epoch, first_batch
