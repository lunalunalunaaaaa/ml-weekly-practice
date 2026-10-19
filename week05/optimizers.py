"""两组实验的优化器配置。"""
from torch.optim import SGD, Adam

CONFIGS = {'SGD': 0.01, 'Adam': 0.001}


def make_optimizer(parameters, name, lr):
    if name == 'SGD':
        return SGD(parameters, lr=lr, momentum=0, weight_decay=0)
    if name == 'Adam':
        return Adam(parameters, lr=lr, weight_decay=0)
    raise ValueError(f'Unknown optimizer: {name}')
