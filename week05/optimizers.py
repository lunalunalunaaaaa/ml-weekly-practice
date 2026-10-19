"""两组实验的优化器配置。"""
from torch.optim import SGD, Adam

# 两个值分别是两套实验的学习率，并不是优化器在所有任务中的最优值。
# 学习率不同，因此结果只能比较这两套配置，不能单独证明某种算法更好。
CONFIGS = {'SGD': 0.01, 'Adam': 0.001}


def make_optimizer(parameters, name, lr):
    """将模型参数交给优化器；梯度由 backward 计算，参数由 step 更新。"""
    # SGD 直接沿负梯度方向更新；这里关闭动量和权重衰减，便于理解。
    if name == 'SGD':
        return SGD(parameters, lr=lr, momentum=0, weight_decay=0)
    if name == 'Adam':
        # Adam 维护梯度的一阶、二阶历史估计，其余参数使用 PyTorch 默认值。
        return Adam(parameters, lr=lr, weight_decay=0)
    raise ValueError(f'Unknown optimizer: {name}')
