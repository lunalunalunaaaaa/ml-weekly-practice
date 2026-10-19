"""数据划分及标准化：仅从训练集学习统计量。"""
import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def step_data():
    # 每行对应一个样本，保留第二维，使输入和目标都是 (样本数, 1)。
    return torch.tensor([[1.], [2.]]), torch.tensor([[2.], [4.]])


def prepare_data():
    """返回三份数据以及反标准化所需的训练集统计量。

    x/y 是模型使用的 float32 张量；inputs/raw_y 保留 NumPy 数据，
    分别供 sklearn 基线和原始单位的指标计算使用。
    """
    dataset = load_diabetes(as_frame=True, scaled=False)
    # 先留出 40%，再将留出部分平分；所有模型复用相同划分。
    xt, xr, yt, yr = train_test_split(dataset.data, dataset.target,
                                     test_size=0.4, random_state=42)
    xv, xe, yv, ye = train_test_split(xr, yr, test_size=0.5, random_state=42)
    # 验证集和测试集只能 transform，不能分别 fit，否则会泄漏其分布信息。
    scaler = StandardScaler().fit(xt)
    raw_y = {k: np.asarray(v) for k, v in zip(
        ('train', 'validation', 'test'), (yt, yv, ye))}
    # 目标也做标准化，减小训练时的数值尺度；评价时再恢复原始单位。
    mean, std = float(raw_y['train'].mean()), float(raw_y['train'].std(ddof=0))
    assert std > 0
    inputs = {k: scaler.transform(v) for k, v in zip(raw_y, (xt, xv, xe))}
    x = {k: torch.tensor(v, dtype=torch.float32) for k, v in inputs.items()}
    y = {k: torch.tensor((v - mean) / std, dtype=torch.float32).reshape(-1, 1)
         for k, v in raw_y.items()}
    return dict(x=x, y=y, inputs=inputs, raw_y=raw_y, y_mean=mean, y_std=std)


def train_loader(data):
    # 每个配置各自调用一次，生成相互独立且同种子的批次序列。
    return DataLoader(TensorDataset(data['x']['train'], data['y']['train']),
                      batch_size=32, shuffle=True, drop_last=False, num_workers=0,
                      generator=torch.Generator().manual_seed(42))
