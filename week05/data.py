"""数据划分及标准化：仅从训练集学习统计量。"""
import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def step_data():
    return torch.tensor([[1.], [2.]]), torch.tensor([[2.], [4.]])


def prepare_data():
    dataset = load_diabetes(as_frame=True, scaled=False)
    xt, xr, yt, yr = train_test_split(dataset.data, dataset.target,
                                     test_size=0.4, random_state=42)
    xv, xe, yv, ye = train_test_split(xr, yr, test_size=0.5, random_state=42)
    scaler = StandardScaler().fit(xt)
    raw_y = {k: np.asarray(v) for k, v in zip(
        ('train', 'validation', 'test'), (yt, yv, ye))}
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
