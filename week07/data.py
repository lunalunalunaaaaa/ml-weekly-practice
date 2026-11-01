"""复用第六周窗口逻辑；只在原始训练时段拟合 LSTM 标准化统计量。"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from week06.data import load_data, make_windows, split_by_target

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'week06' / 'data' / 'ETTh1.csv'
RESULTS = Path(__file__).resolve().parent / 'results'
BOUNDARIES = {
    'train': ('2016-07-01', '2017-06-26'),
    'val': ('2017-06-26', '2017-10-24'),
    'test': ('2017-10-24', '2018-02-21'),
}


def prepare_data(path=DATA_PATH):
    """返回原表、原始窗口、标准化张量、训练统计量；所有结果保留目标索引。

    raw 用于基线和原始尺度评价；x/y 用于 LSTM，避免误把两种预处理混用。
    """
    df = load_data(path)
    if len(df) < 14400:
        raise ValueError('至少需要 14400 条小时记录')
    df = df.iloc[:14400].copy()
    X, y, indices = make_windows(df.OT.to_numpy(), 24)
    raw = split_by_target(X, y, indices, df.date, BOUNDARIES)
    # 注意：统计原始训练时段的 8640 条观测，不能统计重叠窗口或仅训练目标。
    train_mask = (df.date >= BOUNDARIES['train'][0]) & (df.date < BOUNDARIES['train'][1])
    observations = df.loc[train_mask, 'OT'].to_numpy(dtype=np.float64)
    mean, std = float(observations.mean()), float(observations.std(ddof=0))
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError('训练标准差必须为有限正数')
    tensors = {
        split: {
            'x': torch.tensor(((s['X'] - mean) / std)[..., None], dtype=torch.float32),
            'y': torch.tensor(((s['y'] - mean) / std)[:, None], dtype=torch.float32),
        }
        for split, s in raw.items()
    }
    assert [len(raw[k]['y']) for k in BOUNDARIES] == [8616, 2880, 2880]
    # 核对首个验证目标：前 24 个小时必须恰好对应原始序列，且目标不进入输入。
    t = int(raw['val']['target_index'][0])
    assert t == 8640
    np.testing.assert_array_equal(raw['val']['X'][0], df.OT.iloc[t-24:t])
    np.testing.assert_allclose(tensors['val']['x'][0, :, 0].numpy(),
                               (df.OT.iloc[t-24:t].to_numpy()-mean)/std, rtol=1e-6, atol=1e-7)
    assert raw['val']['y'][0] == df.OT.iloc[t]
    return {'df': df, 'raw': raw, 'tensors': tensors, 'mean': mean, 'std': std}
