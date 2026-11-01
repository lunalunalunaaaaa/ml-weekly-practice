"""实践一入口：共享标量参数的五步 RNN，仅反向一次，不训练。"""
from pathlib import Path
import math
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from week06.data import load_data
from week07.data import DATA_PATH, RESULTS
from week07.reporting import update_readme


def trace_rnn(z, order, parameters, dates, case):
    """返回保持计算图的末端预测和普通数值记录；记录不会接回状态计算。"""
    wx, wh, b = parameters
    h = z.new_zeros(())
    rows = []
    for step, source in enumerate(order, 1):
        previous = h
        input_term = wx * z[source]
        memory_term = wh * previous
        h = torch.tanh(input_term + memory_term + b)
        rows.append(dict(case=case, step=step, source_index=source,
                         source_time=dates.iloc[source], z=z[source].item(),
                         h_prev=previous.item(), input_term=input_term.item(),
                         memory_term=memory_term.item(), h=h.item(), input_grad=None))
    return 0.8*h + 0.1, rows


def main():
    df = load_data(DATA_PATH).iloc[:6]
    u = df.OT.to_numpy(dtype=float)
    values = (u-u[0])/10
    # 已变换输入创建为独立叶张量；不对 u→z 的预处理求导。
    z = torch.tensor(values[:5], dtype=torch.float64, requires_grad=True)
    target = torch.tensor(values[5], dtype=torch.float64)
    parameters = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (.6, .5, 0.)]
    prediction, original = trace_rnn(z, range(5), parameters, df.date, 'original')
    loss = (prediction-target).square()
    loss.backward()  # 全实践唯一一次 backward；没有优化器或参数更新。
    for row, gradient in zip(original, z.grad.tolist()):
        row['input_grad'] = gradient
    # 独立使用 Python 数值和 math.tanh 复核第 2 步。
    manual_h1 = math.tanh(.6*values[0])
    manual_h2 = math.tanh(.6*values[1] + .5*manual_h1)
    np.testing.assert_allclose(manual_h2, original[1]['h'], rtol=0, atol=1e-14)
    with torch.no_grad():
        reordered_prediction, reordered = trace_rnn(
            z, [0, 3, 2, 1, 4], parameters, df.date, 'reordered')
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(original+reordered).to_csv(RESULTS/'rnn_trace.csv', index=False)
    gradients = {name: p.grad.item() for name, p in zip(('wx', 'wh', 'b'), parameters)}
    report = f'''原顺序预测 `{prediction.item():.12f}`，目标 `{target.item():.12f}`，平方损失 `{loss.item():.12f}`。

参数梯度：`dL/dwx={gradients['wx']:.12f}`、`dL/dwh={gradients['wh']:.12f}`、`dL/db={gradients['b']:.12f}`。

第 2 步独立核对：`h1=tanh(0.6×{values[0]:.9f})={manual_h1:.9f}`，
`h2=tanh(0.6×{values[1]:.9f}+0.5×h1)={manual_h2:.9f}`，与张量计算一致。

顺序对照的预测为 `{reordered_prediction.item():.12f}`。差异从第 2 步进入：
原输入 `{original[1]['z']:.9f}`，对照输入 `{reordered[1]['z']:.9f}`。
到最后一步两组当前输入相同，但上一状态分别为 `{original[-1]['h_prev']:.9f}`、
`{reordered[-1]['h_prev']:.9f}`，通过 `wh*h_prev` 影响末状态。

五个输入梯度依次为 `{z.grad.tolist()}`，记录在原顺序的五行中。
本例第一个输入虽为零，仍有非零梯度；输入数值和损失敏感程度不是同一概念。
`z1→h1→h2→h3→h4→h5→预测→损失` 是早期输入的依赖路径。
同一 `wh` 在各步使用，其梯度汇总多处贡献；首步 `h0=0`，该处贡献为零。
末状态通过固定的 `0.8*h5+0.1` 连接预测。未更新任何参数，不能把本例称为训练结果。'''
    update_readme('RNN', report)
    print(report)
    print('已保存', RESULTS/'rnn_trace.csv')


if __name__ == '__main__':
    main()
