"""从仓库根目录运行 python -m week10.run_practice4；只做前向对照，不训练。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from week07.data import prepare_data
from week08.models import OilAttention
from week10.models import AttentionHead, TwoHeadAttention

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / 'results'
CHECKPOINT = ROOT / 'week08/results/best_attention.pt'
SEED = 42


def summarize(trace):
    """只比较第 24 小时；保留完整 24 行的前向计算。"""
    return {
        'weight_max_abs_difference': (
            trace.head1.weights[0, 23] - trace.head2.weights[0, 23]
        ).abs().max().item(),
        'readout_max_abs_difference': (
            trace.head1.readout[0, 23] - trace.head2.readout[0, 23]
        ).abs().max().item(),
        'head1_last_weights': trace.head1.weights[0, 23].tolist(),
        'head2_last_weights': trace.head2.weights[0, 23].tolist(),
        'head1_last_readout': trace.head1.readout[0, 23].tolist(),
        'head2_last_readout': trace.head2.readout[0, 23].tolist(),
    }


def check_trace(model, x, trace):
    """核对权重归一化、手动加权求和及两头的拼接方向。"""
    for head, result in ((model.head1, trace.head1), (model.head2, trace.head2)):
        assert result.weights.shape == (1, 24, 24)
        assert result.readout.shape == (1, 24, 8)
        assert torch.isfinite(result.weights).all() and (result.weights >= 0).all()
        torch.testing.assert_close(result.weights.sum(dim=-1), torch.ones(1, 24))
        manual = (result.weights[0, 23, :, None] * head.value(x)[0]).sum(dim=0)
        torch.testing.assert_close(manual, result.readout[0, 23])
    assert trace.concatenated.shape == trace.output.shape == (1, 24, 16)
    torch.testing.assert_close(trace.concatenated[..., :8], trace.head1.readout)
    torch.testing.assert_close(trace.concatenated[..., 8:], trace.head2.readout)
    torch.testing.assert_close(trace.output, model.output_projection(trace.concatenated))


def plot_comparison(cases, target_time, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    hours = np.arange(1, 25)
    titles = ('Copied parameters', 'Reinitialized head 2')
    for ax, case, title in zip(axes, cases.values(), titles):
        ax.plot(hours, case['head1_last_weights'], 'o-', markersize=3, label='Head 1')
        ax.plot(hours, case['head2_last_weights'], 'x--', markersize=4, label='Head 2')
        ax.set(title=title, xlabel='Source hour in window', xlim=(1, 24),
               xticks=[1, 4, 8, 12, 16, 20, 24])
        ax.grid(alpha=0.25)
        ax.legend()
        ax.text(0.02, -0.25,
                f"max |weight difference| = {case['weight_max_abs_difference']:.8g}\n"
                f"max |readout difference| = {case['readout_max_abs_difference']:.8g}",
                transform=ax.transAxes, fontsize=9)
    maximum = max(max(case[key]) for case in cases.values()
                  for key in ('head1_last_weights', 'head2_last_weights'))
    axes[0].set(ylabel='Attention weight (query hour 24)', ylim=(0, maximum * 1.15))
    fig.suptitle(f'ETTh1 | Target {target_time} | Query hour 24')
    fig.text(0.5, 0.025, 'Each head: (1, 24, 8)  ->  Concatenate: (1, 24, 16)'
             '  ->  Output projection: (1, 24, 16)', ha='center', fontsize=9)
    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.12)
    fig.savefig(path, dpi=160)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'需要第八周最佳模型：{CHECKPOINT}；先运行 python -m week08.run')

    data = prepare_data()
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert checkpoint['mean'] == data['mean'] and checkpoint['std'] == data['std']
    previous = OilAttention()
    previous.load_state_dict(checkpoint['state_dict'])
    previous.eval()

    # 和第八周实践一画图使用相同的首个验证窗口、相同的已训练输入映射。
    # 实践一不带位置编码，这里沿用该设置，不额外加入位置。
    raw = data['raw']['val']
    target_index = int(raw['target_index'][0])
    input_times = data['df'].date.iloc[target_index - 24:target_index]
    target_time = str(raw['target_time'][0])
    x = previous.embed(data['tensors']['val']['x'][:1])
    assert x.shape == (1, 24, 16)

    model = TwoHeadAttention().eval()
    # 对照一：包括 bias 在内，复制第一头的全部参数。
    model.head2.load_state_dict(model.head1.state_dict())
    copied = model(x)
    check_trace(model, x, copied)
    torch.testing.assert_close(copied.head1.weights, copied.head2.weights, rtol=0, atol=0)
    torch.testing.assert_close(copied.head1.readout, copied.head2.readout, rtol=0, atol=0)

    # 对照二：只换第二头，X、第一头、输出投影均不变。
    model.head2 = AttentionHead().eval()
    independent = model(x)
    check_trace(model, x, independent)
    torch.testing.assert_close(copied.head1.weights, independent.head1.weights, rtol=0, atol=0)
    torch.testing.assert_close(copied.head1.readout, independent.head1.readout, rtol=0, atol=0)
    # 不断言第二组差值必须大于零：不同随机参数并不保证权重明显不同。
    cases = {'copied': summarize(copied), 'reinitialized': summarize(independent)}
    observations = {
        'seed': SEED, 'device': 'cpu', 'mode': 'eval', 'trained_heads': False,
        'embedding_checkpoint': str(CHECKPOINT.relative_to(ROOT)),
        'with_position': False,
        'mean': data['mean'], 'std': data['std'],
        'window': {'target_index': target_index, 'first_time': str(input_times.iloc[0]),
                   'last_time': str(input_times.iloc[-1]), 'target_time': target_time},
        'shapes': {'X': list(x.shape), 'A_per_head': list(independent.head1.weights.shape),
                   'Z_per_head': list(independent.head1.readout.shape),
                   'concatenated': list(independent.concatenated.shape),
                   'output': list(independent.output.shape)},
        'cases': cases,
    }
    (RESULTS / 'practice4_observations.json').write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pd.DataFrame({
        'hour': np.arange(1, 25), 'time': input_times.to_numpy(),
        'oil_temperature': raw['X'][0],
        'standardized_oil_temperature': data['tensors']['val']['x'][0, :, 0].numpy(),
    }).to_csv(RESULTS / 'practice4_window.csv', index=False)
    table = pd.DataFrame([
        {'case': name, **{key: case[key] for key in
                         ('weight_max_abs_difference', 'readout_max_abs_difference')}}
        for name, case in cases.items()
    ])
    table.to_csv(RESULTS / 'practice4_differences.csv', index=False)
    plot_comparison(cases, target_time, RESULTS / 'practice4_heads.png')
    print(f'验证目标时间：{target_time}')
    print(table.to_string(index=False))
    print(f'形状：{observations["shapes"]}')
    print('权重归一化、手动汇总、参数复制、第一头不变及拼接核对均通过。')
    print(f'结果保存于：{RESULTS}')


if __name__ == '__main__':
    main()
