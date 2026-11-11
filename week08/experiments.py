"""复用同一个已训练模型，核对权重汇总和位置交换。"""
import numpy as np
import torch


@torch.no_grad()
def compare_readers(model, x):
    """复用同批次的 V 和预测层，只在 softmax 后替换末行。"""
    trace = model.attend(x)
    original = trace.weights[:, -1]
    uniform = torch.full_like(original, 1 / original.size(-1))
    last_only = torch.zeros_like(original)
    last_only[:, -1] = 1

    def read(row):
        return (row.unsqueeze(-1) * trace.value).sum(dim=1)

    original_readout = read(original)
    uniform_readout = read(uniform)
    last_readout = read(last_only)
    torch.testing.assert_close(original_readout, trace.readout[:, -1])
    torch.testing.assert_close(uniform_readout, trace.value.mean(dim=1))
    torch.testing.assert_close(last_readout, trace.value[:, -1])
    return {
        'attention': model.head(original_readout),
        'uniform': model.head(uniform_readout),
        'last_only': model.head(last_readout),
    }


@torch.no_grad()
def compare_positions(model, x, raw_window, input_times, restore):
    # 在前 23 小时选取最低和最高油温，保证交换的观测不同。
    earlier = raw_window[:-1]
    i, j = sorted((int(earlier.argmin()), int(earlier.argmax())))
    assert earlier[i] != earlier[j]
    order = torch.arange(x.size(1))
    order[[i, j]] = order[[j, i]]
    exchanged = x[:, order]

    results = {}
    for with_position in (False, True):
        original = model.attend(x, with_position=with_position)
        swapped = model.attend(exchanged, with_position=with_position)
        torch.testing.assert_close(original.query[:, -1], swapped.query[:, -1])
        if not with_position:
            torch.testing.assert_close(original.key[:, order], swapped.key)
            torch.testing.assert_close(original.value[:, order], swapped.value)
            torch.testing.assert_close(original.weights[:, -1, order], swapped.weights[:, -1])
            torch.testing.assert_close(original.readout[:, -1], swapped.readout[:, -1])

        before = model.head(original.readout[:, -1])
        after = model.head(swapped.readout[:, -1])
        if not with_position:
            torch.testing.assert_close(before, after)
        results['with_position' if with_position else 'without_position'] = dict(
            before=float(restore(before)[0]),
            after=float(restore(after)[0]),
            readout_max_difference=(original.readout[:, -1] - swapped.readout[:, -1]).abs().max().item(),
            swapped_weights_before=original.weights[0, -1, [i, j]].tolist(),
            swapped_weights_after=swapped.weights[0, -1, [i, j]].tolist(),
        )

    return dict(
        hours=[i + 1, j + 1],
        times=[str(input_times.iloc[index]) for index in (i, j)],
        temperatures=[float(raw_window[index]) for index in (i, j)],
        **results,
    )


def plot_attention(trace, target_time, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    fig = Figure(figsize=(11, 5.4), layout='constrained')
    FigureCanvasAgg(fig)
    for ax, values, title, cmap in zip(
        fig.subplots(1, 2),
        (trace.scores, trace.weights),
        ('S  |  Scaled dot-product scores', 'A  |  Attention weights'),
        ('coolwarm', 'viridis'),
    ):
        image = ax.imshow(values[0].numpy(), cmap=cmap, interpolation='nearest',
                          extent=(0.5, 24.5, 24.5, 0.5))
        ax.add_patch(Rectangle((2.5, 23.5), 1, 1, fill=False, edgecolor='white', linewidth=2))
        ax.set(title=title, xlabel='Key hour (source)', ylabel='Query hour',
               xticks=np.arange(1, 25), yticks=np.arange(1, 25))
        ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle(f'ETTh1 validation window  |  Target {target_time}')
    fig.savefig(path, dpi=160)
