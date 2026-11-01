"""保存说明和绘图；导入模块不会启动实验或创建图表。"""
from pathlib import Path
import os
import tempfile

# 使用可写临时缓存，不在用户主目录创建 Matplotlib 缓存。
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'ml-week07-matplotlib'))


def update_readme(section, text):
    """只替换自动结果段，保留人工维护的运行说明。"""
    path = Path(__file__).resolve().parent/'README.md'
    content = path.read_text(encoding='utf-8')
    start, end = f'<!-- {section}_START -->', f'<!-- {section}_END -->'
    before, rest = content.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before+start+'\n'+text.strip()+'\n'+end+after, encoding='utf-8')


def plot_results(result_dir, history, predictions, chosen, best_epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    # 使用英文图例，避免不同系统缺少中文字体造成方框。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, ('lstm_h16', 'lstm_h32')):
        rows = history[history.model == name]
        ax.plot(rows.epoch, rows.train_mse, label='Train')
        ax.plot(rows.epoch, rows.val_mse, label='Validation')
        epoch = best_epochs[name]
        value = rows.loc[rows.epoch == epoch, 'val_mse'].iloc[0]
        ax.axvline(epoch, color='gray', linestyle='--', label=f'Best epoch: {epoch}')
        ax.scatter([epoch], [value], color='black', zorder=4)
        ax.set(title=name, xlabel='Epoch', ylabel='MSE (standardized scale)')
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(result_dir/'loss_curves.png', dpi=160)
    plt.close(fig)

    week = predictions[(predictions.split == 'val') &
                       (predictions.target_time >= pd.Timestamp('2017-06-26')) &
                       (predictions.target_time < pd.Timestamp('2017-07-03'))]
    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    truth = week[week.model == 'last_value'].sort_values('target_time')
    assert len(truth) == 168
    ax.plot(truth.target_time, truth.y_true, color='black', linewidth=1.7, label='True OT')
    for name in ('last_value', 'lag_linear', chosen):
        rows = week[week.model == name].sort_values('target_time')
        assert rows.target_time.tolist() == truth.target_time.tolist()
        ax.plot(rows.target_time, rows.y_pred, linewidth=1, alpha=.85, label=name)
    ax.set(title='Next-hour OT prediction: first validation week',
           xlabel='Target time', ylabel='OT (original scale)')
    ax.legend(ncol=4)
    ax.grid(alpha=.2)
    fig.savefig(result_dir/'validation_predictions.png', dpi=160)
    plt.close(fig)
