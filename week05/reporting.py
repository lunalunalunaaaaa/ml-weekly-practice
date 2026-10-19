"""仅保存已计算的记录和图表，不训练模型。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / 'results'


def save_step(rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / 'step_trace.csv', index=False, encoding='utf-8-sig')


def save_experiment(history, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    history, table = pd.DataFrame(history), pd.DataFrame(rows)
    history.to_csv(RESULTS / 'history.csv', index=False, encoding='utf-8-sig')
    table.to_csv(RESULTS / 'metrics.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, ('SGD', 'Adam')):
        part = history[history.config == name]
        best = int(table.loc[table.model == name, 'best_epoch'].iloc[0])
        ax.plot(part.epoch, part.train_mse, label='Train')
        ax.plot(part.epoch, part.validation_mse, label='Validation')
        ax.axvline(best, linestyle='--', color='gray', label=f'Best epoch: {best}')
        ax.set(title=f'{name}, lr={part.learning_rate.iloc[0]:g}', xlabel='Epoch',
               ylabel='MSE (standardized target)')
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'loss_curves.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    positions, width = np.arange(len(table)), 0.36
    ax.bar(positions-width/2, table.train_rmse, width, label='Train')
    ax.bar(positions+width/2, table.validation_rmse, width, label='Validation')
    ax.set_xticks(positions, table.model)
    ax.set(ylabel='RMSE (original target units)', title='MLPs evaluated at their best validation epoch')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'rmse_comparison.png', dpi=180)
    plt.close(fig)
