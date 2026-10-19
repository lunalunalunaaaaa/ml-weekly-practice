"""实践二入口：连接组件，不在入口实现训练或绘图。"""
import copy
import torch
from data import prepare_data, train_loader
from models import mlp, forward_shapes
from optimizers import CONFIGS, make_optimizer
from training import fit_mlp, fit_baselines
from evaluation import evaluate_mlp, evaluate_baseline
from reporting import save_experiment, RESULTS


def main():
    torch.set_num_threads(1)
    data = prepare_data()
    torch.manual_seed(42)
    initial = mlp()  # CPU、float32，正常随机初始化。
    print('Split sizes:', {k: len(v) for k, v in data['raw_y'].items()})
    print('Width 16:', forward_shapes(initial, data['x']['train'][:32]))
    models, history, rows = {}, [], []
    baselines = fit_baselines(data)
    for name, baseline in baselines.items():
        row = dict(model=name, learning_rate=None, best_epoch=None,
                   selected_mlp=False, target_scale='original')
        for split in ('train', 'validation'):
            row.update({f'{split}_{k}': v for k, v in evaluate_baseline(baseline, data, split).items()})
        rows.append(row)
    for name, lr in CONFIGS.items():
        model = copy.deepcopy(initial)
        optimizer = make_optimizer(model.parameters(), name, lr)
        records, best = fit_mlp(model, optimizer, train_loader(data), data, name, lr)
        models[name] = model
        history.extend(records)
        row = dict(model=name, learning_rate=lr, best_epoch=best,
                   selected_mlp=False, target_scale='original')
        for split in ('train', 'validation'):
            row.update({f'{split}_{k}': v for k, v in evaluate_mlp(model, data, split).items()})
        rows.append(row)
    selected = min(rows[2:], key=lambda r: (r['validation_rmse'], r['model'] != 'SGD'))['model']
    for row in rows:
        name = row['model']
        row['selected_mlp'] = name == selected
        row['test_mae'] = row['test_rmse'] = None
        if name in baselines or name == selected:
            scores = (evaluate_baseline(baselines[name], data, 'test') if name in baselines
                      else evaluate_mlp(models[name], data, 'test'))
            row.update({f'test_{k}': v for k, v in scores.items()})
    save_experiment(history, rows)
    # 仅前向检查宽度 8，不重新训练，也不改变正式实验。
    print('Width 8:', forward_shapes(mlp(hidden_dim=8), data['x']['train'][:32]))
    print('Selected MLP:', selected)
    for row in rows:
        print(row)
    print('Results:', RESULTS)


if __name__ == '__main__':
    main()
