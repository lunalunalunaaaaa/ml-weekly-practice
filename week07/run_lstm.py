"""实践二入口：数据 → 两组训练 → 验证选择 → 测试 → 保存和核对。"""
from pathlib import Path
import platform
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import sklearn

from week06.models import build_linear, last_value
from week06.evaluation import verify_linear_numerics
from week07.data import prepare_data, RESULTS
from week07.models import OilLSTM
from week07.training import fit_lstm, predict_scaled
from week07.evaluation import regression_metrics, verify_results
from week07.reporting import plot_results, update_readme


def main():
    # 小网络使用固定 CPU 线程数，避免机器核心数不同导致过多线程开销。
    torch.set_num_threads(1)
    data = prepare_data()
    raw, tensors = data['raw'], data['tensors']
    models, best_epochs, shapes, predictions = {}, {}, {}, {}
    history = []
    for hidden in (16, 32):
        name = f'lstm_h{hidden}'
        torch.manual_seed(42)  # 每组独立初始化；宽度不同不意味着初始参数相同。
        model = OilLSTM(hidden).cpu().float()
        shapes[name] = model.inspect(tensors['train']['x'][:128])
        print(name, shapes[name], flush=True)
        # fit 接口只拿训练和验证张量，测试不参与快照或配置选择。
        rows, best_epoch = fit_lstm(model, {s: tensors[s] for s in ('train', 'val')})
        history.extend(rows)
        models[name], best_epochs[name] = model, best_epoch
        for split in ('train', 'val'):
            predictions[name, split] = predict_scaled(model, tensors[split]['x'])*data['std']+data['mean']
    # 按原始尺度验证 RMSE 选择；完全相同选 hidden_size=16。
    chosen = min(models, key=lambda name: (
        regression_metrics(raw['val']['y'], predictions[name, 'val'])['rmse'],
        models[name].hidden_size))
    print('选定 LSTM:', chosen, flush=True)
    predictions[chosen, 'test'] = predict_scaled(models[chosen], tensors['test']['x'])*data['std']+data['mean']

    # 重新复用第六周实现生成一致基线；只在训练窗口上 fit，y 保留原始尺度。
    linear = build_linear().fit(raw['train']['X'], raw['train']['y'])
    for split, rows in raw.items():
        predictions['last_value', split] = last_value(rows['X'])
        predictions['lag_linear', split] = linear.predict(rows['X'])
    linear_check = verify_linear_numerics(
        linear, raw['train']['X'], raw['train']['y'],
        np.concatenate([s['X'] for s in raw.values()]),
        np.concatenate([predictions['lag_linear', s] for s in raw]))

    metric_rows, prediction_frames = [], []
    for name in ('last_value', 'lag_linear', 'lstm_h16', 'lstm_h32'):
        for split, rows in raw.items():
            if (name, split) not in predictions:
                continue  # 未选 LSTM 不产生测试预测或测试指标。
            pred = predictions[name, split]
            metric_rows.append(dict(model=name, split=split, n_samples=len(pred),
                                    hidden_size=models[name].hidden_size if name in models else None,
                                    best_epoch=best_epochs.get(name),
                                    **regression_metrics(rows['y'], pred), chosen_lstm=name == chosen))
            prediction_frames.append(pd.DataFrame(dict(
                model=name, split=split, target_index=rows['target_index'],
                target_time=rows['target_time'], y_true=rows['y'], y_pred=pred)))
    RESULTS.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    metrics = pd.DataFrame(metric_rows)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    history_df.to_csv(RESULTS/'history.csv', index=False)
    metrics.to_csv(RESULTS/'metrics.csv', index=False)
    pred_df.to_csv(RESULTS/'predictions.csv', index=False)
    # 只保存选定模型的最佳参数，附上重建模型、标准化和恢复轮次所需信息。
    checkpoint = dict(state_dict=models[chosen].state_dict(), hidden_size=models[chosen].hidden_size,
                      best_epoch=best_epochs[chosen], mean=data['mean'], std=data['std'],
                      window_size=24, input_size=1, num_layers=1,
                      selection='minimum original-scale validation RMSE; tie: hidden_size=16',
                      seed=42, epochs=40)
    checkpoint_path = RESULTS/'selected_lstm.pt'
    torch.save(checkpoint, checkpoint_path)
    # 从磁盘重建模型复核，检查保存的权重和标准化统计量确实可以复现预测。
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    restored = OilLSTM(saved['hidden_size'])
    restored.load_state_dict(saved['state_dict'])
    restored_pred = predict_scaled(restored, tensors['val']['x'])*saved['std']+saved['mean']
    np.testing.assert_array_equal(restored_pred, predictions[chosen, 'val'])
    verification = verify_results(RESULTS, data, chosen, best_epochs)
    # 图也从已落盘的记录读取，确保展示与提交的数值一致。
    plot_results(RESULTS, pd.read_csv(RESULTS/'history.csv'),
                 pd.read_csv(RESULTS/'predictions.csv', parse_dates=['target_time']), chosen, best_epochs)

    table = ['| 模型 | 时段 | 样本数 | 最佳轮次 | MAE | RMSE |',
             '|---|---|---:|---:|---:|---:|']
    for r in metrics.itertuples():
        epoch = str(int(r.best_epoch)) if pd.notna(r.best_epoch) else '—'
        table.append(f'| {r.model} | {r.split} | {r.n_samples} | {epoch} | {r.mae:.6f} | {r.rmse:.6f} |')
    shape_lines = []
    for name, s in shapes.items():
        shape_lines.append(f"- `{name}`：input `{s['input']}`，output `{s['output']}`，h_n `{s['h_n']}`，c_n `{s['c_n']}`，prediction `{s['prediction']}`；参数共 **{s['parameters']}** 个。")
    val = metrics[metrics.split == 'val'].set_index('model')
    rmse16, rmse32 = val.loc['lstm_h16', 'rmse'], val.loc['lstm_h32', 'rmse']
    last_delta = val.loc[chosen, 'rmse']-val.loc['last_value', 'rmse']
    linear_delta = val.loc[chosen, 'rmse']-val.loc['lag_linear', 'rmse']
    # 实际首条验证样本的末步表示与线性输出，便于从窗口追到一次预测。
    with torch.no_grad():
        sample = tensors['val']['x'][:1]
        output, (h, c) = models[chosen].lstm(sample)
        scaled_pred = models[chosen].head(output[:, -1, :]).item()
    first = raw['val']['target_index'][0]
    report = f'''实际环境：Python {platform.python_version()}，PyTorch {torch.__version__}，NumPy {np.__version__}，pandas {pd.__version__}，scikit-learn {sklearn.__version__}；CPU、float32、CPU 线程数 1。

原始训练时段共有 8640 个观测，均值 `μ={data['mean']:.12f}`，总体标准差 `σ={data['std']:.12f}`。
训练 / 验证 / 测试样本数：8616 / 2880 / 2880。

训练前同一批 128 条真实样本的检查：

{chr(10).join(shape_lines)}

形状依次表示 `(批量,时间,特征)`、`(批量,时间,隐藏宽度)`、`(层数×方向数,批量,隐藏宽度)`、同形状的细胞状态和每样本一个预测。
两组均通过 `output[:,-1,:] == h_n[0]` 的数值核对。`c_n` 到 `h_n` 还需经过 tanh 和输出门，不能直接将两者当作相同表示。

指标全部为原始油温尺度；两组 LSTM 均来自各自最佳验证轮次：

{chr(10).join(table)}

H=16 与 H=32 的验证 RMSE 分别为 `{rmse16:.6f}`、`{rmse32:.6f}`，按规则选定 **{chosen}**。
选定模型验证 RMSE 减去最近值为 `{last_delta:+.6f}`，减去线性回归为 `{linear_delta:+.6f}`；负值表示误差更低。
成绩差异只反映本次划分和训练设置；即使优于最近值，也可能只利用了最近几步，不能据此证明学会长距离依赖。
结构允许早期信息传入、预测对早期信息敏感、早期信息改善误差，是三个不同层次；本实验未追加消融来验证后两者。

一次真实预测：目标下标 `{first}`，时间 `{raw['val']['target_time'][0]}`；
输入为下标 `{first-24}～{first-1}`，时间 `{data['df'].date.iloc[first-24]}` 至 `{data['df'].date.iloc[first-1]}` 的 24 个真实油温。
使用训练 μ/σ 标准化后，LSTM 从零 h/c 逐步读取，形成 `(1,{models[chosen].hidden_size})` 末步隐藏表示；
其前四个分量为 `{h[0,0,:4].tolist()}`。
线性层得到标准化预测 `{scaled_pred:.9f}`，乘 σ 加 μ 后为 `{predictions[chosen, 'val'][0]:.9f}`，真实目标为 `{raw['val']['y'][0]:.9f}`。
这些隐藏分量是通过预测损失学习的表示，不能逐个解释成某一小时油温；固定宽度末态也可能弱化部分历史。

基线使用 `week06.models.last_value/build_linear` 重新生成，训练范围和目标完全相同。
线性回归按训练窗口的各列拟合标准化，目标不标准化；它不使用 LSTM 的 μ/σ。
独立数值复核：线性预测最大差 `{linear_check['max_prediction_difference']:.3g}`，系数最大差 `{linear_check['max_coefficient_difference']:.3g}`。

{verification} 模型从磁盘恢复后的全部验证预测逐项一致。测试集未用于选择，也未合并训练验证集重训。

![两组损失曲线](results/loss_curves.png)

![验证首周预测](results/validation_predictions.png)'''
    update_readme('LSTM', report)
    print(metrics.to_string(index=False))
    print(verification)
    print('结果已保存到', RESULTS)


if __name__ == '__main__':
    main()
