"""从仓库根目录运行 python -m week08.run。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from week06.evaluation import regression_metrics
from week07.data import RESULTS as WEEK07_RESULTS, prepare_data
from week07.models import OilLSTM
from week07.training import predict_scaled
from week08.experiments import compare_positions, compare_readers, plot_attention
from week08.models import OilAttention
from week08.training import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, fit

RESULTS = Path(__file__).resolve().parent / 'results'
CHECKPOINT = RESULTS / 'best_attention.pt'


def train(data):
    torch.manual_seed(SEED)
    model = OilAttention()
    history, best_epoch, first_batch = fit(
        model, data['tensors']['train'], data['tensors']['val']
    )
    pd.DataFrame(history).to_csv(RESULTS / 'history.csv', index=False)
    torch.save(dict(
        state_dict=model.state_dict(), width=model.width, window_size=24, input_size=1,
        best_epoch=best_epoch, mean=data['mean'], std=data['std'],
        seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        first_batch=first_batch,
    ), CHECKPOINT)


def evaluate(data):
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = OilAttention()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    saved_lstm = torch.load(WEEK07_RESULTS / 'selected_lstm.pt', map_location='cpu', weights_only=True)
    lstm = OilLSTM(saved_lstm['hidden_size'])
    lstm.load_state_dict(saved_lstm['state_dict'])
    for saved in (checkpoint, saved_lstm):
        assert (saved['mean'], saved['std']) == (data['mean'], data['std'])

    def restore(prediction):
        return np.asarray(prediction, dtype=np.float64).reshape(-1) * data['std'] + data['mean']

    x = data['tensors']['val']['x']
    raw = data['raw']['val']
    predictions = {
        'lstm': restore(predict_scaled(lstm, x)),
        'attention': restore(predict_scaled(model, x)),
    }
    # 全部读法在同一批验证输入上前向，各行共享同一个验证目标索引。
    batches = [compare_readers(model, xb) for xb in x.split(BATCH_SIZE)]
    for name in ('attention', 'uniform', 'last_only'):
        prediction = restore(torch.cat([batch[name] for batch in batches]))
        if name == 'attention':
            np.testing.assert_allclose(prediction, predictions[name], rtol=1e-6, atol=5e-6)
        else:
            predictions[name] = prediction
    validation = pd.DataFrame(dict(
        target_index=raw['target_index'], target_time=raw['target_time'], y_true=raw['y'],
        **predictions,
    ))
    assert validation.target_index.is_unique and len(validation) == 2880
    validation.to_csv(RESULTS / 'validation_predictions.csv', index=False)
    saved_validation = pd.read_csv(RESULTS / 'validation_predictions.csv', parse_dates=['target_time'])
    np.testing.assert_array_equal(saved_validation.target_index, raw['target_index'])
    np.testing.assert_array_equal(saved_validation.target_time.to_numpy(), raw['target_time'].to_numpy())
    np.testing.assert_allclose(saved_validation.y_true, raw['y'], rtol=0, atol=1e-12)

    metrics = pd.DataFrame([
        dict(model=name, n_samples=len(saved_validation),
             **regression_metrics(saved_validation.y_true, saved_validation[name]))
        for name in predictions
    ])
    prior = pd.read_csv(WEEK07_RESULTS / 'metrics.csv')
    prior_row = prior[(prior.split == 'val') & prior.chosen_lstm].iloc[0]
    np.testing.assert_allclose(metrics.loc[0, ['mae', 'rmse']].to_numpy(dtype=float),
                               [prior_row.mae, prior_row.rmse], rtol=1e-6)
    history = pd.read_csv(RESULTS / 'history.csv')
    np.testing.assert_array_equal(history.epoch, np.arange(1, checkpoint['epochs'] + 1))
    best = history.loc[history.val_mse.idxmin()]
    assert best.epoch == checkpoint['best_epoch']
    attention_rmse = metrics.loc[metrics.model == 'attention', 'rmse'].item()
    np.testing.assert_allclose(attention_rmse, np.sqrt(best.val_mse) * data['std'], rtol=1e-6)
    metrics.to_csv(RESULTS / 'validation_metrics.csv', index=False)

    # 实践一、二、三共用验证集的第一个真实窗口。
    index = int(raw['target_index'][0])
    input_times = data['df'].date.iloc[index-24:index]
    target_time = str(raw['target_time'][0])
    x0 = x[:1]
    with torch.no_grad():
        trace = model.attend(x0)
        row = trace.weights[0, -1]
        torch.testing.assert_close(row.sum(), torch.tensor(1.0))
        rebuilt = (row[:, None] * trace.value[0]).sum(dim=0)
        torch.testing.assert_close(rebuilt, trace.readout[0, -1])
        readers = compare_readers(model, x0)
        torch.testing.assert_close(readers['attention'], model(x0))
    plot_attention(trace, target_time, RESULTS / 'scores_and_weights.png')

    observations = dict(
        best_epoch=checkpoint['best_epoch'],
        mean=data['mean'],
        std=data['std'],
        first_batch=checkpoint['first_batch'],
        window=dict(
            target_index=index, first_time=str(input_times.iloc[0]),
            last_time=str(input_times.iloc[-1]), target_time=target_time,
            actual=float(raw['y'][0]),
            **{name: float(restore(prediction)[0]) for name, prediction in readers.items()},
        ),
        heatmap=dict(
            score_24_3=trace.scores[0, -1, 2].item(), weight_24_3=row[2].item(),
            last_row_sum=row.sum().item(),
            readout_max_difference=(rebuilt - trace.readout[0, -1]).abs().max().item(),
        ),
        position_swap=compare_positions(model, x0, raw['X'][0], input_times, restore),
    )
    (RESULTS / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
    print(metrics.to_string(index=False))
    print('验证窗口、热图、权重替换、位置交换及数值核对结果已保存。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only', action='store_true', help='载入已有最佳模型，只重跑验证和实验')
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(exist_ok=True)
    data = prepare_data()
    if not args.evaluate_only:
        train(data)
    evaluate(data)


if __name__ == '__main__':
    main()
