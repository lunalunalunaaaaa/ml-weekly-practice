"""运行 python -m week10.run_practice6；--evaluate-only 载入最佳模型并复算验证。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from week06.evaluation import regression_metrics
from week07.data import prepare_data
from week07.training import predict_scaled
from week10.models import OilEncoder
from week10.training import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, fit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / 'results'
CHECKPOINT = RESULTS / 'best_encoder.pt'


@torch.no_grad()
def inspect_window(model, x):
    trace = model.encode(x)
    shapes = {name: list(value.shape) for name, value in zip(trace._fields, trace)}
    for name in ('X', 'M', 'H', 'F', 'E'):
        assert shapes[name] == [1, 24, 16]
    assert shapes['last'] == [1, 16] and shapes['prediction'] == [1, 1]
    # 核对两处残差实际连接的是 X 和 H。
    torch.testing.assert_close(trace.H, model.norm1(trace.X + trace.M))
    torch.testing.assert_close(trace.E, model.norm2(trace.H + trace.F))
    torch.testing.assert_close(trace.last, trace.E[:, 23, :])
    torch.testing.assert_close(model(x), trace.prediction)
    return shapes, trace.prediction.item()


def train(data):
    torch.manual_seed(SEED)
    model = OilEncoder()  # 整个模型从头训练，不载入第八周参数。
    assert not torch.equal(model.attention.head1.query.weight, model.attention.head2.query.weight)
    shapes, _ = inspect_window(model, data['tensors']['val']['x'][:1])
    print(f'训练前接线核对：{shapes}', flush=True)
    history, best_epoch, first_batch = fit(model, data['tensors']['train'], data['tensors']['val'])
    pd.DataFrame(history).to_csv(RESULTS / 'practice6_history.csv', index=False)
    torch.save({
        'state_dict': model.state_dict(), 'best_epoch': best_epoch,
        'best_val_mse': history[best_epoch - 1]['val_mse'],
        'mean': data['mean'], 'std': data['std'], 'seed': SEED,
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE,
        'window_size': 24, 'width': 16, 'heads': 2, 'head_width': 8,
        'feed_forward_width': 32, 'use_causal_mask': False,
        'first_batch': first_batch,
    }, CHECKPOINT)


def evaluate(data):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError('没有最佳模型，请先运行 python -m week10.run_practice6')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert (checkpoint['mean'], checkpoint['std']) == (data['mean'], data['std'])
    model = OilEncoder()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    x, raw = data['tensors']['val']['x'], data['raw']['val']
    scaled = predict_scaled(model, x)
    predicted = scaled * data['std'] + data['mean']
    # 目标也从标准化空间还原，先核对其与原始目标一致。
    restored_target = data['tensors']['val']['y'].numpy().ravel().astype(np.float64) * data['std'] + data['mean']
    np.testing.assert_allclose(restored_target, raw['y'], rtol=0, atol=2e-6)

    # 复用以前保存的逐条预测；先验证目标下标、时间、真值，确保同一批验证目标。
    prior_path = ROOT / 'week08/results/validation_predictions.csv'
    if not prior_path.is_file():
        raise FileNotFoundError(f'缺少以前模型的验证预测：{prior_path}；先运行 python -m week08.run --evaluate-only')
    prior = pd.read_csv(prior_path, parse_dates=['target_time'])
    assert len(prior) == len(raw['y']) == 2880 and prior.target_index.is_unique
    np.testing.assert_array_equal(prior.target_index, raw['target_index'])
    np.testing.assert_array_equal(prior.target_time.to_numpy(), raw['target_time'].to_numpy())
    np.testing.assert_allclose(prior.y_true, raw['y'], rtol=0, atol=1e-12)
    predictions = pd.DataFrame({
        'target_index': raw['target_index'], 'target_time': raw['target_time'],
        'y_true': raw['y'], 'last_value': raw['X'][:, -1],
        'lstm': prior.lstm.to_numpy(), 'single_head': prior.attention.to_numpy(),
        'encoder': predicted,
    })
    predictions.to_csv(RESULTS / 'practice6_validation_predictions.csv', index=False)
    # 从落盘预测重算误差，避免汇总与逐条记录不一致。
    saved = pd.read_csv(RESULTS / 'practice6_validation_predictions.csv')
    sources = {
        'last_value': 'validation window last observation',
        'lstm': 'week08/results/validation_predictions.csv:lstm (week07 selected LSTM)',
        'single_head': 'week08/results/validation_predictions.csv:attention',
        'encoder': f'week10/results/best_encoder.pt; epoch {checkpoint["best_epoch"]}',
    }
    metrics = pd.DataFrame([
        {'model': name, 'n_samples': len(saved),
         **regression_metrics(saved.y_true.to_numpy(), saved[name].to_numpy()), 'source': source}
        for name, source in sources.items()
    ])
    metrics.to_csv(RESULTS / 'practice6_validation_metrics.csv', index=False)
    history = pd.read_csv(RESULTS / 'practice6_history.csv')
    np.testing.assert_array_equal(history.epoch, np.arange(1, checkpoint['epochs'] + 1))
    best = history.loc[history.val_mse.idxmin()]
    assert int(best.epoch) == checkpoint['best_epoch']
    np.testing.assert_allclose(best.val_mse, checkpoint['best_val_mse'], rtol=1e-10)
    encoder_rmse = metrics.loc[metrics.model == 'encoder', 'rmse'].item()
    np.testing.assert_allclose(encoder_rmse, np.sqrt(best.val_mse) * data['std'], rtol=1e-6)

    shapes, single_scaled = inspect_window(model, x[:1])
    np.testing.assert_allclose(single_scaled, scaled[0], rtol=1e-5, atol=1e-6)
    index = int(raw['target_index'][0])
    observation = {
        'best_epoch': checkpoint['best_epoch'], 'best_val_mse': checkpoint['best_val_mse'],
        'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
        'seed': checkpoint['seed'], 'device': 'cpu', 'with_position': True,
        'use_causal_mask': False, 'shapes': shapes, 'first_batch': checkpoint['first_batch'],
        'mean': data['mean'], 'std': data['std'],
        'window': {'target_index': index, 'target_time': str(raw['target_time'][0]),
                   'first_time': str(data['df'].date.iloc[index - 24]),
                   'last_time': str(data['df'].date.iloc[index - 1]),
                   'standardized_prediction': single_scaled,
                   'prediction': single_scaled * data['std'] + data['mean'],
                   'actual': float(raw['y'][0])},
    }
    (RESULTS / 'practice6_observations.json').write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(metrics.to_string(index=False))
    print(f'最佳轮次：{checkpoint["best_epoch"]}；单窗口记录：{observation["window"]}')
    print(f'结果保存于：{RESULTS}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only', action='store_true', help='载入最佳模型，只验证和预测')
    args = parser.parse_args()
    torch.set_num_threads(1)
    RESULTS.mkdir(exist_ok=True)
    data = prepare_data()
    if not args.evaluate_only:
        train(data)
    evaluate(data)


if __name__ == '__main__':
    main()
