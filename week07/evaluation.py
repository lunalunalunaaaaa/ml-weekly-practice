"""原始尺度指标复用第六周；落盘后核对目标覆盖范围及指标。"""
import numpy as np
import pandas as pd

from week06.evaluation import regression_metrics


def verify_results(result_dir, data, chosen, best_epochs):
    """重新读取 CSV，以逐条预测独立核对全部汇总和选择规则。"""
    pred = pd.read_csv(result_dir/'predictions.csv', parse_dates=['target_time'])
    metrics = pd.read_csv(result_dir/'metrics.csv')
    history = pd.read_csv(result_dir/'history.csv')
    expected = {(m, s) for m in ('last_value', 'lag_linear', 'lstm_h16', 'lstm_h32')
                for s in ('train', 'val')}
    expected |= {(m, 'test') for m in ('last_value', 'lag_linear', chosen)}
    assert set(zip(metrics.model, metrics.split)) == expected
    assert set(zip(pred.model, pred.split)) == expected
    assert len(metrics) == 11
    assert not metrics.duplicated(['model', 'split']).any()
    assert not pred.duplicated(['model', 'target_index']).any()
    assert len(pred) == 4*(8616+2880)+3*2880
    for row in metrics.itertuples():
        saved = pred[(pred.model == row.model) & (pred.split == row.split)].sort_values('target_index')
        raw = data['raw'][row.split]
        np.testing.assert_array_equal(saved.target_index, raw['target_index'])
        np.testing.assert_array_equal(saved.target_time.to_numpy(), raw['target_time'].to_numpy())
        np.testing.assert_allclose(saved.y_true, raw['y'], rtol=0, atol=1e-12)
        actual = regression_metrics(saved.y_true.to_numpy(), saved.y_pred.to_numpy())
        np.testing.assert_allclose([row.mae, row.rmse], [actual['mae'], actual['rmse']], rtol=1e-10)
        assert row.n_samples == len(raw['y'])
        assert row.chosen_lstm == (row.model == chosen)
        if row.model.startswith('lstm'):
            assert row.hidden_size == int(row.model.removeprefix('lstm_h'))
            assert row.best_epoch == best_epochs[row.model]
        else:
            assert pd.isna(row.hidden_size) and pd.isna(row.best_epoch)
    assert len(history) == 80 and set(history.model) == {'lstm_h16', 'lstm_h32'}
    assert set(history.scale) == {'standardized'}
    for model, rows in history.groupby('model', sort=False):
        np.testing.assert_array_equal(rows.epoch, np.arange(1, 41))
        assert np.isfinite(rows[['train_mse', 'val_mse']]).all().all()
        assert (rows[['train_mse', 'val_mse']] >= 0).all().all()
        assert (rows.hidden_size == int(model.removeprefix('lstm_h'))).all()
        assert int(rows.loc[rows.val_mse.idxmin(), 'epoch']) == best_epochs[model]
        # 恢复最佳模型后的原始尺度误差，应对应最佳轮次的标准化损失。
        for split in ('train', 'val'):
            rmse = metrics.loc[(metrics.model == model) & (metrics.split == split), 'rmse'].iloc[0]
            mse = rows.loc[rows.epoch == best_epochs[model], f'{split}_mse'].iloc[0]
            np.testing.assert_allclose(rmse, np.sqrt(mse)*data['std'], rtol=1e-6, atol=1e-7)
    candidates = metrics[(metrics.split == 'val') & metrics.model.str.startswith('lstm')]
    assert candidates.sort_values(['rmse', 'hidden_size']).iloc[0].model == chosen
    return f'逐条核对通过：{len(pred)} 条预测、{len(metrics)} 行指标、80 行训练历史；目标无重复或遗漏。'
