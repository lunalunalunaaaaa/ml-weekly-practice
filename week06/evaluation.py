"""原始油温尺度的误差，以及落盘结果的独立核对。"""
import numpy as np
import pandas as pd


# 【函数作用】比较预测与真实答案，返回 mae/rmse 两项误差。
# 输入：相同长度的一维真实值和预测值。输出：{"mae":数值,"rmse":数值}。
# 两项都保留原始油温尺度，越小越好；这里不训练模型。
def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("真实值和预测值必须是非空、同形状的一维数组")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("真实值或预测值含无效数值")
    # 【核心代码】先计算误差；正值偏高，负值偏低。
    error = y_pred - y_true
    # MAE = 平均绝对误差；RMSE = 误差平方的平均值再开根号。
    # 平方会放大大误差，因此 RMSE 对少量较大的失误更敏感。
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean()))}


# 【函数作用】核对保存后的详细预测和指标汇总是否一致。
# 输入：两份 CSV 路径、原始数据、划分结果、模型顺序与选定方法。
# 输出：通过核对的预测行数和指标行数；任何不一致都会报错。
# 阅读重点：按原始下标核对时间/真值，再从保存的预测重新计算误差。
def verify_saved_results(prediction_path, metric_path, raw_df, splits, model_order, selected):
    """重新读取文件，核对每个目标、真值、时间、重复项和全部汇总指标。"""
    pred = pd.read_csv(prediction_path, parse_dates=["target_time"])
    metrics = pd.read_csv(metric_path)
    # 【结构检查】必要列、重复项、总行数和模型名称必须正确。
    assert list(pred.columns) == ["model", "split", "target_index", "target_time", "y_true", "y_pred"]
    assert not pred.duplicated(["model", "target_index"]).any()
    assert not metrics.duplicated(["model", "split"]).any()
    assert len(metrics) == len(model_order) * len(splits)
    assert len(pred) == len(model_order) * sum(len(s["y"]) for s in splits.values())
    assert set(pred["model"]) == set(model_order)
    for model in model_order:
        for split, data in splits.items():
            # 【核心代码 1】取出“一个模型 + 一个时段”的记录，并按原始目标排序。
            rows = pred[(pred.model == model) & (pred.split == split)].sort_values("target_index")
            np.testing.assert_array_equal(rows.target_index.to_numpy(), data["target_index"])
            # 【核心代码 2】回查原始 CSV，确认预测时间与真实答案没有错位。
            expected = raw_df.iloc[data["target_index"]]
            np.testing.assert_array_equal(rows.target_time.to_numpy(), expected.date.to_numpy())
            np.testing.assert_allclose(rows.y_true, expected.OT, rtol=0, atol=1e-12)
            # 【核心代码 3】用落盘的 y_true/y_pred 重算指标，而非直接相信原汇总。
            actual = regression_metrics(rows.y_true.to_numpy(), rows.y_pred.to_numpy())
            summary = metrics[(metrics.model == model) & (metrics.split == split)].iloc[0]
            assert int(summary.n_samples) == len(rows)
            assert bool(summary.selected) == (model == selected)
            for key in ("mae", "rmse"):
                np.testing.assert_allclose(actual[key], summary[key], rtol=1e-10, atol=1e-12)
    return {"verified_prediction_rows": len(pred), "verified_metric_rows": len(metrics)}


# 【函数作用】独立复核线性运算准确性；属于额外检查，可在主流程学完后读。
# 输入：已训练管道、训练输入/答案、全部窗口和对应预测。
# 输出：两种计算方式的最大差异；不改变原模型，也不参与选参数。
# 核心分两步：逐项乘加复核预测，增广最小二乘复核权重。
def verify_linear_numerics(model, train_X, train_y, all_X, saved_prediction):
    """用不经过 BLAS matmul 的逐项求和复核预测，并用增广最小二乘复核拟合。

    本机 NumPy/Accelerate 会对有限、正确的矩阵结果发出浮点警告。
    保留原警告，使用独立数值路径检查结果，而不是直接忽略它。
    """
    from scipy.linalg import lstsq
    scaler, estimator = model.steps[0][1], model.steps[-1][1]
    # 【核心代码 1】只使用训练好的标准化器变换输入，不重新 fit。
    Z = scaler.transform(all_X)
    # 独立计算每一行的 b + Σw[j]*z[j]，与原管道预测比较。
    reference_prediction = np.einsum("ij,j->i", Z, estimator.coef_, optimize=False) + estimator.intercept_
    np.testing.assert_allclose(saved_prediction, reference_prediction, rtol=1e-10, atol=1e-10)
    # 【核心代码 2】只用训练数据重建相同的最小二乘问题，核对已学权重。
    train_Z = scaler.transform(train_X)
    alpha = getattr(estimator, "alpha", 0.0)
    centered_Z = train_Z - train_Z.mean(axis=0)
    centered_y = train_y - train_y.mean()
    # 增广数据 [Z; sqrt(alpha)*I]，对应平方误差 + alpha*权重平方和。
    # y 与 Z 中心化处理截距；普通回归 alpha=0，不增加权重约束。
    A = np.vstack([centered_Z, np.sqrt(alpha) * np.eye(train_Z.shape[1])])
    b = np.concatenate([centered_y, np.zeros(train_Z.shape[1])])
    reference_coef = lstsq(A, b)[0]
    np.testing.assert_allclose(estimator.coef_, reference_coef, rtol=1e-8, atol=1e-8)
    return {"max_prediction_difference": float(np.max(np.abs(saved_prediction - reference_prediction))),
            "max_coefficient_difference": float(np.max(np.abs(estimator.coef_ - reference_coef)))}
