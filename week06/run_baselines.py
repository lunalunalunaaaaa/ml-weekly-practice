"""实践二入口：集中设置实验配置，只用训练集 fit、验证集选择。"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from data import load_data, make_windows, split_by_target
from models import last_value, window_mean, build_linear, build_ridge
from evaluation import regression_metrics, verify_saved_results, verify_linear_numerics
from reporting import save_csv, plot_tuning, plot_series, plot_validation
import matplotlib

# 【阅读路线】main 的核心是：数据准备 → 训练 → 验证调参 → 锁定选择 → 测试评价。
# 搜索“核心步骤”可直接定位；数值检查、绘图、环境记录可稍后阅读。
# 【集中配置】保证各模型使用同一数据范围、24小时输入、单步目标与时间划分。
BASE_DIR = Path(__file__).resolve().parent
N_HOURS = 14400
WINDOW_SIZE = 24
ROUTE = "ridge"
# 运行前提出：0.1 近似弱约束，10 中等，1000 强约束；只做本轮比较。
ALPHAS = (0.1, 10.0, 1000.0)
# 各时段均为 [起点,终点)，归组依据是目标时间，不是窗口起点。
BOUNDARIES = {
    "train": ("2016-07-01 00:00", "2017-06-26 00:00"),
    "val": ("2017-06-26 00:00", "2017-10-24 00:00"),
    "test": ("2017-10-24 00:00", "2018-02-21 00:00"),
}
# 同分时按此顺序选择；最终只保留这四种方法的测试预测。
MODEL_ORDER = ["last_value", "window_mean", "lag_linear", "ridge_tuned"]


# 【函数作用】组织实践二完整实验，调用四个组件，最终保存CSV、PNG和运行摘要。
# 输入：原始 CSV 和文件顶部的配置。输出：results 内正式实验结果及终端指标。
# 核心原则：只在训练集 fit，只用验证集选方案，锁定方案后才评价测试集。
def main():
    if ROUTE != "ridge" or len(set(ALPHAS)) != 3 or min(ALPHAS) <= 0:
        raise ValueError("本实验使用 Ridge 路线，需三个不同的正 alpha")
    results = BASE_DIR / "results"
    results.mkdir(exist_ok=True)
    data_path = BASE_DIR / "data" / "ETTh1.csv"
    # 【核心步骤 1：准备数据】取前14400条，仅历史 OT 作为模型输入。
    raw = load_data(data_path)
    if len(raw) < N_HOURS:
        raise ValueError("原始记录不足 14,400 条")
    df = raw.iloc[:N_HOURS].copy()
    series = df.OT.to_numpy(dtype=float)
    # 【核心步骤 2：样本与划分】先构造窗口，再按目标时间分组。
    # 验证/测试起点仍可使用边界之前已经测到的24条真实历史。
    X, y, target_index = make_windows(series, WINDOW_SIZE)
    splits = split_by_target(X, y, target_index, df.date, BOUNDARIES)
    assert {k: len(v["y"]) for k, v in splits.items()} == {"train": 8616, "val": 2880, "test": 2880}
    for name, part in splits.items():
        t = int(part["target_index"][0])
        print(f"{name}: {len(part['y'])} 条；首个输入 {df.date.iloc[t-WINDOW_SIZE]} 至 {df.date.iloc[t-1]} → {df.date.iloc[t]}")
    # 保存第一条验证样本的全部输入，便于逐项回查原始 CSV。
    # 【辅助输出】首条验证目标及24条历史另存，方便从预测回查原始数据。
    first_val_t = int(splits["val"]["target_index"][0])
    trace = df.iloc[first_val_t-WINDOW_SIZE:first_val_t][["date", "OT"]].copy()
    trace.insert(0, "original_index", trace.index)
    trace["target_index"] = first_val_t
    trace["target_time"] = df.date.iloc[first_val_t]
    save_csv(trace, results / "sample_window_trace.csv")

    # 【核心步骤 3：训练参照】fit 同时学习训练标准化统计量、回归权重和截距。
    train, val = splits["train"], splits["val"]
    linear = build_linear()
    linear.fit(train["X"], train["y"])
    # tuning_records 保存每个配置的成绩；ridge_models 保留已训练候选，选中后直接使用。
    tuning_records, ridge_models = [], {}
    # 【核心步骤 4：三个候选的公平比较】本循环共处理4项：
    # 普通回归参照（已fit），以及 alpha=0.1/10/1000 的独立 Ridge 管道。
    # 每项只看完整训练集与验证集；这里不使用测试集。
    for label, alpha, model in [("lag_linear", None, linear)] + [("ridge", a, build_ridge(a)) for a in ALPHAS]:
        if label == "ridge":
            # 只在训练集 fit；验证集调用 predict 不会更新权重或标准化统计量。
            model.fit(train["X"], train["y"])
            ridge_models[alpha] = model
        # 同时记录训练与验证误差，后续用于比较拟合和泛化表现。
        train_score = regression_metrics(train["y"], model.predict(train["X"]))
        val_score = regression_metrics(val["y"], model.predict(val["X"]))
        # 一行代表一个配置；rmse_gap=验证RMSE-训练RMSE，不是训练轮次。
        tuning_records.append({"model": label, "param_name": "" if alpha is None else "alpha",
                               "param_value": alpha, "train_mae": train_score["mae"], "val_mae": val_score["mae"],
                               "train_rmse": train_score["rmse"], "val_rmse": val_score["rmse"],
                               "rmse_gap": val_score["rmse"] - train_score["rmse"], "chosen_candidate": False})
    # 完全相同的验证 RMSE 按更小 alpha 打破平局；普通回归不参与候选选择。
    # 【核心步骤 5：路线内部选择】[1:] 跳过普通回归，只比较三个 Ridge。
    # 先按 val_rmse，完全相同时再按更小 alpha；不按训练误差选。
    chosen = min(tuning_records[1:], key=lambda row: (row["val_rmse"], row["param_value"]))
    chosen["chosen_candidate"] = True
    best_alpha = chosen["param_value"]
    # 【核心步骤 6：四种最终方法】固定规则直接算；回归使用之前fit好的模型。
    # 这里只保存“怎样预测”，没有再次训练。
    predictors = {"last_value": last_value, "window_mean": window_mean,
                  "lag_linear": linear.predict, "ridge_tuned": ridge_models[best_alpha].predict}
    # 用四种方法的完整验证 RMSE 做总体比较。
    validation_scores = {name: regression_metrics(val["y"], predict(val["X"]))["rmse"]
                         for name, predict in predictors.items()}
    # min 在平局时保留 MODEL_ORDER 中先出现的项目。
    # selected 是四方法的总体选择，区别于 chosen_candidate 的 Ridge 内部选择。
    selected = min(MODEL_ORDER, key=lambda name: validation_scores[name])
    print(f"Ridge 选定 alpha={best_alpha:g}；四种方法总体选择：{selected}")
    # 到这里才首次预测测试集；绝不合并训练与验证重新 fit。
    # 【核心步骤 7：最终评价】方案已锁定，现在才包含测试预测。
    # 每条窗口来自当时已获得的真实历史；逐时刻单步预测，不是一次预测120天。
    predictions, metric_records = [], []
    for name in MODEL_ORDER:
        for split, part in splits.items():
            # 【预测核心】只把历史 X 交给方法；真实答案 y 仅在下一行评价时使用。
            y_pred = predictors[name](part["X"])
            score = regression_metrics(part["y"], y_pred)
            # 汇总表每行=一个模型+一个时段，共4×3=12行。
            metric_records.append({"model": name, "split": split, "n_samples": len(part["y"]),
                                   **score, "selected": name == selected})
            # 详细表每行=一个模型对一个目标的预测，保留原始下标和时间供回查。
            predictions.append(pd.DataFrame({"model": name, "split": split,
                                             "target_index": part["target_index"], "target_time": part["target_time"],
                                             "y_true": part["y"], "y_pred": y_pred}))
    # 【辅助检查】独立核对最终线性模型的数值计算，不参与调参；可稍后阅读。
    numeric_checks = {}
    for name, model in [("lag_linear", linear), ("ridge_tuned", ridge_models[best_alpha])]:
        all_predictions = np.concatenate([part["y_pred"].to_numpy() for part in predictions if part["model"].iloc[0] == name])
        numeric_checks[name] = verify_linear_numerics(model, train["X"], train["y"], X, all_predictions)
    # 【步骤 8：输出与核对】保存调参、汇总、详细预测，再从磁盘重新读回核验。
    tuning = pd.DataFrame(tuning_records)
    metrics = pd.DataFrame(metric_records)
    pred = pd.concat(predictions, ignore_index=True)
    save_csv(tuning, results / "tuning.csv")
    save_csv(metrics, results / "metrics.csv")
    save_csv(pred, results / "predictions.csv")
    verification = verify_saved_results(results / "predictions.csv", results / "metrics.csv", df, splits, MODEL_ORDER, selected)
    # 图直接使用已落盘数据，保证图表和 CSV 一致。
    # 【步骤 9：画图】绘图直接使用保存的CSV，避免图表与表格使用不同结果。
    saved_tuning = pd.read_csv(results / "tuning.csv")
    saved_pred = pd.read_csv(results / "predictions.csv", parse_dates=["target_time"])
    plot_tuning(saved_tuning, results / "tuning_comparison.png")
    plot_series(df, WINDOW_SIZE, BOUNDARIES, results / "series_split.png")
    plot_validation(saved_pred, df, MODEL_ORDER, BOUNDARIES["val"][0], "2017-07-03 00:00", results / "validation_predictions.png")
    # 【辅助记录】保存固定设置、最终选择、环境和数据摘要，供复现及核对。
    metadata = {"n_raw": len(raw), "raw_start": str(raw.date.iloc[0]), "raw_end": str(raw.date.iloc[-1]),
                "used_hours": N_HOURS, "window_size": WINDOW_SIZE, "horizon": 1,
                "route": ROUTE, "alphas": ALPHAS, "chosen_alpha": best_alpha, "selected_model": selected,
                "boundaries": BOUNDARIES, "split_samples": {k: len(v["y"]) for k, v in splits.items()},
                "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
                "verification": verification, "numeric_checks": numeric_checks}
    (results / "run_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print("落盘核对通过：", verification)


if __name__ == "__main__":
    main()
