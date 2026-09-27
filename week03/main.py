"""第三周实践一：多特征线性回归与 Ridge 对照。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
RIDGE_ALPHAS = (1, 100)


def evaluate(y_true, y_pred):
    """返回 MAE 和 RMSE；二者都是越小越好。"""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def add_metrics(rows, model_name, split_name, y_true, y_pred, selected=False):
    """向指标结果列表中加入一行。"""
    scores = evaluate(y_true, y_pred)
    rows.append(
        {
            "model": model_name,
            "split": split_name,
            "MAE": scores["MAE"],
            "RMSE": scores["RMSE"],
            "selected_ridge": selected,
        }
    )


def main():
    # scaled=False 保留原始量纲，后续由我们只使用训练集统计量做标准化。
    diabetes = load_diabetes(as_frame=True, scaled=False)
    x = diabetes.data.copy()  # 只包含 10 个特征，不包含 target
    y = diabetes.target.copy()
    feature_names = x.columns.tolist()

    # 与第二周一致：先分出 40% 临时集，再将临时集平分为验证集和测试集。
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    # StandardScaler 只在训练集上拟合；验证集和测试集复用训练集参数。
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    bmi_index = feature_names.index("bmi")
    x_train_bmi = x_train_scaled[:, [bmi_index]]
    x_val_bmi = x_val_scaled[:, [bmi_index]]
    x_test_bmi = x_test_scaled[:, [bmi_index]]

    mean_baseline = y_train.mean()
    single_model = LinearRegression().fit(x_train_bmi, y_train)
    multi_model = LinearRegression().fit(x_train_scaled, y_train)
    ridge_models = {
        alpha: Ridge(alpha=alpha).fit(x_train_scaled, y_train)
        for alpha in RIDGE_ALPHAS
    }

    full_inputs = {
        "train": (x_train_scaled, y_train),
        "validation": (x_val_scaled, y_val),
        "test": (x_test_scaled, y_test),
    }
    bmi_inputs = {
        "train": (x_train_bmi, y_train),
        "validation": (x_val_bmi, y_val),
        "test": (x_test_bmi, y_test),
    }

    # 先用验证集 RMSE 在两组 Ridge 中选 alpha，测试集不参与选择。
    ridge_val_rmse = {
        alpha: evaluate(y_val, model.predict(x_val_scaled))["RMSE"]
        for alpha, model in ridge_models.items()
    }
    selected_alpha = min(RIDGE_ALPHAS, key=lambda alpha: (ridge_val_rmse[alpha], alpha))

    metrics_rows = []
    # 五组模型都记录训练和验证结果。
    for split_name in ("train", "validation"):
        _, y_part = full_inputs[split_name]
        add_metrics(
            metrics_rows,
            "mean_baseline",
            split_name,
            y_part,
            np.full(len(y_part), mean_baseline),
        )

        x_bmi, y_part = bmi_inputs[split_name]
        add_metrics(
            metrics_rows,
            "single_feature_linear_regression",
            split_name,
            y_part,
            single_model.predict(x_bmi),
        )

        x_all, y_part = full_inputs[split_name]
        add_metrics(
            metrics_rows,
            "multi_feature_linear_regression",
            split_name,
            y_part,
            multi_model.predict(x_all),
        )
        for alpha, model in ridge_models.items():
            add_metrics(
                metrics_rows,
                f"ridge_alpha_{alpha}",
                split_name,
                y_part,
                model.predict(x_all),
                selected=alpha == selected_alpha,
            )

    # 测试集只评价基线、单特征、多特征和验证集选出的 Ridge。
    x_test_all, y_test_part = full_inputs["test"]
    add_metrics(
        metrics_rows,
        "mean_baseline",
        "test",
        y_test_part,
        np.full(len(y_test_part), mean_baseline),
    )
    add_metrics(
        metrics_rows,
        "single_feature_linear_regression",
        "test",
        y_test_part,
        single_model.predict(x_test_bmi),
    )
    add_metrics(
        metrics_rows,
        "multi_feature_linear_regression",
        "test",
        y_test_part,
        multi_model.predict(x_test_all),
    )
    add_metrics(
        metrics_rows,
        f"ridge_alpha_{selected_alpha}",
        "test",
        y_test_part,
        ridge_models[selected_alpha].predict(x_test_all),
        selected=True,
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

    # 三组多特征模型的权重。输入已经标准化，因此权重单位可直接对照。
    weights_df = pd.DataFrame(
        {
            "feature": feature_names,
            "linear_regression": multi_model.coef_,
            "ridge_alpha_1": ridge_models[1].coef_,
            "ridge_alpha_100": ridge_models[100].coef_,
        }
    )
    weights_df.to_csv(RESULTS_DIR / "weights.csv", index=False, encoding="utf-8-sig")

    # 五组模型的训练/验证 RMSE 分组柱状图。
    model_order = [
        "mean_baseline",
        "single_feature_linear_regression",
        "multi_feature_linear_regression",
        "ridge_alpha_1",
        "ridge_alpha_100",
    ]
    labels = ["Mean baseline", "BMI linear", "All-feature linear", "Ridge α=1", "Ridge α=100"]
    rmse_table = (
        metrics_df[metrics_df["split"].isin(["train", "validation"])]
        .pivot(index="model", columns="split", values="RMSE")
        .reindex(model_order)
    )
    positions = np.arange(len(model_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions - width / 2, rmse_table["train"], width, label="Train")
    ax.bar(positions + width / 2, rmse_table["validation"], width, label="Validation")
    ax.set_ylabel("RMSE")
    ax.set_title("Training and Validation RMSE")
    ax.set_xticks(positions, labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "train_validation_rmse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 三组多特征模型的权重对比图。
    fig, ax = plt.subplots(figsize=(11, 5))
    weight_width = 0.25
    feature_positions = np.arange(len(feature_names))
    ax.bar(feature_positions - weight_width, multi_model.coef_, weight_width, label="Linear regression")
    ax.bar(feature_positions, ridge_models[1].coef_, weight_width, label="Ridge α=1")
    ax.bar(feature_positions + weight_width, ridge_models[100].coef_, weight_width, label="Ridge α=100")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coefficient (standardized input)")
    ax.set_title("Coefficient Comparison")
    ax.set_xticks(feature_positions, feature_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "weights_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"数据划分：训练集 {len(y_train)}，验证集 {len(y_val)}，测试集 {len(y_test)}")
    print(f"根据验证集 RMSE 选定的 Ridge alpha：{selected_alpha}")
    print("\n指标结果：")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n结果已保存到：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
