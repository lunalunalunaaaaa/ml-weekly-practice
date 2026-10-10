"""第四周：PCA 回归对照与 K-means 聚类。运行后生成四份 CSV、三张 PNG。"""

import os
from pathlib import Path

# 限制默认线程数，避免 Windows + MKL 下小样本 KMeans 的内存泄漏警告。
# 用户已设置的线程配置会被保留；必须在导入数值计算库之前设置。
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")  # 保存图片，无需打开图形窗口，适合命令行复现。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_STATE = 42
PCA_DIMENSIONS = (2, 5)


def evaluate(y_true, y_pred):
    """预测误差，与输入特征的重构误差是两种不同指标。"""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_regression(inputs, targets):
    """训练四组模型；只根据验证 RMSE 选择 PCA 维数。"""
    pcas, projected, models, rows, pca_rows = {}, {}, {}, [], []
    models["mean_baseline"] = DummyRegressor(strategy="mean").fit(
        inputs["train"], targets["train"]
    )
    models["linear_10_features"] = LinearRegression().fit(
        inputs["train"], targets["train"]
    )
    model_inputs = {name: inputs for name in models}
    for dimensions in PCA_DIMENSIONS:
        # PCA 不接收 target，方向只从训练特征中学习。
        pca = PCA(n_components=dimensions, svd_solver="full", whiten=False)
        pca.fit(inputs["train"])
        pcas[dimensions] = pca
        projected[dimensions] = {
            split: pca.transform(values) for split, values in inputs.items()
        }
        name = f"pca_{dimensions}_linear"
        model_inputs[name] = projected[dimensions]
        models[name] = LinearRegression().fit(
            projected[dimensions]["train"], targets["train"]
        )
        info = {
            "n_components": dimensions,
            "cumulative_explained_variance_ratio": pca.explained_variance_ratio_.sum(),
        }
        for split in ("train", "validation"):
            restored = pca.inverse_transform(projected[dimensions][split])
            # 对所有样本、全部 10 列取平均；两边均在标准化空间。
            info[f"{split}_reconstruction_mse"] = np.mean(
                (inputs[split] - restored) ** 2
            )
        pca_rows.append(info)

    for name, model in models.items():
        row = {
            "model": name,
            "pca_dimensions": int(name.split("_")[1]) if name.startswith("pca_") else None,
        }
        for split in ("train", "validation"):
            scores = evaluate(targets[split], model.predict(model_inputs[name][split]))
            row.update({f"{split}_{metric}": value for metric, value in scores.items()})
        rows.append(row)

    # 误差完全相同时，元组的第二项让 2 维优先。测试集不参与选择。
    selected = min(
        (row for row in rows if row["pca_dimensions"] is not None),
        key=lambda row: (row["validation_rmse"], row["pca_dimensions"]),
    )["pca_dimensions"]
    for row in rows:
        name = row["model"]
        row["selected_pca"] = row["pca_dimensions"] == selected
        row["test_mae"] = row["test_rmse"] = np.nan
        if row["pca_dimensions"] is None or row["selected_pca"]:
            scores = evaluate(targets["test"], models[name].predict(model_inputs[name]["test"]))
            row.update({f"test_{metric}": value for metric, value in scores.items()})

    metrics = pd.DataFrame(rows)
    summary = pd.DataFrame(pca_rows)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / "pca_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    positions, width = np.arange(4), 0.36
    ax.bar(positions - width / 2, metrics.train_rmse, width, label="Train")
    ax.bar(positions + width / 2, metrics.validation_rmse, width, label="Validation")
    ax.set_xticks(positions, ["Mean baseline", "Linear (10 features)", "PCA 2 + Linear", "PCA 5 + Linear"])
    ax.set(ylabel="RMSE", title="Training and validation prediction error")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "rmse_comparison.png")

    full_pca = PCA(n_components=10, svd_solver="full", whiten=False).fit(inputs["train"])
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, 11), cumulative, "o-", label="Training cumulative variance")
    for dimensions in PCA_DIMENSIONS:
        value = cumulative[dimensions - 1]
        ax.axvline(dimensions, linestyle="--", alpha=0.4)
        ax.scatter(dimensions, value, s=80, zorder=3)
        ax.annotate(f"{dimensions} PCs: {value:.1%}", (dimensions, value),
                    xytext=(10, -24), textcoords="offset points")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance ratio",
           title="PCA explained variance (training features)", xticks=np.arange(1, 11), ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, "pca_variance.png")
    return metrics, summary, pcas[2], projected[2]["train"], selected


def run_clustering(x_train, x_train_scaled, pca, coordinates):
    """聚类使用 10 维标准化特征，绘图复用训练集的二维 PCA。"""
    metric_rows, summary_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")
    for ax, k in zip(axes, (2, 3)):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(x_train_scaled)
        labels = model.labels_
        score = silhouette_score(x_train_scaled, labels, metric="euclidean")
        metric_rows.append({"k": k, "silhouette_score": score, "random_state": RANDOM_STATE,
                            "n_init": 10, "feature_dimensions": x_train_scaled.shape[1]})
        # labels 是位置顺序；assign 按相同位置与原始训练行对应。
        grouped = x_train.assign(cluster=labels).groupby("cluster").agg(
            sample_count=("bmi", "size"), bmi_mean=("bmi", "mean"), bp_mean=("bp", "mean")
        )
        assert grouped.sample_count.sum() == len(x_train), "聚类样本数与训练集不一致"
        centers = pca.transform(model.cluster_centers_)
        for cluster, stats in grouped.iterrows():
            summary_rows.append({"k": k, "cluster": cluster, **stats.to_dict()})
            mask = labels == cluster
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=25, alpha=0.65,
                       color=colors(cluster), label=f"Cluster {cluster}")
            ax.scatter(*centers[cluster], marker="X", s=180, color=colors(cluster),
                       edgecolor="black", linewidth=1.2, zorder=4)
        ax.scatter([], [], marker="X", s=100, color="white", edgecolor="black", label="Projected centers")
        ax.set(title=f"K={k}, silhouette={score:.3f}", xlabel="PC1")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("PC2")
    fig.suptitle("K-means: clustering in 10D, displaying in 2D")
    save_figure(fig, "cluster_comparison.png")
    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summary_rows)
    summary["sample_count"] = summary["sample_count"].astype(int)
    metrics.to_csv(RESULTS_DIR / "cluster_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / "cluster_summary.csv", index=False, encoding="utf-8-sig")
    return metrics, summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_diabetes(as_frame=True, scaled=False)
    x_train, x_temp, y_train, y_temp = train_test_split(
        data.data, data.target, test_size=0.4, random_state=RANDOM_STATE
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(x_train)
    inputs = {"train": scaler.transform(x_train), "validation": scaler.transform(x_val),
              "test": scaler.transform(x_test)}
    targets = {"train": y_train, "validation": y_val, "test": y_test}
    metrics, pca_summary, pca, coordinates, selected = run_regression(inputs, targets)
    cluster_metrics, cluster_summary = run_clustering(x_train, inputs["train"], pca, coordinates)
    print(f"数据划分：训练 {len(y_train)} / 验证 {len(y_val)} / 测试 {len(y_test)}")
    print(f"根据验证 RMSE 选定 PCA 维数：{selected}（未选中配置的测试指标留空）")
    for title, table in (("回归指标", metrics), ("PCA 信息", pca_summary),
                         ("聚类指标", cluster_metrics), ("组内原始特征统计", cluster_summary)):
        print(f"\n{title}：")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n结果保存到：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
