"""实践二：单特征线性回归、梯度下降与模型评估。"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(y_true, y_pred):
    """返回 MAE 和 RMSE。数值越小越好。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def gradient_descent(x, y, learning_rate=0.01, epochs=500):
    """使用全量训练样本，手动实现一元线性回归梯度下降。"""
    w, b = 0.0, 0.0
    losses = []

    for _ in range(epochs):
        prediction = w * x + b
        error = prediction - y
        losses.append(np.mean(error ** 2))

        # 梯度均使用本轮更新前的参数计算
        dw = 2 * np.mean(error * x)
        db = 2 * np.mean(error)
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, losses


def main():
    data = load_diabetes(as_frame=True, scaled=False)
    x = data.frame[["bmi"]].to_numpy(dtype=float).ravel()
    y = data.target.to_numpy(dtype=float)

    # 固定随机种子，先划分 60% 训练集、40% 临时集，再平分验证集和测试集
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    # 只用训练集计算标准化参数，避免验证集和测试集信息泄漏
    train_mean = x_train.mean()
    train_std = x_train.std()
    x_train_std = (x_train - train_mean) / train_std
    x_val_std = (x_val - train_mean) / train_std
    x_test_std = (x_test - train_mean) / train_std

    datasets = {
        "train": (x_train_std, y_train),
        "val": (x_val_std, y_val),
        "test": (x_test_std, y_test),
    }

    # 均值基线：只使用训练集目标值均值
    baseline_value = y_train.mean()
    results = []
    for name, (x_part, y_part) in datasets.items():
        prediction = np.full_like(y_part, baseline_value)
        mae, rmse = evaluate(y_part, prediction)
        results.append(["mean_baseline", name, mae, rmse])

    # 两种学习率使用相同初始化和迭代次数
    models = {}
    for learning_rate in (0.01, 0.1):
        w, b, losses = gradient_descent(
            x_train_std, y_train, learning_rate=learning_rate, epochs=500
        )
        model_name = f"gradient_descent_lr_{learning_rate}"
        models[model_name] = (w, b, losses)
        for name, (x_part, y_part) in datasets.items():
            prediction = w * x_part + b
            mae, rmse = evaluate(y_part, prediction)
            results.append([model_name, name, mae, rmse])

    # sklearn 线性回归作为最小二乘对照
    library_model = LinearRegression().fit(x_train_std.reshape(-1, 1), y_train)
    library_name = "sklearn_linear_regression"
    for name, (x_part, y_part) in datasets.items():
        prediction = library_model.predict(x_part.reshape(-1, 1))
        mae, rmse = evaluate(y_part, prediction)
        results.append([library_name, name, mae, rmse])

    print("数据集划分：训练集 {}, 验证集 {}, 测试集 {}".format(
        len(y_train), len(y_val), len(y_test)
    ))
    print("\n模型评估结果（MAE / RMSE）：")
    for model, split, mae, rmse in results:
        print(f"{model:30s} {split:5s} MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"\n梯度下降 lr=0.01: w={models['gradient_descent_lr_0.01'][0]:.4f}, "
          f"b={models['gradient_descent_lr_0.01'][1]:.4f}")
    print(f"梯度下降 lr=0.1:  w={models['gradient_descent_lr_0.1'][0]:.4f}, "
          f"b={models['gradient_descent_lr_0.1'][1]:.4f}")
    print(f"LinearRegression: w={library_model.coef_[0]:.4f}, b={library_model.intercept_:.4f}")

    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "split", "MAE", "RMSE"])
        writer.writerows(results)

    # 两种学习率的训练损失曲线
    plt.figure(figsize=(7, 4))
    for learning_rate in (0.01, 0.1):
        losses = models[f"gradient_descent_lr_{learning_rate}"][2]
        plt.plot(losses, label=f"learning_rate={learning_rate}")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("Gradient Descent Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 选取验证集 RMSE 更小的梯度下降模型进行测试集预测可视化
    selected_name = min(
        models,
        key=lambda name: next(r[3] for r in results if r[0] == name and r[1] == "val"),
    )
    w, b, _ = models[selected_name]
    test_prediction = w * x_test_std + b
    order = np.argsort(y_test)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, test_prediction, alpha=0.7)
    limits = [min(y_test.min(), test_prediction.min()), max(y_test.max(), test_prediction.max())]
    plt.plot(limits, limits, "r--", label="y=x")
    plt.xlabel("True target")
    plt.ylabel("Predicted target")
    plt.title(f"{selected_name}: Test Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "test_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n结果已保存到：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
