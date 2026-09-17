# 第二周实践：线性回归与 GitHub 代码管理

## 任务内容

使用 diabetes 数据集完成 bmi 单特征线性回归、梯度下降和模型评价。

## 文件说明

- `main.py`：主要实验代码
- `results/`：运行结果
- `requirements.txt`：依赖包

## 运行方法

```bash
pip install -r requirements.txt
python main.py
# 第二周实践：线性回归与梯度下降

## 实践内容

使用 scikit-learn 自带的 diabetes 数据集，仅选择 `bmi` 一个特征，完成单特征线性回归实验：

- 按 60%/20%/20% 划分训练集、验证集和测试集
- 仅使用训练集的均值和标准差对 `bmi` 做标准化
- 建立训练集目标值均值的基线模型
- 使用 NumPy 手动实现全量梯度下降
- 比较学习率 `0.01` 和 `0.1` 的训练损失
- 使用 `LinearRegression` 与手写梯度下降结果进行对照
- 使用 MAE 和 RMSE 评估模型

## 环境依赖

```bash
pip install -r requirements.txt
```

依赖包括：

- `numpy`：数组运算和手写梯度下降
- `scikit-learn`：数据集、数据划分、评估指标和线性回归对照模型
- `matplotlib`：绘制损失曲线和预测散点图

## 运行代码

在 `week02` 目录下执行：

```bash
python main.py
```

## 输出结果

程序会在 `week02/results` 文件夹中生成：

- `metrics.csv`：各模型在训练集、验证集和测试集上的 MAE、RMSE
- `loss_curves.png`：两种学习率的训练损失曲线
- `test_predictions.png`：测试集真实值与预测值散点图，包含 `y=x` 参考线

程序运行时还会在终端输出数据集划分、各模型的 MAE/RMSE，以及梯度下降和库函数模型的参数。

## 结果解读

MAE 和 RMSE 都是误差指标，数值越小越好。验证集用于选择梯度下降配置，测试集只用于最终评估，避免根据测试结果反复调整模型。
