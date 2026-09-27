# 第三周实践：多特征回归与 Ridge

## 实践内容

使用 scikit-learn 自带的 diabetes 回归数据集，对比均值基线、BMI 单特征线性回归、全部 10 个特征的线性回归，以及 `alpha=1`、`alpha=100` 的 Ridge 回归。

## 数据来源

代码通过 `sklearn.datasets.load_diabetes(as_frame=True, scaled=False)` 直接读取数据。输入使用数据集提供的 10 个特征，预测目标为 `target`。`target` 不会放入输入特征。

## 主要文件

- `main.py`：数据划分、标准化、模型训练、评价、Ridge 选择和结果导出
- `requirements.txt`：运行所需依赖
- `results/metrics.csv`：训练、验证和最终测试指标，以及选定 Ridge 标记
- `results/weights.csv`：多特征线性回归和两组 Ridge 的权重
- `results/train_validation_rmse.png`：五组模型的训练/验证 RMSE 对比图
- `results/weights_comparison.png`：三组多特征模型的权重对比图

## 环境与安装

本实践已在 Python 3.10.18 下运行验证。在 `week03` 目录执行：

```bash
python -m pip install -r requirements.txt
```

## 运行方法

```bash
python main.py
```

程序会重新生成 `results/` 中的 CSV 和 PNG 文件，并在终端打印数据集大小、全部指标以及由验证集选出的 Ridge `alpha`。

本次运行中，`alpha=1` 的验证集 RMSE 小于 `alpha=100`，因此最终选择 `alpha=1`。程序每次运行时都会根据实际验证结果重新完成选择。

## 关键设置

- 随机种子为 `42`
- 按约 60%/20%/20% 划分训练集、验证集和测试集
- `StandardScaler` 只在训练集上拟合，验证集和测试集复用训练集的均值与标准差
- 单特征模型使用标准化后的 `bmi`；多特征模型使用全部 10 个标准化特征
- Ridge 只比较 `alpha=1` 与 `alpha=100`
- 根据验证集 RMSE 选择 Ridge；测试集不参与配置选择
- 测试集评价均值基线、单特征线性回归、多特征线性回归和选定的 Ridge

## 结果位置

运行完成后，请查看 `week03/results/`。`metrics.csv` 的 `selected_ridge` 列会明确标注最终选定的 Ridge 配置。

## 结果解读

本次运行得到的测试集 RMSE 如下：

| 模型 | 测试 RMSE |
|---|---:|
| 均值基线 | 76.9057 |
| BMI 单特征线性回归 | 67.0440 |
| 多特征线性回归 | 56.9703 |
| Ridge `alpha=1` | 57.0834 |

使用全部 10 个特征后，测试 RMSE 明显低于只使用 `bmi`，说明其他特征提供了额外的预测信息。Ridge `alpha=1` 与普通多特征线性回归的效果非常接近，说明较弱的正则化对预测结果影响不大。

验证集上的 Ridge 结果为：

- `alpha=1`：RMSE 为 `49.2033`
- `alpha=100`：RMSE 为 `50.8149`

因此根据验证集 RMSE 选择 `alpha=1`。`alpha=100` 的正则化更强，权重被压缩得更明显，但模型拟合能力下降，验证误差反而变大。这说明正则化不一定保证预测误差下降，强度需要通过验证集选择。

`train_validation_rmse.png` 展示五组模型在训练集和验证集上的 RMSE，能够看到多特征模型整体优于均值基线和 BMI 单特征模型。`weights_comparison.png` 展示三组多特征模型的权重变化，其中 `alpha=100` 通常会让权重整体更接近 0。权重的正负表示线性关联方向，但不能据此推断因果关系或简单判断特征的重要性。
