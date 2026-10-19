# 第五周：MLP 与 PyTorch 训练基础

## 运行

使用 Python 3.10 或更高版本，CPU 即可。依赖范围见 requirements.txt；编写时已检测到的本机环境为 Python 3.10.18、NumPy 2.2.5、pandas 2.3.3、scikit-learn 1.7.2、Matplotlib 3.10.6，但该解释器缺少 PyTorch。本交付尚未执行 PyTorch 训练，不能把预期数值当作实测结果。

进入 week05 后，在已安装 PyTorch 的环境执行：

```sh
python -m pip install -r requirements.txt
python run_step.py
python run_mlp.py
```

如果环境已经具备依赖，直接执行后两个命令。结果路径相对于脚本位置，切换工作目录不会改变保存位置。

## 文件及调用关系

| 文件 | 职责 |
|---|---|
| data.py | 两样本演示；糖尿病数据划分、输入/目标标准化、张量、DataLoader |
| models.py | 零初始化线性模型、正常随机初始化 MLP、前向形状检查 |
| optimizers.py | SGD/Adam 创建与正式实验学习率 |
| training.py | 一次更新、逐批逐轮训练、最佳参数记录和恢复、基线拟合 |
| evaluation.py | 无梯度预测、标准化 MSE、原始单位 MAE/RMSE、基线评价 |
| reporting.py | 仅保存 CSV 与图形，不训练模型 |
| run_step.py | 连接数据、线性模型、SGD、一次更新与记录保存 |
| run_mlp.py | 连接数据、基线、两组 MLP、验证选择、测试评价和图表 |

核心位置：training.one_step 与 training.fit_mlp 内的 zero_grad、前向、MSELoss、backward、step。backward 计算梯度，step 更新参数。fit_mlp 的内循环使用当前 xb/yb；外循环结束后调用 evaluation.scaled_mse，重新评价完整训练集与验证集。

## 实验设置

- 与第四周一致：load_diabetes(as_frame=True, scaled=False)，10 个输入；先用种子 42 划出 40%，再用种子 42 将这部分平分为验证与测试，得到训练 265 / 验证 88 / 测试 89。
- StandardScaler 仅 fit 训练输入。目标均值与 std(ddof=0) 仅来自训练目标；三份数据沿用同一套统计量。预测用 pred_scaled * y_std + y_mean 恢复原始单位。
- CPU、float32、10 → 16 → ReLU → 1，输出层不加 ReLU，允许负的标准化目标预测。
- torch.manual_seed(42) 创建初始模型，各配置用 deepcopy 获得相同初始参数。独立 DataLoader 分别使用新建的 Generator.manual_seed(42)。batch_size=32、drop_last=False、num_workers=0。
- SGD lr=0.01、momentum=0、weight_decay=0；Adam lr=0.001、weight_decay=0，其余默认。两组都训练满 200 轮。学习率也不同，结果反映两套配置，不能只归因于算法。
- 每轮记录整份训练/验证数据的标准化 MSE；严格小于历史最佳验证 MSE 时 deepcopy 参数，相同保留较早轮次；结束后恢复最佳参数。
- 均值基线与线性回归重新拟合。只根据验证 RMSE 选择 MLP，相同选 SGD。仅均值基线、线性回归与选中 MLP 评价测试集；未选中 MLP 的测试指标留空。无合并重训、无额外搜索。

## 输出

两个入口都运行后，results/ 恰有三份 CSV、两张 PNG：

| 文件 | 内容 |
|---|---|
| step_trace.csv | 清梯度后、反传后、更新后的权重/偏置/梯度；更新前后损失 |
| history.csv | 两组各 200 行，配置、学习率、epoch、训练/验证 MSE、standardized 尺度标记 |
| metrics.csv | 四模型各一行，学习率、最佳 epoch、原始单位训练/验证/指定测试指标、选中标记 |
| loss_curves.png | SGD/Adam 双面板训练与验证曲线，标记最佳轮次；纵轴为标准化目标 MSE |
| rmse_comparison.png | 四模型训练/验证 RMSE，原始目标单位；MLP 为最佳轮次参数 |

实践一包含数值断言：预期初始 loss=10，dw=-10，db=-6，更新后 w=1、b=0.6、重新计算 loss≈0.26。原来保存的 pred 不随参数更新自动重算；step 后梯度仍保留，下一次 zero_grad 才会清理。

## 隐藏层修改练习与组会说明

models.mlp 的 hidden_dim 同时控制 Linear(10, hidden_dim) 与 Linear(hidden_dim, 1)。run_mlp.main 最后保留 mlp(hidden_dim=8) 的前向检查，临时模型不训练、不产生额外图表。

以 32 个样本为例，程序将打印以下形状（需实际运行确认）：

```text
Width 16: [(32, 10), (32, 16), (32, 16), (32, 1)]
Width 8:  [(32, 10), (32, 8), (32, 8), (32, 1)]
```

只改第一层为 8 而第二层仍接收 16 会导致矩阵维数不匹配；最后输出维数仍为 1，所以每条样本只有一个预测。隐藏特征是已有输入的可学习变换，不增加原始测量信息。没有 ReLU 时，两层仿射变换仍能合并为单层仿射变换。最终损失沿计算图依赖传播，第一层同样得到梯度。

最后一批 loss 来自更新前的当前批次；整轮训练 MSE 来自本轮所有更新完成后的完整训练集，二者不能替代。Adam 利用梯度的一阶、二阶历史估计调整各参数步长，学习率仍控制总体更新幅度。最佳模型由验证集决定，测试集仅用于最终评价。

## 放入原仓库

将本目录复制到原仓库，路径应为 ml-weekly-practice/week05/。在仓库根 README 的每周任务列表增加：

```markdown
- [第五周实践：MLP 与 PyTorch 训练基础](week05/)
```

运行两个入口，确认三份 CSV、两张 PNG 后，将 week05 以及根 README 一起提交。按任务要求在组会当周周三北京时间 16:00 前 push，并在 GitHub 网页确认。此交付没有执行提交或推送。
