# 第一周实践：数据读取与可视化

## 实践内容

使用 scikit-learn 自带的 diabetes 回归数据集，完成以下操作：

- 加载 diabetes 数据集并查看基本信息
- 查看数据集的特征列、目标列和前几行数据
- 统计数据集中的缺失值
- 查看数值列的描述性统计信息
- 绘制目标值分布直方图
- 绘制 BMI 与目标值的散点图
- 将特征从 `bmi` 修改为 `s5`，重新绘制散点图

## 环境依赖

安装项目依赖：

```bash
pip install -r requirements.txt
```

## 运行代码

在 `week01` 目录下执行：

```bash
python main.py
```

## 输出结果

程序会在 `week01/results` 文件夹中生成以下图片：

- `target_distribution.png`：目标值分布图
- `bmi_vs_target.png`：BMI 与目标值散点图
- `s5_vs_target.png`：s5 与目标值散点图

程序同时会在终端输出数据集的行列数、列名、前几行数据、缺失值统计和描述性统计信息。
