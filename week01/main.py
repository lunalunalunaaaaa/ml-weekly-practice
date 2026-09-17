# 实践二：数据读取与绘图
# 使用 scikit-learn 自带的 diabetes 回归数据集

import sys

# 禁止本次运行生成 .pyc 和 __pycache__ 缓存文件
sys.dont_write_bytecode = True

from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from pathlib import Path


# 图片统一保存到 week01/results 文件夹
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# 1. 加载 diabetes 数据集
dataset = load_diabetes(as_frame=True, scaled=False)
df = dataset.frame


# 2. 查看数据基本信息
print("=" * 60)
print("1. 数据基本信息")
print("=" * 60)
print("数据行数和列数：")
print(df.shape)
print("\n数据列名：")
print(df.columns.tolist())
print("\n前几行数据：")
print(df.head())


# 3. 指出特征列和目标列
print("\n" + "=" * 60)
print("2. 特征与目标变量")
print("=" * 60)
features = dataset.feature_names
target = dataset.target.name
print("特征列：")
print(features)
print("\n目标列：")
print(target)


# 4. 统计各列缺失值
print("\n" + "=" * 60)
print("3. 缺失值统计")
print("=" * 60)
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("\n结论：数据集中没有缺失值。")
else:
    print("\n结论：数据集中存在缺失值。")


# 5. 查看数值列的基本统计信息
print("\n" + "=" * 60)
print("4. 数值列基本统计")
print("=" * 60)
print(df.describe())


# 6. 绘制目标值分布图
print("\n" + "=" * 60)
print("5. 绘制目标值分布图")
print("=" * 60)
plt.figure(figsize=(6, 4))
plt.hist(df[target], bins=20, edgecolor="black")
plt.xlabel("target")
plt.ylabel("Frequency")
plt.title("Distribution of Target")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "target_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# 7. 绘制 BMI 与目标值的散点图
print("\n" + "=" * 60)
print("6. BMI 与目标值散点图")
print("=" * 60)
feature = "bmi"
plt.figure(figsize=(6, 4))
plt.scatter(df[feature], df[target], alpha=0.6)
plt.xlabel("bmi")
plt.ylabel("target")
plt.title("BMI vs Target")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "bmi_vs_target.png", dpi=300, bbox_inches="tight")
plt.show()


# 8. 小修改：将 BMI 替换为 s5
print("\n" + "=" * 60)
print("7. 小修改：更换散点图特征")
print("=" * 60)
feature_modified = "s5"
print("修改前的特征：bmi")
print("修改后的特征：s5")
plt.figure(figsize=(6, 4))
plt.scatter(df[feature_modified], df[target], alpha=0.6)
plt.xlabel("s5")
plt.ylabel("target")
plt.title("s5 vs Target")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "s5_vs_target.png", dpi=300, bbox_inches="tight")
plt.show()

print("=" * 60)
print("程序运行完成")
print("=" * 60)
