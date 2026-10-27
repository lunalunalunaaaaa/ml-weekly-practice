"""两个固定规则与标准化线性模型。所有 fit 均由入口使用训练集调用。"""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 【函数作用】最近值基线，不需要训练。
# 输入：每行一段历史的 X。输出：每行一个预测，长度等于样本数。
def last_value(X):
    # 【核心代码】下一小时预测 = 窗口内最近的真实油温；不是历史预测值。
    return X[:, -1]  # 最后一列是最近一次真实观测。


# 【函数作用】历史窗口均值基线，不需要训练。
# 输入：原始油温窗口 X。输出：每条窗口的均值预测。
# 连续升温时，较早的低温可能拉低预测；降温时则可能相反。
def window_mean(X):
    # 【核心代码】每条样本内部平均，保留每条样本各自的预测。
    return X.mean(axis=1)  # axis=1：分别对每一行的历史求平均。


# 【函数作用】创建“标准化 → 普通线性回归”管道；此处只是创建，尚未训练。
# 输出：待拟合的模型；run_baselines.main 调用 fit 后才学到统计量和权重。
def build_linear():
    # 【核心代码】fit 时：按训练集每列学习均值/标准差，再学习回归权重。
    # predict 时：沿用训练统计量，计算 b + Σ(w[j] * z[j])；y 不标准化。
    return make_pipeline(StandardScaler(), LinearRegression())


# 【函数作用】创建“标准化 → Ridge 回归”管道。
# 输入：正的 alpha。输出：待拟合模型；每个候选创建独立管道。
# alpha 越大，权重约束越强；并不保证预测更好。
def build_ridge(alpha):
    if alpha <= 0:
        raise ValueError("alpha 必须为正")
    # 【核心代码】Ridge 在训练误差平方和之外增加 alpha * Σw[j]^2。
    # 与普通回归使用相同的标准化和训练样本，便于公平比较。
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
