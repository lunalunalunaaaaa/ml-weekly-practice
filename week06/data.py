"""读取原始数据、构造历史窗口、按目标时间划分。此模块不训练模型。"""
import numpy as np
import pandas as pd


# 【函数作用】读取并检查原始 CSV，返回保留全部列的表格 df。
# 输入：CSV 文件路径。输出：通过检查的 DataFrame。
# 阅读重点：下面的“核心代码”负责读取，其余检查保证一行确实对应一小时。
def load_data(path):
    """返回保留全部原始列的 DataFrame；拒绝无效记录，不静默插值。"""
    # 【核心代码：读取】保留油温原始尺度，并把 date 识别为可比较的日期时间。
    df = pd.read_csv(path, parse_dates=["date"])
    # 【数据检查】检查必要字段、时间有效/不重复/有序、每小时连续、数值有限。
    required = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    if not set(required).issubset(df.columns):
        raise ValueError("原始 CSV 缺少必要列")
    dates = df["date"]
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("时间必须有效、递增且不重复")
    if not dates.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("相邻记录必须恰好相差一小时")
    if not np.isfinite(df[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("数值列存在缺失或无效值")
    return df


# 【函数作用】把一条连续油温序列，整理为“历史输入 → 下一小时答案”。
# 输入：N 个油温、历史长度 L。输出：X(N-L,L)、y(N-L,)、target_index(N-L,)。
# 例：[10,12,11,15]，L=3 → X=[[10,12,11]]，y=[15]，目标下标=[3]。
def make_windows(series, window_size):
    """X[i] 从旧到新；y[i] 紧接窗口；target_index[i] 为原始目标下标。"""
    # 【输入检查】只接受一维有效序列与合理整数窗口；这些不是预测计算。
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError("series 必须是一维有效数值序列")
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)):
        raise ValueError("window_size 必须是整数")
    if not 1 <= window_size < len(series):
        raise ValueError("window_size 必须在 1 到 N-1 之间")
    # 三份结果逐行对应：历史窗口、真实答案、答案在原始序列中的位置。
    X, y, target_index = [], [], []
    # t 是答案的位置，切片右端不包含 t，所以不会把答案放进输入。
    # 【核心代码：窗口构造】t 是目标位置；前 L 条形成首个完整输入。
    # 每向后移动一个目标，就生成一条样本，总数 N-L。
    for t in range(window_size, len(series)):
        # 输入只包含 t-L 到 t-1，绝不包含目标 t，避免提前看到答案。
        X.append(series[t - window_size:t])
        # 目标是下一小时的真实观测，不是模型算出的预测。
        y.append(series[t])
        # 保留原始位置：后面据此找日期、分组、对齐预测图。
        target_index.append(t)
    return np.asarray(X), np.asarray(y), np.asarray(target_index, dtype=int)


# 【函数作用】按“要预测的时刻”划分训练、验证、测试，不打乱时间。
# 输入：全部窗口/目标/原始下标、原始时间、各组起止时间。
# 输出：splits[组名]，其中包含该组 X、y、target_index、target_time。
# 边界之前已发生的历史可以作为输入，因此必须先构造窗口、再划分。
def split_by_target(X, y, target_index, dates, boundaries):
    """先构造完整窗口，再以目标时刻分组；边界处允许使用已发生的历史。"""
    # 【核心代码 1：找到目标时间】例如原始下标 8640 对应首条验证目标。
    target_times = pd.DatetimeIndex(dates)[target_index]
    # 【完整性检查用】统计每条样本被选中几次，最终必须恰好为 1。
    assigned = np.zeros(len(y), dtype=int)
    splits = {}
    for name, (start, end) in boundaries.items():
        # 【核心代码 2：时间筛选】mask 是每条样本“选/不选”的清单。
        # start <= 目标时间 < end：包含起点、不含终点，边界不重复归组。
        # 例如验证范围 [6月26日,10月24日)，10月24日 00:00 应属于测试。
        mask = (target_times >= pd.Timestamp(start)) & (target_times < pd.Timestamp(end))
        assigned += mask.astype(int)
        if not mask.any():
            raise ValueError(f"{name} 没有样本")
        # 【核心代码 3：同步选取】输入、答案、下标、时间使用同一 mask，保持配对。
        splits[name] = {
            "X": X[mask], "y": y[mask],
            "target_index": target_index[mask], "target_time": target_times[mask],
        }
    # 0 次表示遗漏，超过 1 次表示时间范围重叠；两者都应停止实验。
    if not np.all(assigned == 1):
        raise ValueError("可用目标存在遗漏或重复归组")
    return splits
