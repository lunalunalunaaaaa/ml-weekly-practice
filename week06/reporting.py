"""只接收已计算好的记录：保存 CSV 和画图；不拟合、不选模型。"""
import os
import tempfile
from pathlib import Path

# 缓存写到临时目录，不依赖用户主目录可写。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "week06-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import pandas as pd


# 【函数作用】从本机已有字体中选可显示中文的字体；只影响显示。
# 输入：无。输出：修改绘图字体设置；不返回数据。
def configure_fonts():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    # 【核心代码】选到一种可用字体就使用，避免中文标签显示成方框。
    for name in ["PingFang SC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


# 【函数作用】把已算好的表格原样保存，不重新计算指标或预测。
# 输入：DataFrame和保存路径。输出：一份CSV文件。
def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 【核心代码】不额外写入表格行号；保留足够数值精度便于重新计算误差。
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


# 【函数作用】展示普通回归和三个Ridge配置的训练/验证误差。
# 输入：从tuning.csv读取的表格、图片路径。输出：调参比较PNG。
# 阅读重点：横轴是配置类别，不是训练轮次；纵轴是原始尺度RMSE。
def plot_tuning(tuning, path):
    configure_fonts()
    labels = ["lag_linear" if row.model == "lag_linear" else f"Ridge α={row.param_value:g}"
              for row in tuning.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    positions = list(range(len(labels)))
    # 【核心代码】同一个配置画训练、验证两项误差，观察改善是否延续到验证集。
    ax.plot(positions, tuning.train_rmse, "o-", label="训练 RMSE")
    ax.plot(positions, tuning.val_rmse, "s-", label="验证 RMSE")
    ax.set_xticks(positions, labels)
    ax.set(xlabel="不同模型配置（不是训练轮次）", ylabel="RMSE（原始油温尺度）", title="普通线性回归与 Ridge 三个候选")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# 【函数作用】展示实验时段与初始历史，帮助解释时间划分。
# 输入：实验使用的原始表、历史长度、时段边界、保存路径。输出：划分图PNG。
# 上图看全局划分，下图放大首条样本，图区分“初始历史”和“首个目标”。
def plot_series(df, window_size, boundaries, path):
    configure_fonts()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    # 【核心代码 1】以原始日期为横轴画真实油温，不使用样本编号代替时间。
    ax.plot(df.date, df.OT, lw=.65, label="真实油温")
    # 标出验证、测试起点；模型成绩必须结合这些时段的变化来理解。
    for name, color in [("val", "tab:orange"), ("test", "tab:red")]:
        start = pd.Timestamp(boundaries[name][0])
        ax.axvline(start, color=color, ls="--", label=f"{name} 起点 {start:%Y-%m-%d}")
    # 初始输入是下标0至L-1；下标L是首个目标，阴影右边界标到该时刻。
    history_end = df.date.iloc[window_size]
    ax.axvspan(df.date.iloc[0], history_end, color="tab:green", alpha=.25, label="初始 24 条历史（下图放大）")
    ax.set(ylabel="油温 OT（原始数据尺度）", title="前 14,400 小时：按目标时刻划分")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # 【核心代码 2】放大L条历史+1个目标，红点是要预测的答案，不是输入。
    zoom = df.iloc[:window_size + 1]
    axes[1].plot(zoom.date, zoom.OT, "o-", ms=3)
    axes[1].axvspan(zoom.date.iloc[0], history_end, alpha=.15, color="tab:green")
    axes[1].scatter([history_end], [zoom.OT.iloc[-1]], color="tab:red", zorder=3, label="首个目标；不属于输入")
    axes[1].set(xlabel="时间", ylabel="油温 OT（原始数据尺度）", title="首条样本：前 24 个观测作为输入，随后 1 个观测作为目标")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# 【函数作用】对照验证首周的真实油温、四种预测与同期HUFL。
# 输入：详细预测表、原始数据、模型顺序、展示起止时间、路径。
# 输出：共用时间轴的上下两图PNG；本函数不计算完整验证指标。
def plot_validation(predictions, df, model_order, start, end, path):
    configure_fonts()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # 【核心代码 1】按目标真实时间截取验证首周，不能把其他时段混入。
    rows = predictions[(predictions.split == "val") & (predictions.target_time >= start) & (predictions.target_time < end)]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 7), layout="constrained")
    # 四种方法共用同一组真实答案，只取一份绘制，避免重复画真实曲线。
    truth = rows[rows.model == model_order[0]].sort_values("target_time")
    axes[0].plot(truth.target_time, truth.y_true, color="black", lw=1.8, label="真实油温")
    styles = ["--", ":", "-", "-."]
    # 【核心代码 2】每种方法先按目标时间排序，再画其预测，保证位置对齐。
    for name, style in zip(model_order, styles):
        part = rows[rows.model == name].sort_values("target_time")
        axes[0].plot(part.target_time, part.y_pred, ls=style, lw=1, alpha=.85, label=name)
    axes[0].set(ylabel="油温 OT（原始数据尺度）", title="验证集首周：2017-06-26 至 2017-07-02（每小时预测下一小时）")
    axes[0].legend(ncol=3, fontsize=9)
    # 【核心代码 3】从原始表取完全相同时间范围的HUFL，并核对时间逐项一致。
    load = df[(df.date >= start) & (df.date < end)].sort_values("date")
    assert truth.target_time.tolist() == load.date.tolist()
    # HUFL只帮助事后观察；没有进入本周预测输入，不能仅凭曲线认定因果。
    axes[1].plot(load.date, load.HUFL, color="tab:purple", lw=1)
    axes[1].set(ylabel="HUFL（原始数据尺度）", xlabel="目标时间", title="同一时段的 HUFL 负载通道（仅用于事后观察）")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
