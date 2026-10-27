"""实践一入口：python run_windows.py。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_data, make_windows
from reporting import save_csv

# 【阅读路线】main：取前12条 → 分别构造L=3/5 → 核对首尾和全部样本 → 保存CSV。
BASE_DIR = Path(__file__).resolve().parent


# 【函数作用】组织实践一，验证窗口构造；这里没有训练或模型预测。
# 输入：固定路径的原始 CSV。输出：终端核对信息和 window_examples.csv。
# 函数不返回表格；结果通过保存文件交付。
def main():
    # 【步骤 1：准备短序列】时间用于定位；OT 用于构造输入和答案。
    df = load_data(BASE_DIR / "data" / "ETTh1.csv").iloc[:12]
    series = df.OT.to_numpy(dtype=float)
    times = df.date.to_numpy()
    print("前 12 条原始记录：\n", df[["date", "OT"]].to_string(index=False))
    records = []
    # 【核心步骤 2：对比窗口】同一段数据、同一函数，只改变历史长度。
    for L in (3, 5):
        # 预期 L=3 得到9行3列，L=5 得到7行5列；最近输入距目标仍是1小时。
        X, y, target_index = make_windows(series, L)
        print(f"\nL={L}: X={X.shape}, y={y.shape}, target_index={target_index.shape}")
        assert X.shape == (12 - L, L)
        assert y.shape == target_index.shape == (12 - L,)
        # 【步骤 3：逐条核对】row 是样本编号，t 是目标在原始记录中的位置。
        # 例如第一条 L=3 样本：row=0，但 t=3；不能混用。
        for row, t in enumerate(target_index):
            assert (times[t - L:t] < times[t]).all()
            assert times[t] - times[t - 1] == np.timedelta64(1, "h")
            np.testing.assert_array_equal(X[row], series[t - L:t])
            assert y[row] == series[t]
            # 【核心步骤 4：形成输出行】每行记录一条“历史 → 答案”。
            # input_end=t-1 是实际包含的末尾位置；target_value 是实测，非预测。
            records.append({"window_size": L, "input_start": int(t - L), "input_end": int(t - 1),
                            "target_index": int(t), "target_time": times[t],
                            "input_values": json.dumps(X[row].tolist()), "target_value": y[row]})
        # 只打印首尾，便于肉眼检查；中间样本也构造并保存了。
        for row in (0, len(y) - 1):
            t = target_index[row]
            print(f"输入 {times[t-L]} 至 {times[t-1]}: {X[row]} → 目标 {times[t]}: {y[row]}")
    # 【步骤 5：保存】两组共 9+7=16 行；写入 results/window_examples.csv。
    frame = pd.DataFrame(records)
    assert frame.groupby("window_size").size().to_dict() == {3: 9, 5: 7}
    path = BASE_DIR / "results" / "window_examples.csv"
    save_csv(frame, path)
    print("已保存：", path)


if __name__ == "__main__":
    main()
