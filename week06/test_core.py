"""检查手算窗口、跨边界历史、误差公式以及标准化无泄漏。"""
import unittest
import numpy as np
import pandas as pd
from data import make_windows, split_by_target
from models import build_linear
from evaluation import regression_metrics


class CoreTests(unittest.TestCase):
    # 【测试作用】用手算答案核对窗口、目标和下标，并检查非法输入会报错。
    # 核心是下面三项数组比较；不是在评价模型预测精度。
    def test_hand_calculated_windows(self):
        X, y, t = make_windows([10, 12, 11, 15, 14, 16], 3)
        np.testing.assert_array_equal(X, [[10, 12, 11], [12, 11, 15], [11, 15, 14]])
        np.testing.assert_array_equal(y, [15, 14, 16])
        np.testing.assert_array_equal(t, [3, 4, 5])
        for values, size in [([1, 2], 2), ([1, 2], 1.5), ([1, np.nan, 3], 1)]:
            with self.assertRaises(ValueError):
                make_windows(values, size)

    # 【测试作用】边界样本应能用前一时段已知历史；目标仍按自己的时间归组。
    # 核心是验证首个窗口为[2,3]、测试首个窗口为[4,5]；重叠范围必须拒绝。
    def test_boundary_history_and_overlap(self):
        dates = pd.date_range("2020-01-01", periods=8, freq="h")
        X, y, t = make_windows(np.arange(8), 2)
        bounds = {"train": (dates[0], dates[4]), "val": (dates[4], dates[6]),
                  "test": (dates[6], dates[-1] + pd.Timedelta(hours=1))}
        result = split_by_target(X, y, t, dates, bounds)
        np.testing.assert_array_equal(result["val"]["X"][0], [2, 3])
        self.assertEqual(result["val"]["target_index"][0], 4)
        np.testing.assert_array_equal(result["test"]["X"][0], [4, 5])
        bounds["train"] = (dates[0], dates[5])
        with self.assertRaises(ValueError):
            split_by_target(X, y, t, dates, bounds)

    # 【测试作用】用误差[1,-2]手算 MAE=1.5、RMSE=sqrt(2.5)，核对指标公式。
    # 输入两条真实/预测值；核心是下面两项数值比较。
    def test_metric_formula(self):
        score = regression_metrics(np.array([1., 3.]), np.array([2., 1.]))
        self.assertAlmostEqual(score["mae"], 1.5)
        self.assertAlmostEqual(score["rmse"], np.sqrt(2.5))

    # 【测试作用】预测新数据时，标准化器必须继续使用训练均值，不能重新学习。
    # 输入三条训练样本及一条远离训练范围的新输入。
    # 核心是predict前后均值相同，且已见样本数仍为3，防止验证/测试泄漏。
    def test_prediction_keeps_training_statistics(self):
        X = np.array([[0., 1.], [2., 3.], [4., 5.]])
        model = build_linear().fit(X, np.array([2., 4., 6.]))
        scaler = model.named_steps["standardscaler"]
        before = scaler.mean_.copy()
        model.predict(np.array([[100., 101.]]))
        np.testing.assert_array_equal(before, [2., 3.])
        np.testing.assert_array_equal(scaler.mean_, before)
        self.assertEqual(scaler.n_samples_seen_, 3)


if __name__ == "__main__":
    unittest.main()
