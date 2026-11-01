"""关键语义检查：梯度不断链、训练统计量无泄漏、窗口状态独立。"""
import unittest
import numpy as np
import torch

from week07.data import prepare_data
from week07.models import OilLSTM
from week07.run_rnn import trace_rnn


class SequenceChecks(unittest.TestCase):
    def test_rnn_bptt_matches_finite_difference(self):
        # 用与自动求导不同的 NumPy 数值路径检验所有输入与共享参数的梯度。
        import pandas as pd
        values = np.array([.1, -.2, .3, .15, -.1], dtype=float)
        z = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        params = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in [.6, .5, 0.]]
        dates = pd.Series(pd.date_range('2020-01-01', periods=5, freq='h'))
        pred, _ = trace_rnn(z, range(5), params, dates, 'check')
        (pred-.2).square().backward()
        def loss(v):
            h = 0.
            for x in v[:5]:
                h = np.tanh(v[5]*x+v[6]*h+v[7])
            return (.8*h+.1-.2)**2
        all_values = np.r_[values, .6, .5, 0.]
        numeric = []
        for i in range(8):
            plus, minus = all_values.copy(), all_values.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numeric.append((loss(plus)-loss(minus))/2e-6)
        analytic = np.r_[z.grad.numpy(), [p.grad.item() for p in params]]
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-10)

    def test_data_boundary_and_scaling(self):
        data = prepare_data()
        df, raw = data['df'], data['raw']
        self.assertAlmostEqual(data['mean'], df.OT.iloc[:8640].mean(), places=12)
        self.assertAlmostEqual(data['std'], df.OT.iloc[:8640].std(ddof=0), places=12)
        for split, count in [('train', 8616), ('val', 2880), ('test', 2880)]:
            self.assertEqual(len(raw[split]['y']), count)
            self.assertEqual(data['tensors'][split]['x'].shape, (count, 24, 1))
            self.assertEqual(data['tensors'][split]['x'].dtype, torch.float32)
            np.testing.assert_array_equal(raw[split]['X'][:, -1],
                                          df.OT.to_numpy()[raw[split]['target_index']-1])
        self.assertEqual(raw['val']['target_time'][0], df.date.iloc[8640])
        self.assertEqual(raw['test']['target_index'][0], 11520)

    def test_windows_do_not_share_state(self):
        torch.manual_seed(42)
        model = OilLSTM(16).eval()
        a, b = torch.randn(2, 24, 1), torch.randn(3, 24, 1)
        with torch.no_grad():
            expected = model(a)
            model(b)  # 处理其他窗口不应改变 a 的预测。
            torch.testing.assert_close(model(a), expected)
            # 同一批中的两个窗口也应相互独立。
            torch.testing.assert_close(model(a), torch.cat([model(a[:1]), model(a[1:])]))
        for width in (16, 32):
            info = OilLSTM(width).inspect(a)
            self.assertEqual(info['parameters'], 4*width*(width+3)+width+1)


if __name__ == '__main__':
    unittest.main()
