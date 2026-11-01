"""单层单向 LSTM：历史窗口 → 末步隐藏表示 → 无激活的线性预测。"""
import torch
from torch import nn


class OilLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, dropout=0, bidirectional=False, proj_size=0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 不传入上批状态：每个窗口默认从零 h、零 c 开始，权重仍保留学习结果。
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = output[:, -1, :]  # (B,24,H) → (B,H)，取最后时间步。
        return self.head(last_hidden)   # (B,H) → (B,1)，不加激活函数。

    @torch.no_grad()
    def inspect(self, x):
        """只检查一批真实输入，不更新参数，也不传递状态。"""
        output, (h_n, c_n) = self.lstm(x)
        torch.testing.assert_close(output[:, -1, :], h_n[0])
        prediction = self.head(output[:, -1, :])
        return {'input': list(x.shape), 'output': list(output.shape),
                'h_n': list(h_n.shape), 'c_n': list(c_n.shape),
                'prediction': list(prediction.shape),
                'parameters': sum(p.numel() for p in self.parameters())}
