"""实践一入口。"""
import math
from data import step_data
from models import linear_model
from optimizers import make_optimizer
from training import one_step
from reporting import save_step


def main():
    x, y = step_data()
    model = linear_model()
    rows = one_step(model, make_optimizer(model.parameters(), 'SGD', 0.1), x, y)
    # 以下预期值只属于本入口的固定样例：y_hat = w*x+b，初始 w=b=0。
    # MSE=(4+16)/2=10；dw=-10、db=-6；SGD 使用 参数 -= lr*梯度。
    # 更新后 w=1、b=0.6，预测为 1.6、2.6，MSE=(0.16+1.96)/2=1.06。
    before, backward, after = rows
    expected = ((before['loss'], 10.), (backward['weight_grad'], -10.),
                (backward['bias_grad'], -6.), (after['weight'], 1.),
                (after['bias'], 0.6), (after['loss'], 1.06))
    for actual, target in expected:
        assert math.isclose(actual, target, abs_tol=1e-6), (actual, target)
    save_step(rows)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
