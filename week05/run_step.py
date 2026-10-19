"""实践一入口。"""
from data import step_data
from models import linear_model
from optimizers import make_optimizer
from training import one_step
from reporting import save_step


def main():
    x, y = step_data()
    model = linear_model()
    rows = one_step(model, make_optimizer(model.parameters(), 'SGD', 0.1), x, y)
    save_step(rows)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
