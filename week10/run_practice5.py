"""从仓库根目录运行 python -m week10.run_practice5；固定参数验证因果遮罩。"""
import json

import pandas as pd
import torch

from week07.data import prepare_data
from week08.models import OilAttention
from week10.models import AttentionHead, TwoHeadAttention
from week10.run_practice4 import CHECKPOINT, RESULTS, ROOT, SEED


def row_record(result):
    """第一头第 6 行：关注第 18 小时、所有未来及整行的权重。"""
    row = result.head1.weights[0, 5, :]
    return {
        'head1_weight_6_to_18': row[17].item(),
        'head1_future_max_weight': row[6:].max().item(),
        'head1_row_weight_sum': row.sum().item(),
    }


def check_weights(result, *, use_causal_mask):
    """两个头、全部查询行都检查，防止只给第一头或第 6 行加遮罩。"""
    for head in (result.head1, result.head2):
        weights = head.weights
        assert weights.shape == (1, 24, 24)
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 24))
        if use_causal_mask:
            # 对角线上方全部严格为 0；对角线不遮住，第一行仍然可读自己。
            assert torch.count_nonzero(weights.triu(1)).item() == 0
            assert (weights.diagonal(dim1=-2, dim2=-1) > 0).all()
            torch.testing.assert_close(weights[0, 0, 0], torch.tensor(1.0))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'需要第八周最佳模型：{CHECKPOINT}；先运行 python -m week08.run')

    data = prepare_data()
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert (checkpoint['mean'], checkpoint['std']) == (data['mean'], data['std'])
    previous = OilAttention()
    previous.load_state_dict(checkpoint['state_dict'])
    previous.eval()

    # 复用实践四的首个验证窗口。先在标准化油温上改一项，再分别做输入映射。
    original = data['tensors']['val']['x'][:1].clone()  # (1, 24, 1)
    changed = original.clone()
    changed[0, 17, 0] += 1  # 第 18 小时，加一个训练集标准差。
    assert torch.count_nonzero(changed - original).item() == 1
    x = previous.embed(original)                     # (1, 24, 16)
    changed_x = previous.embed(changed)
    torch.testing.assert_close(x[:, :17], changed_x[:, :17], rtol=0, atol=0)
    torch.testing.assert_close(x[:, 18:], changed_x[:, 18:], rtol=0, atol=0)

    # 与实践四第二组的初始化顺序相同，复现它的双头参数。
    # 整个遮罩开/关实验只创建这一套模型，循环中不会换参数。
    model = TwoHeadAttention().eval()
    model.head2.load_state_dict(model.head1.state_dict())
    model.head2 = AttentionHead().eval()

    rows = []
    cases = {}
    traces = {}
    for use_causal_mask in (False, True):
        before = model(x, use_causal_mask=use_causal_mask)
        after = model(changed_x, use_causal_mask=use_causal_mask)
        check_weights(before, use_causal_mask=use_causal_mask)
        check_weights(after, use_causal_mask=use_causal_mask)
        # 比较双头拼接、输出投影之后的第 6 行，共 16 个特征。
        difference = (before.output[0, 5] - after.output[0, 5]).abs().max().item()
        if use_causal_mask:
            # 第 1～17 行都不应被第 18 小时影响；包含任务要求的第 6 行。
            torch.testing.assert_close(before.output[:, :17], after.output[:, :17],
                                       rtol=0, atol=1e-7)
            for first, second in ((before.head1, after.head1), (before.head2, after.head2)):
                torch.testing.assert_close(first.readout[:, :17], second.readout[:, :17],
                                           rtol=0, atol=1e-7)
        name = 'mask_on' if use_causal_mask else 'mask_off'
        row = {'case': name, **row_record(before),
               'output_row6_max_abs_difference': difference}
        rows.append(row)
        cases[name] = {
            **row,
            'changed_input_weights': row_record(after),
            'original_output_row6': before.output[0, 5].tolist(),
            'changed_output_row6': after.output[0, 5].tolist(),
            'head1_original_weights_row6': before.head1.weights[0, 5].tolist(),
            'head1_changed_weights_row6': after.head1.weights[0, 5].tolist(),
            'head2_original_weights_row6': before.head2.weights[0, 5].tolist(),
            'head2_changed_weights_row6': after.head2.weights[0, 5].tolist(),
        }
        traces[name] = (before, after)

    # 第 24 行没有窗口内的未来；开关遮罩不应改变它的计算结果。
    for unmasked, masked in zip(traces['mask_off'], traces['mask_on']):
        torch.testing.assert_close(unmasked.output[:, 23], masked.output[:, 23])

    raw = data['raw']['val']
    index = int(raw['target_index'][0])
    times = data['df'].date.iloc[index - 24:index]
    observation = {
        'seed': SEED, 'device': 'cpu', 'mode': 'eval', 'trained_heads': False,
        'embedding_checkpoint': str(CHECKPOINT.relative_to(ROOT)),
        'head_parameters': 'same initialization as practice4 reinitialized case',
        'with_position': False, 'mean': data['mean'], 'std': data['std'],
        'window': {'target_index': index, 'target_time': str(raw['target_time'][0]),
                   'first_time': str(times.iloc[0]), 'last_time': str(times.iloc[-1])},
        'perturbation': {'hour': 18, 'python_index': 17, 'time': str(times.iloc[17]),
                         'standardized_increase': 1.0,
                         'raw_temperature_increase': data['std'],
                         'original_standardized': original[0, 17, 0].item(),
                         'changed_standardized': changed[0, 17, 0].item()},
        'query_hour': 6, 'compared_output_shape': [16], 'cases': cases,
    }
    (RESULTS / 'practice5_observations.json').write_text(
        json.dumps(observation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / 'practice5_comparison.csv', index=False)
    pd.DataFrame({
        'hour': range(1, 25), 'time': times.to_numpy(),
        'original_standardized': original[0, :, 0].numpy(),
        'changed_standardized': changed[0, :, 0].numpy(),
    }).to_csv(RESULTS / 'practice5_inputs.csv', index=False)
    print(f'验证目标时间：{raw["target_time"][0]}；查询第 6 小时，只改第 18 小时。')
    print(table.to_string(index=False))
    print('双头未来权重为零、权重行和、因果不变性及末行遮罩开关核对均通过。')
    print(f'结果保存于：{RESULTS}')


if __name__ == '__main__':
    main()
