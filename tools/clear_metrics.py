import numpy as np
from avalanche.benchmarks.classic.clear import CLEARMetric


def compute_clear_metrics(test_metrics: dict, **kwargs):
    num_exp = len(test_metrics)
    res = np.zeros((num_exp, num_exp), dtype=float)
    for index in range(num_exp):
        metrics = test_metrics[index]
        for i in range(num_exp):
            try:
                acc = metrics[f'Top1_Acc_Exp/eval_phase/test_stream/Task{i:0>3d}/Exp{i:0>3d}']
            except KeyError:
                acc = metrics[f'Top1_Acc_Stream/eval_phase/test_stream/Task{i:0>3d}']
            res[index][i] = acc

    cm = CLEARMetric()
    clear_metric = cm.get_metrics(res)

    # other metrics
    for key in kwargs:
        clear_metric[key] = eval(f'{key}(res)')

    return clear_metric


def final_domain(matrix):
    r, _ = matrix.shape
    res = [matrix[i, -1] for i in range(r - 1)]
    return sum(res) / (r - 1)
