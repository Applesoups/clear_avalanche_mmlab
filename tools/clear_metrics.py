import numpy as np
from avalanche.benchmarks.classic.clear import CLEARMetric


def compute_clear_metrics(test_metrics: dict, num_exp: int = 10):
    assert len(test_metrics) == num_exp
    res = np.zeros((num_exp, num_exp), dtype=float)
    for index in range(num_exp):
        metrics = test_metrics[index]
        for i in range(num_exp):
            acc = metrics[f'Top1_Acc_Exp/eval_phase/test_stream/Task{i:0>3d}/Exp{i:0>3d}']
            res[index][i] = acc

    cm = CLEARMetric()
    clear_metric = cm.get_metrics(res)

    return clear_metric
