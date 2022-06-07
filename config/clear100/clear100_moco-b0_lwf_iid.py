import numpy as np

_base_ = [
    '../_base_/clear100.py',
    '../_base_/lwf.py'
]

batch_size = 256
num_epochs = 200

timestamp = 11
alpha = np.linspace(0, 2, num=timestamp).tolist()

name = 'clear100_moco-b0_lwf_iid'

scenario = dict(
    feature_type='moco_b0',
    evaluation_protocol='iid',
    seed=0)

model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=101,
        in_channels=2048))

cl_strategy = dict(
    alpha=alpha,
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=num_epochs,
)

work_dir = f'./work_dirs/{name}'
loggers = [
    dict(type='TensorboardLogger'),
    dict(type='TextLogger', file=f'{work_dir}/log.txt'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='avalanche', run_name=name)
]
