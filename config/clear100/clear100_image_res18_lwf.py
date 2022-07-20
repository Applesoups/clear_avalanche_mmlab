_base_ = [
    '../_base_/clear100.py',
    '../_base_/lwf.py'
]

import numpy as np
timestamp = 11
alpha = np.linspace(0, 2, num=timestamp).tolist()

name = 'clear100_image_res18_lwf_new'
num_epochs = 70
buffer_size = 10000  # TODO: different bucket sizes for clear100?


model = dict(
    backbone=dict(type='ResNet', depth=18),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512))

cl_strategy = dict(
    alpha=alpha,
    train_epochs=num_epochs,
    optimizer=dict(lr=0.01, weight_decay=1e-5),
    scheduler=dict(step_size=30))

scenario = dict(dataset_root='dataset/testing/')

work_dir = f'./work_dirs/{name}'
loggers = [
    dict(type='TextLogger', file=f'{work_dir}/log.txt'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='clear100-date', run_name=name,params=dict(entity='0shot'))
]
