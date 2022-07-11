_base_ = [
    '../_base_/clear10.py',
    '../_base_/naive.py'
]

name = 'clear10_r50'
batch_size = 64
num_epochs = 70

model = dict(
    backbone=dict(type='ResNet', depth=50),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=11,
        in_channels=2048))

cl_strategy = dict(
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=num_epochs,
    optimizer=dict(lr=0.01, weight_decay=1e-5),
    scheduler=dict(step_size=30))

work_dir = f'./work_dirs/{name}'
loggers = [
    dict(type='TextLogger'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='avalanche', run_name=name)
]
