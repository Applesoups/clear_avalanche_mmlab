_base_ = [
    '../_base_/clear10.py',
    '../_base_/lwf.py'
]

name = 'clear10_image_res18_lwf'
batch_size = 64
num_epochs = 70

model = dict(
    backbone=dict(type='ResNet', depth=18),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=11,
        in_channels=512))

cl_strategy = dict(
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=num_epochs,
    optimizer=dict(lr=0.01, weight_decay=1e-5),
    scheduler=dict(step_size=30))

work_dir = f'./work_dirs/{name}'
loggers = [
    dict(type='TextLogger', file=f'{work_dir}/log.txt'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='avalanche', run_name=name)
]
