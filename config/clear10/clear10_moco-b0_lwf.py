_base_ = [
    '../_base_/clear10.py',
    '../_base_/lwf.py'
]

name = 'clear10_moco-b0_lwf'

scenario = dict(feature_type='moco_b0')

model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=11,
        in_channels=2048))

work_dir = f'./work_dirs/{name}'
loggers = [
    dict(type='TextLogger', file=f'{work_dir}/log.txt'),
    dict(type='InteractiveLogger'),
    #dict(type='WandBLogger', project_name='avalanche', run_name=name)
]
