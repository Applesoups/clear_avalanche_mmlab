_base_ = [
    '../_base_/clear100.py',
    '../_base_/agem.py'
]

name = 'clear100_moco-b0_agem-reservoir'

num_epochs = 200
buffer_size = 10000  # TODO: different bucket sizes for clear100?

scenario = dict(feature_type='moco_b0')

cl_strategy = dict(
    train_epochs=num_epochs,
    patterns_per_exp=buffer_size,
    sample_size=buffer_size)

model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048))

work_dir = f'./work_dirs/{name}'
loggers = [
    dict(type='TextLogger', file=f'{work_dir}/log.txt'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='clear100', run_name=name,
         params=dict(entity='clear_avalanche_mmlab'))
]
