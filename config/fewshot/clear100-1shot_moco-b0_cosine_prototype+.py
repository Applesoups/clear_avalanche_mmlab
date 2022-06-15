_base_ = [
    '../_base_/clear100.py',
    '../_base_/prototype.py'
]

name = 'clear100-1shot_moco-b0_cosine_prototype+'

shot = 1
scenario = dict(
    feature_type='moco_b0',
    evaluation_protocol='iid',
    seed=0,
    shot=shot)

cl_strategy = dict(
    shot=shot,
    cumulate=True)

model = dict(
    head=dict(
        type='CosineDistanceHead',
        num_classes=100,
        in_channels=2048))

work_dir = f'./work_dirs/{name}'
loggers = [
    dict(type='TextLogger', file=f'{work_dir}/log.txt'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='clear100', run_name=name,
         params=dict(entity='clear_avalanche_mmlab'))
]
