import numpy as np

_base_ = ["./models/resnet50.py"]
'''
some special settings for cl_strategy
timestamp: 
    when strate=='JointTraining' and current_mode=='offline':
    train_epochs=args.nepoch*args.timestamp//3
    when strate=='LwF': alpha= np.linspace(0,2,num=args.timestamp).tolist()#算出来也没用上啊
nepoch: for most cases, train_epochs=nepoch

step_schedular_decay: step_size in schedular
schedular_step: gamma in schedular
start_lr: lr in optimizer
'''

timestamp = 10
alpha=np.linspace(0, 2, num=timestamp).tolist()
temperature=1
nepoch=70

dataset_type = 'CLEAR'
#feature='moco_b0'
class_number=11
batch_size=32
img_norm_cfg=dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]

test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]

scenario=dict(
    dataset_type = dataset_type,
    train_transform=train_pipeline,
    eval_transform=test_pipeline,
    dataset_root='dataset/CLEAR-10-PUBLIC/',
    evaluation_protocol= "streaming",
    seed = None
    )

# model=dict(
#     name='SLDAResNetModel',
#     cfgs=dict(
#         arch='resnet50',
#         output_layer_name="layer4.1",
#         imagenet_pretrained=False,
#     )
# )

model=dict(
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(3, ),
    #     style='pytorch',
    #     #norm_cfg=dict(type='SyncBN', requires_grad=True)
    # ),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=class_number,
        in_channels=2048,
        init_cfg=dict(type='Normal', layer='Linear', std=0.01),
        
    )
)
loggers=[
    dict(type='TensorboardLogger'),
    dict(type='TextLogger', file='log.txt'),
    dict(type='InteractiveLogger'),
    #dict(type='WandBLogger', project_name='avalanche', run_name='clear_resnet50')
]

metrics=[
    dict(type='accuracy_metrics',minibatch=True, epoch=True, experience=True, stream=True),
    dict(type='loss_metrics',minibatch=True,epoch=True,experience=True,stream=True),
    dict(type='timing_metrics',epoch=True,epoch_running=True),
    dict(type='cpu_usage_metrics',experience=True),
    dict(type='forgetting_metrics',experience=True, stream=True),
    dict(type='confusion_matrix_metrics',num_classes=class_number, save_image=True, stream=True),
    dict(type='disk_usage_metrics',minibatch=True, epoch=True, experience=True, stream=True)
]



# 暂时无法实现lr_config的功能
# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     min_lr_ratio=1e-2,
#     warmup='linear',
#     warmup_ratio=1e-3,
#     warmup_iters=20,
#     warmup_by_epoch=True)

cl_strategy=dict(
    type='LwF',
    alpha=alpha,
    temperature=temperature,
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=nepoch,
    optimizer = dict(
        type='SGD',
        lr=0.000625,
        weight_decay=1e-5,
        momentum=0.9
    )
    scheduler=dict(
        type='StepLR',
        step_size=30,
        gamma=0.1),
    #optimizer_config = dict(grad_clip=dict(max_norm=5.0))
    loss = dict(
        type='CrossEntropyLoss',
        loss_weight=1.0
    )
)

save_model=dict(
    model_root='models',
    frequency=2
)