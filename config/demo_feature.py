from cv2 import PARAM_SCALAR


dataset_type = 'CLEAR'
feature='moco_b0'
class_number=11
batch_size=256
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
    feature_type = feature,
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
    torchmodel=dict(
        type='Linear',
        in_features=2048,
        out_features=class_number,
    )
)
loggers=[
    dict(type='TensorboardLogger'),
    dict(type='TextLogger', file='log.txt'),
    dict(type='InteractiveLogger'),
    #dict(type='WandBLogger', project_name='avalanche', run_name='clear_feature')
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
    type='Naive',
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=10,
    optimizer = dict(
        type='AdamW',
        lr=5e-4*batch_size/512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    scheduler=dict(
        type='StepLR',
        step_size=60,
        gamma=0.1),
    loss = dict(
        type='CrossEntropyLoss',
        loss_weight=1.0
    )
)

save_model=dict(
    model_root='models',
    frequency=10
)