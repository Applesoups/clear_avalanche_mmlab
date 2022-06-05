dataset_type = 'CLEAR'
num_classes = 11

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
train_pipeline = [
    dict(type='Resize', size=224),
    dict(type='RandomCrop', size=224),
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

scenario = dict(
    dataset_type=dataset_type,
    train_transform=train_pipeline,
    eval_transform=test_pipeline,
    dataset_root='dataset/CLEAR-10-PUBLIC/',
    evaluation_protocol="streaming",
    seed=None)

metrics = [
    dict(type='accuracy_metrics', minibatch=True, epoch=True, experience=True, stream=True),
    dict(type='loss_metrics', minibatch=True, epoch=True, experience=True, stream=True),
    dict(type='forgetting_metrics', experience=True, stream=True),
    # dict(type='confusion_matrix_metrics', num_classes=num_classes, save_image=True, stream=True),
]
