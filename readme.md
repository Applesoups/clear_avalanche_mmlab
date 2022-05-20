gi# Demo for CLEAR Dataset Training Based on Avalanche and Openmmlab

## Install

首先安装pytorch

### 安装avalanche

安装avalanche使请不要直接使用pip进行安装，pip的版本更新相较于github版本更慢，当前版本并不支持clear数据集

安装教程请看：

https://avalanche.continualai.org/getting-started/how-to-install

跟随教程中的Installing the Master Branche Using Anaconda进行安装

### 安装mmlab

mmlab中首先安装mim

https://github.com/open-mmlab/mim

然后安装mmclassification（不使用mim安装的原因为其源码需要经常查看，直接git clone下来比较方便）

https://github.com/open-mmlab/mmclassification

最后使用mim安装mmcv（见mim的readme）

### 其他requiements

（之后补整理一份requirements.txt，现在先缺啥补啥吧）

## 数据集构建

原数据集架构avalanche不能直接读取，但CLEAR官方给出了支持avalanche的教程

https://github.com/linzhiqiu/continual-learning/#prepare-folders-for-avalanche-training

跟随官方教程构建完毕即可

## 运行

测试用的config文件为config/demo.py

运行方式：

```shell
python train_config.py config/demo.py  --device cuda
```

## config简要介绍

代码整体框架依照avalanche框架进行搭建，config中将各个模块可调参数拆分出来进行控制

### train and test pipeline

使用的是torchvision.transforms中的模块，config依照其中各个class的需求构建即可

```python
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
```

### scenario

scenario需要提供dataset_type，为avalanche.benchmarks.classic中提供的类别。由于每一类传参存在差距，所以暂时只支持CLEARshujuji的scenario的构建

通过修改dataset_root来修改dataset所在的位置

```python 
scenario=dict(
    dataset_type = dataset_type,
    train_transform=train_pipeline,
    eval_transform=test_pipeline,
    dataset_root='dataset/CLEAR-10-PUBLIC/',
    evaluation_protocol= "streaming",
    #feature_type = feature,
    seed = None
    )
```

### model

model使用的是mmcls中的model架构，由backbone，neck和head组成。head部分与mmlab略有区别，不包括loss损失函数，其他参数和mmlab的config构建方式相同。

暂不支持从mmlab的config中继承（该部分之后会完成），所以当前只能从config/models中复制粘贴backbone和neck的部分，重写head的部分

重构后的head在tools/heads文件夹中可进行查看

```python 
model=dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=class_number,
        in_channels=2048,
        init_cfg=dict(type='Normal', layer='Linear', std=0.01)
    )
```

若想要直接使用torch.nn中的model，则不用提供backbone，neck和head

```python
model=dict(
    torchmodel=dict(
        type='Linear',
        in_features=2048,
        out_features=class_number,
    )
)
```



### loggers&metrics

使用的是avalanche的loggers和metrics，具体功能请查看avalanche教程。

tips：对于TextLogger，区别于原函数，只需要提供file的名字即可

```python
loggers=[
    dict(type='TensorboardLogger'),
    dict(type='TextLogger', file='log.txt'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='avalanche', run_name='clear_resnet50')
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
```

### cl_strategy

需要提供strategy type，使用的是avalanche.training.supervised（老版本的库位置会不同）中的strategy，具体内容请看教程和continual ai提供的continual learning course。其中需要提供optimizer，scheduler和loss，optimizer和scheduler使用的是torch.optim库，losses使用的是mmcls.models库。

```python
cl_strategy=dict(
    type='Naive',
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=100,
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
```

### checkpoint 保存

增加了checkpoint的保存设置

```python
save_model=dict(
    model_root='models',
    frequency=2
)
```

文件会被保存在model_root下的当前时间文件夹下

