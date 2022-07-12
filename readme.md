# Demo for CLEAR Dataset Training Based on Avalanche and Openmmlab

## Install

首先安装pytorch

### 安装avalanche

请安装当前仓库中的avalanche库，参考https://avalanche.continualai.org/getting-started/how-to-install#developer-mode-install

### 安装mmlab

mmlab中首先安装mim：

https://github.com/open-mmlab/mim

然后安装mmclassification（不使用mim安装的原因为其源码需要经常查看，直接git clone下来比较方便）：

https://github.com/open-mmlab/mmclassification

最后使用mim安装mmcv（见mim的readme）。

### 其他requiements

（之后补整理一份requirements.txt，现在先缺啥补啥吧）

## 数据集构建

原数据集架构avalanche不能直接读取，但CLEAR官方给出了支持avalanche的教程：

https://github.com/linzhiqiu/continual-learning/#prepare-folders-for-avalanche-training

跟随官方教程构建完毕即可。

## 运行

测试用的config文件为config/demo.py，运行方式：

```shell
python train_config.py config/demo.py --device cuda
```

## config简要介绍

代码整体框架依照avalanche框架进行搭建，config则按照mmlab的风格，将各个模块可调参数拆分出来进行控制。

### \_base\_

参考mmlab的config格式，可以使用`_base_`列表来包括一些较为重复的参数。

当前文件中的参数会覆盖上述列表中的文件的同名参数。

### train and test pipeline

使用的是torchvision.transforms中的模块，config依照其中各个class的需求构建即可：

```python
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
```

### scenario

scenario需要提供dataset_type，为avalanche.benchmarks.classic中提供的类别。由于每一类传参存在差距，所以暂时只支持CLEAR数据集的scenario的构建

通过修改dataset_root来修改dataset所在的位置

```python
scenario=dict(
    dataset_type='CLEAR',
    data_name='clear10',
    train_transform=train_pipeline,
    eval_transform=test_pipeline,
    dataset_root='dataset/CLEAR-10-PUBLIC/',
    evaluation_protocol= "streaming",
    #feature_type = feature,
    seed = None)
```

### model

model使用的是mmcls中的model架构，由backbone，neck和head组成

```python
model=dict(
    backbone=dict(
        type='ResNet',
        depth=50),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=11,
        in_channels=2048,
        init_cfg=dict(type='Normal', layer='Linear', std=0.01)))
```

### loggers&metrics

使用的是avalanche的loggers和metrics，具体功能请查看avalanche教程。

tips：对于TextLogger，区别于原函数，只需要提供file的名字即可，或不提供file而默认使用当前时间作为文件名

```python
loggers=[
    dict(type='TextLogger'),
    dict(type='InteractiveLogger'),
    dict(type='WandBLogger', project_name='avalanche', run_name='clear_resnet50')
]
metrics=[
    dict(type='accuracy_metrics',minibatch=True, epoch=True, experience=True, stream=True),
    dict(type='loss_metrics',minibatch=True, epoch=True, experience=True, stream=True),
    dict(type='timing_metrics',epoch=True,epoch_running=True),
    dict(type='cpu_usage_metrics',experience=True),
    dict(type='forgetting_metrics',experience=True, stream=True),
    dict(type='confusion_matrix_metrics', num_classes=11, save_image=True, stream=True),
    dict(type='disk_usage_metrics',minibatch=True, epoch=True, experience=True, stream=True)
]
```

### cl_strategy

需要提供strategy type，使用的是avalanche.training.supervised（老版本的库位置会不同）中的strategy，具体内容请看教程和continual ai提供的continual learning course。其中需要提供optimizer，scheduler和loss，optimizer和scheduler使用的是torch.optim库，losses使用的是mmcls.models库。

```python
batch_size = 64
cl_strategy=dict(
    type='Naive',
    train_mb_size=batch_size,
    eval_mb_size=batch_size,
    train_epochs=70,
    optimizer = dict(
        type='SGD',
        lr=0.01 * batch_size / 64,
        weight_decay=1e-5),
    scheduler=dict(
        type='StepLR',
        step_size=30,
        gamma=0.1),
    loss = dict(
        type='CrossEntropyLoss',
        loss_weight=1.0))
```

### work_dir

需要提供工作路径，用于存储模型，训练结果等。

### checkpoint

若需要保存checkpoint，则提供interval即可，文件会保存在work_dir：

```python
checkpoint_config = dict(interval=10)
```

若需要从checkpoint开始训练，则提供load_from：

```python
load_from = 'ckpt.pth'
```

## 进阶使用

### 使用自己的pretrained model

当训练中需要固定feature时，模型forward的时间会很长，可以先生成自己的pretrained feature来加速训练。在python console运行：

```python
from tools.clear_dataset import get_pretrained_features
get_pretrained_features(
    'dataset/CLEAR-10-PUBLIC',
    'intern_r50'
    'intern-r50.pth',
    arch='resnet50'  # 
)
```

### few-shot learning

可以自行构建clear的few-shot scenario。在python console运行：

```python
from tools.clear_dataset import get_fewshot_indices
get_fewshot_indices(
    'dataset/CLEAR-10-PUBLIC',
    shot=5,
    seed=0
)
```

同时修改config：

```python
scenario=dict(
    evaluation_protocol= 'iid',
    shot=5,
    seed=0)
```

### 根据metadata构建新的数据集

可以从metadata构建新的数据集，如按`date_taken`字段排序。在python console运行：

```python
from tools.clear_dataset import create_dataset
from functools import partial
create_dataset(
    'dataset/CLEAR-10-PUBLIC',  # 原本的数据集路径，包含labeled_metadata
    'dataset/clear-10-date',  # 新的路径
    process=partial(sorted, key=lambda x: x['date_taken'])
)
```
