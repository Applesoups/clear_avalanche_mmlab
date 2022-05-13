from torchvision import transforms
from torchvision.transforms import Compose
#from mmcls.datasets.pipelines import Compose
#from mmcls.datasets import pipelines
from mmcv.utils import build_from_cfg
from avalanche.benchmarks import classic
import copy

def Build_scenario(cfg):
    scenario_cfg=copy.deepcopy(cfg.scenario)
    train_transforms=[]
    eval_transforms=[]
    
    for transform in scenario_cfg.pop('train_transform'):
        trans_type=getattr(transforms,transform.pop('type'))
        train_transforms.append(trans_type(**transform))
    for transform in scenario_cfg.pop('eval_transform'):
        trans_type=getattr(transforms,transform.pop('type'))
        eval_transforms.append(trans_type(**transform))
    train_pipeline=Compose(transforms=train_transforms)
    eval_pipeline=Compose(transforms=eval_transforms)
    
    scenario=getattr(classic,scenario_cfg.pop('dataset_type'))
    return scenario(train_transform=train_pipeline,  
                    eval_transform=eval_pipeline,
                    **scenario_cfg)
    