from avalanche import models
from mmcv.runner.base_module import BaseModule
from torch import nn
from mmcls.models.builder import CLASSIFIERS, build_backbone,  build_neck
from . import heads
import copy

class Complete_Model(BaseModule):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 torchmodel=None,
                 init_cfg=None):
        super(Complete_Model, self).__init__(init_cfg)
        if torchmodel is not None:
            model_cfg=copy.deepcopy(torchmodel)
            model_type=getattr(nn, model_cfg.pop('type'))
            self.model=model_type(**model_cfg)
        else:
            self.model=None
        if neck is not None:
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = None
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck=None
        if head is not None:
            head_cfg=copy.deepcopy(head)
            head_type=getattr(heads, head_cfg.pop('type'))
            keys=head_cfg.copy().keys()
            for key in keys:
                if key not in ['num_classes','in_channels','init_cfg']:
                    head_cfg.pop(key)
            self.head=head_type(**head_cfg)
        else:
            self.head=None
    def forward(self,x):
        if self.model:
            return self.model(x)
        if self.backbone:
            x=self.backbone(x)
        if self.neck:
            x=self.neck(x)
        if self.head:
            x=self.head(x)
            
        return x
        


    

def Build_model(cfg):
    model_cfg=copy.deepcopy(cfg.model)
    if 'type' in model_cfg.keys():
        model_cfg.pop('type')
    model=Complete_Model(**model_cfg)
    return model
