from avalanche import models
from mmcv.runner.base_module import BaseModule
from mmcls.models.builder import CLASSIFIERS, build_backbone,  build_neck
from . import heads
import copy

class Complete_Model(BaseModule):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None):
        super(Complete_Model, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck=None
        if head is not None:
            head_cfg=copy.deepcopy(head)
            head_type=getattr(heads, head_cfg.pop('type'))
            self.head=head_type(**head_cfg)
        else:
            self.head=None
    def forward(self,x):
        x=self.backbone(x)
        if self.neck:
            x=self.neck(x)
        if self.head:
            x=self.head(x)
            
        return x
        


    

def Build_model(cfg):
    model_cfg=copy.deepcopy(cfg.model)
    model=Complete_Model(**model_cfg)
    return model
