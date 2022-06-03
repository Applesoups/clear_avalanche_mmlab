import torch
from mmcv.runner.base_module import BaseModule
from torch import nn
from mmcls.models.builder import build_backbone, build_neck, build_head
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
            model_cfg = copy.deepcopy(torchmodel)
            model_type = getattr(nn, model_cfg.pop('type'))
            self.model = model_type(**model_cfg)
        else:
            self.model = None
        if backbone is not None:
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = None
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        if head is not None:
            self.head = build_head(head)
        else:
            self.head = None

    def forward(self, x):
        if self.backbone:
            x = self.backbone(x)
        if self.neck:
            x = self.neck(x)
        if self.head:
            x = self.head.simple_test(x, softmax=False, post_process=False)
        return x


def Build_model(cfg):
    """Build model for continual learning."""
    model_cfg = copy.deepcopy(cfg.model)
    model = Complete_Model(**model_cfg)

    checkpoint = cfg.get('load_from', '')
    if checkpoint != '':
        print('Loading checkpoint from', checkpoint)
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict_modify = dict()
        for key in state_dict.keys():
            if 'head.' not in key:
                state_dict_modify[key] = state_dict[key]
        model.load_state_dict(state_dict_modify, strict=False)

    return model
