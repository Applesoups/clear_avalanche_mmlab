import torch
from mmcv.runner.base_module import BaseModule
from torch import nn
from mmcls.models.builder import build_backbone, build_neck, build_head
import copy
from models import *


class Complete_Model(BaseModule):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 # torchmodel=None,
                 init_cfg=None):
        super(Complete_Model, self).__init__(init_cfg)
        # if torchmodel is not None:
        #     model_cfg = copy.deepcopy(torchmodel)
        #     model_type = getattr(nn, model_cfg.pop('type'))
        #     self.model = model_type(**model_cfg)
        # else:
        #     self.model = None
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
        if len(state_dict) == 1:
            state_dict = state_dict[list(state_dict)[0]]
        state_dict_modify = dict()
        for key in state_dict.keys():
            if 'head.' not in key:
                new_key = key if key.startswith('backbone.') else 'backbone.' + key
                state_dict_modify[new_key] = state_dict[key]
        incompatible_keys = model.load_state_dict(state_dict_modify, strict=False)
        if incompatible_keys.missing_keys:
            missing_keys = incompatible_keys.missing_keys
            print('Warning: Error(s) in loading state_dict for CompleteModel:')
            print('\tMissing key(s) in state_dict:', ', '.join(k for k in missing_keys), end='.\n')
            for name, param in model.named_parameters():
                # update not pretrained and head parameters
                if name in missing_keys or 'head.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    return model
