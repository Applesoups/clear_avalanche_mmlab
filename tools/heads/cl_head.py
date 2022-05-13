from mmcv.runner.base_module import BaseModule
import torch.nn as nn

class LinearClsHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)        
    
    def forward(self,x):
        x=x[-1]
        x=self.fc(x)
        
        return x