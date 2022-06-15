from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead


@HEADS.register_module()
class CosineDistanceHead(ClsHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 temperature: Optional[float] = None,
                 eps: float = 0.00001,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        if temperature is None:
            self.temperature = 2.
        else:
            self.temperature = temperature
        self.eps = eps

        self.fc = nn.Linear(self.in_channels, self.num_classes, bias=False)
        # self.fc = nn.utils.weight_norm(self.fc, name='weight', dim=0)
        # weight_norm is not supported for deepcopy

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        w = self.fc.weight
        cls_score = self.temperature * torch.mm(
            F.normalize(x),
            F.normalize(w).transpose(0, 1))

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred
