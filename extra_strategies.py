from typing import Optional, List

import torch
from torch.nn import Module
from torch.optim import SGD

from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from tools.build_model import Complete_Model


class Prototype(SupervisedTemplate):
    """Prototype based strategy.

    At each experience the model is reset and the prototype
    (mean training feature) of each class is directly assigned
    to the head weight.
    This strategy only supports pre-trained feature forward.
    The model must contain a head with a fc layer.
    This strategy does not use task identities.
    """

    def __init__(self,
                 model: Module,
                 shot: int = 5,
                 cumulate: bool = False,
                 train_mb_size: int = None,
                 eval_mb_size: int = None,
                 device=None,
                 plugins: Optional[List[SupervisedPlugin]] = None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1,
                 **base_kwargs):
        self.num_classes = model.head.fc.weight.shape[0]
        self.shot = shot
        self.cumulate = cumulate
        train_mb_size = train_mb_size or shot * self.num_classes

        super().__init__(
            model,
            optimizer=SGD(model.parameters(), 0.01),
            train_mb_size=train_mb_size,
            train_epochs=1,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs)

    def training_epoch(self, **kwargs):
        """Assign class prototypes to model weights."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            # Forward to compute metrics
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Just to make sure self.loss is not None
            self.loss = self.criterion()

            # self.mb_x is the feature
            centroids = get_centroids_from_features(
                self.mb_x, self.mb_y, self.num_classes)
            self.model = update_model_by_centroids(
                self.model, centroids, self.cumulate)

            self._after_training_iteration(**kwargs)


def get_centroids_from_features(
        features: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = None):
    """Get centroids from features.

    Args:
        features (torch.Tensor): Extracted features.
        targets (torch.Tensor): Labels.
        num_classes (int): Number of classes.
    """
    assert features.shape[0] == targets.shape[0]
    if num_classes is None:
        num_classes = torch.max(targets).item() + 1

    centroids = torch.cat([
        features[targets == class_id].mean(0, keepdim=True)
        for class_id in range(num_classes)
    ], dim=0)

    return centroids


def update_model_by_centroids(
        model: Complete_Model,
        centroids: torch.Tensor,
        cumulate: bool = False):
    """Update model by centroids.

    Args:
        model (Complete_Model): Model to be updated.
        centroids (torch.Tensor): Centroids.
        cumulate (bool): Whether to cumulate the centroids.
    """
    try:
        if cumulate:
            model.head.fc.weight.data += centroids
        else:
            model.head.fc.weight.data = centroids
        model.head.fc.weight.requires_grad = False
    except AttributeError:
        print('No head found in the model.')

    return model
