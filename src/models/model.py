"""ResNet-18 binary classifier (good vs defective) with optional layer freezing."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes: int = config.NUM_CLASSES,
                pretrained: bool = True,
                freeze_until: str | None = config.FREEZE_BACKBONE_UNTIL) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with a fresh binary head.

    freeze_until: name of the last ResNet block to FREEZE (inclusive). With small
    datasets, freezing early blocks (e.g. 'layer3') reduces overfitting AND
    reduces catastrophic forgetting, since fewer weights drift between tasks.
    Set to None to fine-tune the whole network.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    if freeze_until is not None:
        # ResNet block order: conv1, bn1, layer1, layer2, layer3, layer4, fc
        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        if freeze_until not in order:
            raise ValueError(f"freeze_until must be one of {order} or None")
        cutoff = order.index(freeze_until)
        for name, module in net.named_children():
            if name in order and order.index(name) <= cutoff:
                for p in module.parameters():
                    p.requires_grad = False
    return net
