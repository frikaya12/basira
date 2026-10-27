"""Réseau de neurones : un réseau convolutif pré-entraîné sur ImageNet, dont la
dernière couche est remplacée par une seule sortie (rétinopathie à adresser ou non).
"""

from __future__ import annotations

from torch import nn

ARCHITECTURES = ("resnet18", "efficientnet_b0")


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    from torchvision import models

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Architecture inconnue : {arch}. Choix : {ARCHITECTURES}")
    return model
