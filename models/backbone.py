"""
ResNet backbone (18/101) that extracts 4-scale feature maps (res2-res5).
"""

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet101_Weights,
    ResNet18_Weights,
    resnet101,
    resnet18,
)


BACKBONE_SPECS = {
    "resnet18": {
        "builder": resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "out_channels": {"res2": 64, "res3": 128, "res4": 256, "res5": 512},
    },
    "resnet101": {
        "builder": resnet101,
        "weights": ResNet101_Weights.IMAGENET1K_V2,
        "out_channels": {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048},
    },
}


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet101", pretrained: bool = True):
        super().__init__()
        if name not in BACKBONE_SPECS:
            raise ValueError(
                f"Unsupported backbone '{name}'. "
                f"Choose from: {list(BACKBONE_SPECS.keys())}"
            )

        spec = BACKBONE_SPECS[name]
        weights = spec["weights"] if pretrained else None
        base = spec["builder"](weights=weights)

        self.name = name
        self.out_channels = spec["out_channels"]

        # Stem: conv1 + bn1 + relu + maxpool -> stride 4
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # res2
        self.layer2 = base.layer2  # res3
        self.layer3 = base.layer3  # res4
        self.layer4 = base.layer4  # res5

    def forward(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}


class ResNet18Backbone(ResNetBackbone):
    out_channels = BACKBONE_SPECS["resnet18"]["out_channels"]

    def __init__(self, pretrained: bool = True):
        super().__init__(name="resnet18", pretrained=pretrained)


class ResNet101Backbone(ResNetBackbone):
    out_channels = BACKBONE_SPECS["resnet101"]["out_channels"]

    def __init__(self, pretrained: bool = True):
        super().__init__(name="resnet101", pretrained=pretrained)
