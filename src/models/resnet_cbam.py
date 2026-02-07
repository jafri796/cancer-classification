"""
ResNet50 with CBAM attention for PCam center-region detection.
"""

from typing import List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)


class CBAMChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn).view(b, c, 1, 1)
        return x * attn


class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attn = torch.sigmoid(self.conv(pooled))
        return x * attn


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = CBAMChannelAttention(channels, reduction)
        self.spatial = CBAMSpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ResNet50CBAM(nn.Module):
    """
    ResNet50 backbone with CBAM inserted after conv4_x and conv5_x blocks.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.5,
        hidden_dims: List[int] = [512],
        cbam_reduction: int = 16,
        freeze_stages: int = 4,
    ):
        super().__init__()
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            logger.info("Loaded pretrained ResNet50 for CBAM")
        else:
            backbone = resnet50(weights=None)
            logger.info("Initialized ResNet50 from scratch for CBAM")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.cbam4 = CBAMBlock(1024, cbam_reduction)
        self.cbam5 = CBAMBlock(2048, cbam_reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)

        classifier_layers = []
        in_features = 2048
        for hidden in hidden_dims:
            classifier_layers.extend(
                [
                    nn.Linear(in_features, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden
        classifier_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self._freeze_stages(freeze_stages)

    def _freeze_stages(self, n_stages: int):
        if n_stages == 0:
            return
        stages = [
            [self.conv1, self.bn1],
            [self.layer1],
            [self.layer2],
            [self.layer3],
            [self.layer4],
        ]
        for i in range(min(n_stages, len(stages))):
            for module in stages[i]:
                for param in module.parameters():
                    param.requires_grad = False
        logger.info(f"Froze first {n_stages} stages for ResNet50-CBAM")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam4(x)
        x = self.layer4(x)
        x = self.cbam5(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)


def create_resnet50_cbam(config: dict) -> ResNet50CBAM:
    model = ResNet50CBAM(
        pretrained=config.get("pretrained", True),
        num_classes=config.get("num_classes", 1),
        dropout=config.get("dropout", 0.5),
        hidden_dims=config.get("hidden_dims", [512]),
        cbam_reduction=config.get("cbam_reduction", 16),
        freeze_stages=config.get("freeze_stages", 0),
    )
    return model
