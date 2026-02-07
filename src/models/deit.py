"""
DeiT-Small wrapper for PCam center-region detection.
"""

from typing import List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

logger = logging.getLogger(__name__)


def _resize_pos_embed(pos_embed: torch.Tensor, grid_size: int) -> torch.Tensor:
    cls_token = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    orig_size = int(patch_pos.shape[1] ** 0.5)
    if orig_size == grid_size:
        return pos_embed
    patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(grid_size, grid_size), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, -1)
    return torch.cat([cls_token, patch_pos], dim=1)


class DeiTSmallPCam(nn.Module):
    """
    DeiT-Small/16 adapted to 96x96 input.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        freeze_stages: int = 6,
    ):
        super().__init__()
        self.model = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=0)
        self.embed_dim = self.model.num_features
        self.grid_size = 6  # 96 / 16

        self.model.pos_embed = nn.Parameter(_resize_pos_embed(self.model.pos_embed, self.grid_size))

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

        self._freeze_stages(freeze_stages)

    def _freeze_stages(self, n_blocks: int):
        if n_blocks <= 0:
            return
        for block in self.model.blocks[:n_blocks]:
            for param in block.parameters():
                param.requires_grad = False
        logger.info(f"Froze first {n_blocks} DeiT blocks")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        cls = x[:, 0]
        return self.head(cls)


def create_deit_small(config: dict) -> DeiTSmallPCam:
    return DeiTSmallPCam(
        pretrained=config.get("pretrained", True),
        num_classes=config.get("num_classes", 1),
        dropout=config.get("dropout", 0.3),
        freeze_stages=config.get("freeze_stages", 0),
    )
