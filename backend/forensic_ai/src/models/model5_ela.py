"""
Model 5: Error Level Analysis (ELA) Forensics Network
Input:  3xHxW ELA map (precomputed or computed on-the-fly)
Output: ELA-based manipulation probability + embedding
No hand-crafted thresholds — fully learned CNN.
"""

import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from typing import Tuple, Optional


class ELAMultiQuality(nn.Module):
    """
    Computes ELA at multiple JPEG quality levels and fuses them.
    Different manipulation types are detectable at different quality levels.
    """

    def __init__(self, qualities: Tuple[int, ...] = (70, 80, 90, 95)):
        super().__init__()
        self.qualities = qualities
        n_maps = len(qualities) * 3  # 3 channels per quality
        self.fusion = nn.Sequential(
            nn.Conv2d(n_maps, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 1, bias=False),
            nn.Sigmoid(),
        )

    def _compute_ela_tensor(
        self, image_batch_np: np.ndarray, quality: int, scale: float = 10.0
    ) -> torch.Tensor:
        """
        image_batch_np: B x H x W x 3  uint8 numpy array
        Returns: B x 3 x H x W float tensor
        """
        ela_maps = []
        for img_np in image_batch_np:
            img_pil = Image.fromarray(img_np)
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            recomp = Image.open(buf).convert("RGB")
            ela = np.abs(
                np.array(img_pil, dtype=np.float32) - np.array(recomp, dtype=np.float32)
            )
            ela = np.clip(ela * scale, 0, 255) / 255.0  # normalize [0, 1]
            ela_maps.append(ela.transpose(2, 0, 1))      # 3 x H x W
        return torch.from_numpy(np.stack(ela_maps)).float()

    def forward(self, images_np: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Returns: B x 3 x H x W fused multi-quality ELA
        """
        ela_list = []
        for q in self.qualities:
            ela = self._compute_ela_tensor(images_np, q).to(device)
            ela_list.append(ela)
        combined = torch.cat(ela_list, dim=1)   # B x (Q*3) x H x W
        return self.fusion(combined)            # B x 3 x H x W


class ELASpatialAttention(nn.Module):
    """Spatial attention that focuses on high-ELA regions."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_map: torch.Tensor, ela_map: torch.Tensor) -> torch.Tensor:
        # Resize ela to match feature map
        ela_resized = F.interpolate(
            ela_map, size=feat_map.shape[2:], mode="bilinear", align_corners=False
        )
        # Concatenate and compute attention
        combined = feat_map * ela_resized.mean(dim=1, keepdim=True)
        attn = self.conv(combined)
        return feat_map * attn


class ELAForensicsNet(nn.Module):
    """
    Fully learned ELA forgery detection.
    Backbone: EfficientNet-B2 on ELA maps.
    No thresholds — entirely data-driven.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        embed_dim: int = 512,
        use_multi_quality: bool = False,   # enable for best accuracy (slower)
    ):
        super().__init__()
        self.use_multi_quality = use_multi_quality

        if use_multi_quality:
            self.ela_multi = ELAMultiQuality(qualities=(70, 80, 90, 95))

        # ── Backbone ──
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained,
                num_classes=0, global_pool=""
            )
        except Exception:
            self.backbone = timm.create_model(
                "efficientnet_b2", pretrained=pretrained,
                num_classes=0, global_pool=""
            )

        feat_dim = self.backbone.num_features

        # ── ELA spatial attention ──
        self.ela_spatial_attn = ELASpatialAttention(feat_dim)

        # ── Pooling ──
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # ── Feature head ──
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # ── Localization auxiliary ──
        # Lightweight decoder to produce rough manipulation map
        self.ela_localizer = nn.Sequential(
            nn.Conv2d(feat_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ela_input: torch.Tensor,
        return_features: bool = False,
        images_np: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            ela_input: B x 3 x H x W ELA map (precomputed)
            images_np: optionally provide raw images for multi-quality ELA
        """
        if self.use_multi_quality and images_np is not None:
            ela_input = self.ela_multi(images_np, ela_input.device)

        # Backbone
        feat_map = self.backbone(ela_input)     # B x C x H' x W'

        # ELA spatial attention
        feat_map = self.ela_spatial_attn(feat_map, ela_input)

        # Rough localization map
        loc_map = self.ela_localizer(feat_map)  # B x 1 x H' x W'

        # Pool
        avg    = self.gap(feat_map).flatten(1)
        mx     = self.gmp(feat_map).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)

        embed  = self.feature_proj(pooled)
        logit  = self.classifier(embed)
        prob   = torch.sigmoid(logit).squeeze(1)

        if return_features:
            return prob, embed, loc_map

        return prob, embed, loc_map


class ELAForensicsLoss(nn.Module):
    """
    Loss = BCE detection + localization consistency.
    """

    def __init__(
        self,
        lambda_loc: float = 0.2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_loc = lambda_loc
        self.label_smoothing = label_smoothing

    def forward(
        self,
        prob: torch.Tensor,
        loc_map: torch.Tensor,
        label: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(prob, target)

        losses = {"bce": bce}
        total  = bce

        if gt_mask is not None:
            # Resize gt_mask to loc_map size
            gt_resized = F.interpolate(
                gt_mask.float(), size=loc_map.shape[2:],
                mode="nearest"
            )
            loc_loss = F.binary_cross_entropy(loc_map, gt_resized)
            losses["loc"] = loc_loss
            total = bce + self.lambda_loc * loc_loss

        # Global-local consistency: if label=0, loc_map should be ~0
        fake_mask = label.float().view(-1, 1, 1, 1)
        consistency = F.mse_loss(
            loc_map * (1 - fake_mask),
            torch.zeros_like(loc_map) * (1 - fake_mask)
        )
        losses["consistency"] = consistency
        total = total + 0.05 * consistency

        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    model = ELAForensicsNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)  # ELA maps
    prob, embed, loc_map = model(x)
    print(f"prob: {prob.shape}, embed: {embed.shape}, loc_map: {loc_map.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
