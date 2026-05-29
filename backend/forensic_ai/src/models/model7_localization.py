"""
Model 7: Manipulation Localization Network
Architectures: SegFormer-B4 (primary) + U-Net++ (secondary)
Input:  3xHxW RGB image
Output: pixel-level manipulation mask (1xHxW) + global forgery score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional, List


# ─────────────────────────────────────────────
#  Decoder Building Blocks
# ─────────────────────────────────────────────

class ConvBNGELU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlusBlock(nn.Module):
    """U-Net++ dense skip connection node."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNGELU(in_channels, out_channels),
            ConvBNGELU(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Standard decoder block with upsampling."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBNGELU(in_c + skip_c, out_c),
            ConvBNGELU(out_c, out_c),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Align sizes
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
#  SegFormer-style MLP Decoder
# ─────────────────────────────────────────────

class SegFormerDecoder(nn.Module):
    """
    Lightweight MLP decoder à la SegFormer.
    Takes multi-scale features from encoder.
    """

    def __init__(
        self,
        in_channels: List[int],
        embed_dim: int = 256,
        num_classes: int = 1,
    ):
        super().__init__()
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            for c in in_channels
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(
        self, features: List[torch.Tensor], target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        features: list of B x C_i x H_i x W_i at different scales
        """
        # Project each scale to embed_dim
        projected = []
        for feat, linear in zip(features, self.linear_layers):
            B, C, H, W = feat.shape
            # Reshape to B x HW x C for linear, then back
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            feat_proj = linear(feat_flat).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            # Upsample to largest feature scale
            feat_proj = F.interpolate(
                feat_proj, size=features[0].shape[2:],
                mode="bilinear", align_corners=False
            )
            projected.append(feat_proj)

        fused = self.fuse_conv(torch.cat(projected, dim=1))
        logit = self.seg_head(fused)

        # Upsample to target resolution
        out = F.interpolate(logit, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


# ─────────────────────────────────────────────
#  Main Localization Network
# ─────────────────────────────────────────────

class ManipulationLocalizationNet(nn.Module):
    """
    Pixel-level manipulation localization + image-level classification.
    Primary: SegFormer-style (hierarchical ViT encoder + MLP decoder)
    Also generates image-level score from global features.
    """

    SWIN_CHANNELS = {
        "swin_base_patch4_window7_224":  [128, 256, 512, 1024],
        "swin_large_patch4_window7_224": [192, 384, 768, 1536],
        "swin_tiny_patch4_window7_224":  [96, 192, 384, 768],
    }

    def __init__(
        self,
        backbone: str = "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        pretrained: bool = True,
        embed_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── Encoder ──
        try:
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained,
                features_only=True, out_indices=(0, 1, 2, 3)
            )
            # Get channel counts from feature info
            feat_info = self.encoder.feature_info.info
            in_channels = [f["num_chs"] for f in feat_info]
        except Exception as e:
            print(f"[LocalizationNet] Swin load failed ({e}). Using ResNet50.")
            self.encoder = timm.create_model(
                "resnet50", pretrained=pretrained,
                features_only=True, out_indices=(1, 2, 3, 4)
            )
            in_channels = [256, 512, 1024, 2048]

        self.in_channels = in_channels

        # ── SegFormer MLP decoder ──
        self.decoder = SegFormerDecoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_classes=1,
        )

        # ── U-Net++ style skip connections ──
        # Intermediate dense nodes
        self.dense_nodes = nn.ModuleList([
            UNetPlusPlusBlock(c, c) for c in in_channels[:-1]
        ])

        # ── Image-level classifier from global features ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.image_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels[-1], 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # ── Boundary refinement ──
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def _normalize_features(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Handle Swin Transformer output (B x H x W x C) → (B x C x H x W)."""
        normalized = []
        for f in feats:
            if f.dim() == 4 and f.shape[-1] != f.shape[-2]:
                # Swin outputs B x H x W x C
                f = f.permute(0, 3, 1, 2).contiguous()
            normalized.append(f)
        return normalized

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape

        # Multi-scale features
        features = self.encoder(x)
        features = self._normalize_features(features)

        # Dense skip connections (U-Net++ style enrichment)
        enriched = []
        for i, (feat, node) in enumerate(zip(features[:-1], self.dense_nodes)):
            enriched.append(node(feat))
        enriched.append(features[-1])  # last scale unchanged

        # Segmentation mask
        mask = self.decoder(enriched, target_size=(H, W))   # B x 1 x H x W

        # Boundary refinement
        mask = self.boundary_refine(mask)

        # Image-level score
        global_feat = self.global_pool(features[-1])
        img_logit   = self.image_classifier(global_feat)
        img_prob    = torch.sigmoid(img_logit).squeeze(1)

        if return_features:
            return mask, img_prob, features

        return mask, img_prob


class LocalizationLoss(nn.Module):
    """
    Combined loss for manipulation localization:
    - Dice + BCE for segmentation mask
    - BCE for image-level classification
    - Edge/boundary awareness loss
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.alpha = alpha   # weight for dice
        self.beta  = beta    # weight for bce-seg
        self.gamma = gamma   # weight for img-level
        self.label_smoothing = label_smoothing

    def dice_loss(
        self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        pred_flat   = pred.flatten(1)
        target_flat = target.flatten(1).float()
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = 1 - (2 * intersection + eps) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps
        )
        return dice.mean()

    def boundary_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Penalize errors near manipulation boundaries."""
        # Compute edge map via gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        target_f = target.float()
        gx = F.conv2d(target_f, sobel_x, padding=1)
        gy = F.conv2d(target_f, sobel_y, padding=1)
        edge = (gx**2 + gy**2).sqrt().clamp(0, 1)
        # Weight BCE loss by edge proximity
        weighted_bce = F.binary_cross_entropy(pred, target_f, weight=(1 + 5 * edge))
        return weighted_bce

    def forward(
        self,
        pred_mask: torch.Tensor,
        img_prob: torch.Tensor,
        label: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        losses = {}

        # Image-level BCE
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        img_bce = F.binary_cross_entropy(img_prob, target)
        losses["img_bce"] = img_bce
        total = self.gamma * img_bce

        if gt_mask is not None:
            # Segmentation dice
            dice = self.dice_loss(pred_mask, gt_mask)
            # Segmentation BCE
            seg_bce = F.binary_cross_entropy(pred_mask, gt_mask.float())
            # Boundary loss
            bnd = self.boundary_loss(pred_mask, gt_mask)

            losses["dice"]    = dice
            losses["seg_bce"] = seg_bce
            losses["boundary"] = bnd

            total = (self.alpha * dice +
                     self.beta  * seg_bce +
                     self.gamma * img_bce +
                     0.1 * bnd)
        else:
            # No GT mask: encourage consistent predictions
            # If label=0 (real), mask should be all zeros
            # If label=1 (fake), mask can be anything (we don't know where)
            real_mask = (1 - label).float().view(-1, 1, 1, 1)
            consistency = F.mse_loss(
                pred_mask * real_mask,
                torch.zeros_like(pred_mask) * real_mask
            )
            losses["consistency"] = consistency
            total = img_bce + 0.1 * consistency

        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    model = ManipulationLocalizationNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    mask, img_prob = model(x)
    print(f"mask: {mask.shape}, img_prob: {img_prob.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
