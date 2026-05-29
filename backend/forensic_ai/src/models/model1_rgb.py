"""
Model 1: RGB Forensics Network
Backbone: ConvNeXt V2 Large (falls back to EfficientNetV2-L)
Input:  3xHxW RGB image
Output: manipulation probability scalar + feature embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional


class AttentionPool(nn.Module):
    """Attention-based pooling over spatial features."""
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x N x C
        w = self.attn(x)             # B x N x 1
        w = F.softmax(w, dim=1)
        return (x * w).sum(dim=1)    # B x C


class RGBForensicsNet(nn.Module):
    """
    ConvNeXt V2 Large backbone with forensic-specific head.
    If convnextv2_large is unavailable, falls back to efficientnetv2_l.
    """

    def __init__(
        self,
        backbone: str = "convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        embed_dim: int = 512,
    ):
        super().__init__()
        # ── Backbone ──
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0, global_pool=""
            )
            feat_dim = self.backbone.num_features
        except Exception:
            print(f"[RGBForensicsNet] Falling back to efficientnetv2_l")
            self.backbone = timm.create_model(
                "tf_efficientnetv2_l.in21k_ft_in1k", pretrained=pretrained,
                num_classes=0, global_pool=""
            )
            feat_dim = self.backbone.num_features

        # ── Feature head ──
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── Classification head ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

        # ── Manipulation type head (multi-label auxiliary) ──
        # Classes: none, deepfake, face_swap, ai_gen, splice, copy_move, removal, photoshop
        self.manip_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 8),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.feature_proj, self.classifier, self.manip_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        # Feature extraction
        feat_map = self.backbone(x)          # B x C x H x W

        # Dual pooling
        avg = self.gap(feat_map).flatten(1)  # B x C
        mx  = self.gmp(feat_map).flatten(1)  # B x C
        pooled = torch.cat([avg, mx], dim=1) # B x 2C

        # Project to embedding
        embed = self.feature_proj(pooled)    # B x embed_dim

        # Heads
        logit = self.classifier(embed)       # B x 1
        manip = self.manip_head(embed)       # B x 8

        prob = torch.sigmoid(logit).squeeze(1)  # B

        if return_features:
            return prob, embed, manip, feat_map

        return prob, embed, manip

    def get_cam_target_layer(self):
        """Return the last conv layer for GradCAM."""
        for name, module in reversed(list(self.backbone.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                return module
        return None


class RGBForensicsLoss(nn.Module):
    """
    Combined loss:
    - Binary cross-entropy for forgery detection
    - Multi-label cross-entropy for manipulation type (auxiliary)
    """

    def __init__(self, alpha: float = 0.7, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(
        self,
        prob: torch.Tensor,
        manip_logit: torch.Tensor,
        label: torch.Tensor,
        manip_label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Primary: binary detection
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(prob, target)

        losses = {"bce": bce}
        total = bce

        # Auxiliary: manipulation type
        if manip_label is not None:
            aux = F.cross_entropy(manip_logit, manip_label)
            losses["manip_aux"] = aux
            total = self.alpha * bce + (1 - self.alpha) * aux

        losses["total"] = total
        return total, losses


# ─────────────────────────────────────────────
#  EMA wrapper
# ─────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model weights for stable inference."""

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


if __name__ == "__main__":
    model = RGBForensicsNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    prob, embed, manip = model(x)
    print(f"prob: {prob.shape}, embed: {embed.shape}, manip: {manip.shape}")
    # prob: torch.Size([2])
    # embed: torch.Size([2, 512])
    # manip: torch.Size([2, 8])
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
