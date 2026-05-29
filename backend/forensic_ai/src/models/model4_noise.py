"""
Model 4: Noise Forensics Network
Uses SRM (Steganalysis Rich Model) filters for noise residual extraction.
Input:  3xHxW RGB image (SRM preprocessing applied internally or externally)
Output: noise-based forgery probability + embedding
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, List


# ─────────────────────────────────────────────
#  SRM Filter Bank (30 filters)
# ─────────────────────────────────────────────

def build_srm_filters() -> torch.Tensor:
    """
    Builds the full SRM (Steganalysis Rich Model) filter bank.
    Returns 30 x 1 x 5 x 5 filter tensor.
    These are high-pass filters that extract pixel-level noise residuals.
    """
    filters = []

    # ── First-order filters ──
    f1 = np.array([[0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0, -1,  1,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    f2 = np.array([[0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  1, -1, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    f3 = np.array([[0,  0, -1,  0, 0],
                   [0,  0,  1,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    f4 = np.array([[0,  0,  0,  0, 0],
                   [0,  0,  1,  0, 0],
                   [0,  0, -1,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    # ── Second-order filters ──
    f5 = np.array([[0,  0,  0,  0, 0],
                   [0, -1,  2, -1, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    f6 = np.array([[0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0, -1,  2, -1, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    f7 = np.array([[0,  0, -1,  0, 0],
                   [0,  0,  2,  0, 0],
                   [0,  0, -1,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    f8 = np.array([[0,  0,  0,  0, 0],
                   [0,  0, -1,  0, 0],
                   [0,  0,  2,  0, 0],
                   [0,  0, -1,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    # ── Third-order filters ──
    f9 = np.array([[0,  0,  0,  0, 0],
                   [0, -1,  3, -3, 1],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0],
                   [0,  0,  0,  0, 0]], dtype=np.float32)

    f10 = np.array([[0,  0,  0,  0, 0],
                    [0,  0,  0,  0, 0],
                    [0, -1,  3, -3, 1],
                    [0,  0,  0,  0, 0],
                    [0,  0,  0,  0, 0]], dtype=np.float32)

    # ── 2D second-order ──
    f11 = np.array([[0,  0,  0,  0, 0],
                    [0, -1,  2, -1, 0],
                    [0,  2, -4,  2, 0],
                    [0, -1,  2, -1, 0],
                    [0,  0,  0,  0, 0]], dtype=np.float32) / 4

    f12 = np.array([[-1, 2, -2,  2, -1],
                    [ 2,-6,  8, -6,  2],
                    [-2, 8,-12,  8, -2],
                    [ 2,-6,  8, -6,  2],
                    [-1, 2, -2,  2, -1]], dtype=np.float32) / 12

    # ── Edge filters ──
    sobel_x = np.array([[0, 0, 0, 0, 0],
                         [0,-1, 0, 1, 0],
                         [0,-2, 0, 2, 0],
                         [0,-1, 0, 1, 0],
                         [0, 0, 0, 0, 0]], dtype=np.float32)

    sobel_y = sobel_x.T

    laplacian = np.array([[0,  0, -1,  0, 0],
                           [0, -1, -2, -1, 0],
                           [-1,-2, 16, -2,-1],
                           [0, -1, -2, -1, 0],
                           [0,  0, -1,  0, 0]], dtype=np.float32) / 8

    # Collect base filters
    base = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
            sobel_x, sobel_y, laplacian]

    # Mirror/rotate to get 30 total
    for f in base:
        filters.append(f)
        if len(filters) >= 30:
            break

    # Generate remaining via rotations
    for deg in [90, 180, 270]:
        for f in base:
            filters.append(np.rot90(f, k=deg // 90).copy())
            if len(filters) >= 30:
                break
        if len(filters) >= 30:
            break

    filters = filters[:30]

    # Stack: 30 x 1 x 5 x 5
    kernel = np.stack(filters, axis=0)[:, np.newaxis, :, :]
    return torch.from_numpy(kernel).float()


class SRMFilterLayer(nn.Module):
    """
    Applies SRM filter bank to RGB image.
    Non-learnable (fixed filters) + optional learnable residual combination.
    """

    def __init__(self, learnable: bool = False):
        super().__init__()
        srm_kernel = build_srm_filters()          # 30 x 1 x 5 x 5
        # Expand to 3 input channels: 30 x 3 x 5 x 5 (replicate per channel)
        srm_kernel_rgb = srm_kernel.repeat(1, 3, 1, 1) / 3.0
        self.register_buffer("srm_weight", srm_kernel_rgb)

        if learnable:
            self.combination = nn.Conv2d(30, 3, 1, bias=False)
        else:
            self.learnable = False
            # Fixed combination: take first 3 or mean
            self.combination = None
        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x 3 x H x W  (values expected in ~[-1, 1] or [0, 1])
        noise = F.conv2d(x, self.srm_weight, padding=2)  # B x 30 x H x W

        if self.learnable and self.combination is not None:
            return self.combination(noise)  # B x 3 x H x W

        # Non-learnable: return first 3 channels (most discriminative empirically)
        return noise[:, :3]   # B x 3 x H x W


class NoiseConsistencyAttention(nn.Module):
    """
    Detects inconsistent noise patterns between image regions.
    Real images have consistent camera noise; tampered regions differ.
    """

    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_heads, in_channels), in_channels)
        self.query = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (in_channels // 4) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        q = self.query(x_norm).view(B, -1, H * W).permute(0, 2, 1)   # B x HW x C//4
        k = self.key(x_norm).view(B, -1, H * W)                       # B x C//4 x HW
        v = self.value(x_norm).view(B, -1, H * W).permute(0, 2, 1)    # B x HW x C

        attn = torch.bmm(q, k) * self.scale                           # B x HW x HW
        attn = F.softmax(attn, dim=-1)
        out  = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)

        return x + self.gamma * out


class NoiseForensicsNet(nn.Module):
    """
    Noise residual forensics network.
    1. Fixed SRM filters extract noise map
    2. ResNet/EfficientNet backbone analyzes noise patterns
    3. Noise consistency attention detects local inconsistencies
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        embed_dim: int = 512,
        srm_learnable: bool = False,
    ):
        super().__init__()

        # ── SRM preprocessing ──
        self.srm = SRMFilterLayer(learnable=srm_learnable)

        # ── Backbone ──
        # We patch the first conv to accept 3-channel SRM output
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained,
                num_classes=0, global_pool=""
            )
        except Exception:
            self.backbone = timm.create_model(
                "resnet50", pretrained=pretrained,
                num_classes=0, global_pool=""
            )

        feat_dim = self.backbone.num_features

        # ── Noise consistency attention (applied to backbone features) ──
        self.noise_attn = NoiseConsistencyAttention(feat_dim)

        # ── Pooling ──
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # ── Feature projection ──
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

        # ── Noise level estimator (auxiliary task) ──
        # Predicts noise standard deviation (helps with compression robustness)
        self.noise_level_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # positive output
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: B x 3 x H x W (raw RGB, SRM applied internally)
        """
        # SRM noise residual
        noise_map = self.srm(x)              # B x 3 x H x W

        # Backbone
        feat_map = self.backbone(noise_map)  # B x C x H' x W'

        # Noise consistency attention
        feat_map = self.noise_attn(feat_map)

        # Pool
        avg    = self.gap(feat_map).flatten(1)
        mx     = self.gmp(feat_map).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)

        embed  = self.feature_proj(pooled)
        logit  = self.classifier(embed)
        prob   = torch.sigmoid(logit).squeeze(1)
        sigma  = self.noise_level_head(embed).squeeze(1)  # predicted noise level

        if return_features:
            return prob, embed, sigma, feat_map

        return prob, embed, sigma

    def get_noise_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw SRM noise residual for visualization."""
        return self.srm(x)


class NoiseForensicsLoss(nn.Module):
    """
    Loss with auxiliary noise level estimation.
    """

    def __init__(self, lambda_noise: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        self.lambda_noise = lambda_noise
        self.label_smoothing = label_smoothing

    def forward(
        self,
        prob: torch.Tensor,
        sigma: torch.Tensor,
        label: torch.Tensor,
        sigma_gt: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(prob, target)
        losses = {"bce": bce}
        total = bce

        if sigma_gt is not None:
            noise_loss = F.mse_loss(sigma, sigma_gt)
            losses["noise_reg"] = noise_loss
            total = bce + self.lambda_noise * noise_loss

        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    model = NoiseForensicsNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    prob, embed, sigma = model(x)
    print(f"prob: {prob.shape}, embed: {embed.shape}, sigma: {sigma.shape}")
    noise_map = model.get_noise_map(x)
    print(f"noise_map: {noise_map.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
