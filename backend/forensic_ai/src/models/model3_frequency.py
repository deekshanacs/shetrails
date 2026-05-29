"""
Model 3: Frequency Domain Forensics Network
Input:  9xHxW (FFT + DCT + Wavelet concatenated)
Output: frequency-domain forgery probability + embedding
Architecture: EfficientNet-B4 backbone with frequency-aware attention
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple


class FrequencyChannelAttention(nn.Module):
    """
    Channel attention specialized for frequency domain inputs.
    Learns to weight FFT, DCT, and Wavelet channels differently.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.fc(x).view(x.shape[0], -1, 1, 1)
        return x * attn


class FrequencyInputStem(nn.Module):
    """
    Maps 9-channel frequency input to 3-channel for standard backbones.
    Learned fusion of FFT, DCT, Wavelet representations.
    """

    def __init__(self, in_channels: int = 9):
        super().__init__()
        self.conv_fft = nn.Conv2d(3, 16, 1, bias=False)
        self.conv_dct = nn.Conv2d(3, 16, 1, bias=False)
        self.conv_wav = nn.Conv2d(3, 16, 1, bias=False)
        self.channel_attn = FrequencyChannelAttention(48)
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 3, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )
        self.freq_attn = FrequencyChannelAttention(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x 9 x H x W (FFT:3, DCT:3, WAV:3)
        fft = self.conv_fft(x[:, 0:3])
        dct = self.conv_dct(x[:, 3:6])
        wav = self.conv_wav(x[:, 6:9])
        cat = torch.cat([fft, dct, wav], dim=1)   # B x 48 x H x W
        cat = self.channel_attn(cat)
        out = self.fusion(cat)                     # B x 3 x H x W
        out = self.freq_attn(out)
        return out


class SpectralPeakDetector(nn.Module):
    """
    Detects periodic artifacts in FFT spectrum (GAN fingerprints, grid patterns).
    Operates on the radial frequency profile.
    """

    def __init__(self, n_bins: int = 64, embed_dim: int = 128):
        super().__init__()
        self.n_bins = n_bins
        self.mlp = nn.Sequential(
            nn.Linear(n_bins, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _radial_profile(self, fft_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Compute radial average of 2D power spectrum.
        fft_magnitude: B x H x W (single channel)
        Returns: B x n_bins
        """
        B, H, W = fft_magnitude.shape
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=fft_magnitude.device).float() - cy
        x = torch.arange(W, device=fft_magnitude.device).float() - cx
        r = torch.sqrt(y.view(-1, 1)**2 + x.view(1, -1)**2)  # H x W
        r_int = r.long().clamp(0, min(cy, cx) - 1)
        profile = torch.zeros(B, self.n_bins, device=fft_magnitude.device)
        for b in range(B):
            mag_flat = fft_magnitude[b].flatten()
            r_flat   = r_int.flatten()
            for i in range(self.n_bins):
                mask = (r_flat == i)
                if mask.sum() > 0:
                    profile[b, i] = mag_flat[mask].mean()
        return profile

    def forward(self, freq_input: torch.Tensor) -> torch.Tensor:
        # Use FFT channel (first 3 channels, take mean)
        fft_ch = freq_input[:, 0:3].mean(dim=1)   # B x H x W
        profile = self._radial_profile(fft_ch)
        return self.mlp(profile)                   # B x embed_dim


class FrequencyForensicsNet(nn.Module):
    """
    Full frequency-domain forgery detector.
    Combines backbone features with spectral peak analysis.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_m.in21k_ft_in1k",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        embed_dim: int = 512,
        use_spectral_peak: bool = True,
    ):
        super().__init__()
        self.use_spectral_peak = use_spectral_peak

        # ── Input stem: 9ch → 3ch ──
        self.input_stem = FrequencyInputStem(in_channels=9)

        # ── Backbone ──
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained,
                num_classes=0, global_pool=""
            )
        except Exception:
            self.backbone = timm.create_model(
                "efficientnet_b4", pretrained=pretrained,
                num_classes=0, global_pool=""
            )

        feat_dim = self.backbone.num_features

        # ── Spectral peak detector ──
        spectral_dim = 128 if use_spectral_peak else 0
        if use_spectral_peak:
            self.spectral = SpectralPeakDetector(n_bins=64, embed_dim=spectral_dim)

        # ── Pooling ──
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # ── Feature head ──
        combined_dim = feat_dim * 2 + spectral_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(combined_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        # x: B x 9 x H x W (frequency maps)

        # Stem: 9ch → 3ch
        x_stem = self.input_stem(x)              # B x 3 x H x W

        # Backbone
        feat_map = self.backbone(x_stem)          # B x C x H' x W'
        avg = self.gap(feat_map).flatten(1)
        mx  = self.gmp(feat_map).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)     # B x 2C

        # Spectral peak features
        if self.use_spectral_peak:
            spectral_feat = self.spectral(x)      # B x 128
            combined = torch.cat([pooled, spectral_feat], dim=1)
        else:
            combined = pooled

        embed = self.feature_proj(combined)       # B x embed_dim
        logit = self.classifier(embed)            # B x 1
        prob  = torch.sigmoid(logit).squeeze(1)   # B

        if return_features:
            return prob, embed, feat_map

        return prob, embed


class DCTStatisticsExtractor(nn.Module):
    """
    Extract higher-order statistics from DCT coefficients.
    GAN/Diffusion artifacts often manifest as non-natural DCT coefficient distributions.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        # 4 statistics (mean, std, skew proxy, kurt proxy) x 3 frequency bands = 12
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, dct_input: torch.Tensor) -> torch.Tensor:
        # dct_input: B x 3 x H x W
        feats = []
        for band in range(3):
            ch = dct_input[:, band]  # B x H x W
            ch_flat = ch.flatten(1)
            mean = ch_flat.mean(dim=1)
            std  = ch_flat.std(dim=1)
            # proxy skewness
            skew = ((ch_flat - mean.unsqueeze(1))**3).mean(dim=1) / (std**3 + 1e-8)
            # proxy kurtosis
            kurt = ((ch_flat - mean.unsqueeze(1))**4).mean(dim=1) / (std**4 + 1e-8)
            feats += [mean, std, skew, kurt]
        stats = torch.stack(feats, dim=1)   # B x 12
        return self.mlp(stats)


class FrequencyForensicsLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        prob: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy(prob, target)
        return loss, {"total": loss, "bce": loss}


if __name__ == "__main__":
    model = FrequencyForensicsNet(pretrained=False)
    x = torch.randn(2, 9, 224, 224)
    prob, embed = model(x)
    print(f"prob: {prob.shape}, embed: {embed.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
