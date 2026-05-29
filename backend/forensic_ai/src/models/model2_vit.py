"""
Model 2: Vision Transformer Forensics Network
Backbone: ViT-Large / Swin-Large / EVA ViT
Input:  3xHxW RGB image
Output: forgery probability + patch-level attention map + embedding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional, List


class PatchAttentionRollout:
    """
    Computes Attention Rollout across all transformer layers.
    Produces a spatial attention map at the patch level.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean",
                 discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions: List[torch.Tensor] = []
        self._hooks = []

    def _hook_fn(self, module, input, output):
        # Works for timm ViT blocks that expose attn_drop
        if hasattr(module, "attn") and hasattr(module.attn, "attn_drop"):
            pass  # timm already stores attention
        self.attentions.append(output.detach())

    def register_hooks(self):
        """Register hooks on all Attention modules."""
        for name, module in self.model.named_modules():
            if "attn_drop" in name or type(module).__name__ in ["Attention", "WindowAttention"]:
                h = module.register_forward_hook(self._hook_fn)
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.attentions.clear()

    def compute_rollout(self, attention_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Rollout attention matrices across layers.
        Returns attention from CLS token to all patches: B x N
        """
        B = attention_list[0].shape[0]
        result = torch.eye(attention_list[0].shape[-1], device=attention_list[0].device)
        result = result.unsqueeze(0).expand(B, -1, -1)

        for attn in attention_list:
            # attn: B x H x N x N
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)      # B x N x N
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                attn_fused = attn.min(dim=1).values

            flat = attn_fused.reshape(B, -1)
            threshold = flat.quantile(self.discard_ratio, dim=-1, keepdim=True)
            attn_fused[attn_fused < threshold.unsqueeze(-1)] = 0

            # Add residual connection
            attn_fused = attn_fused + torch.eye(attn_fused.shape[-1], device=attn_fused.device)
            attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-8)

            result = torch.bmm(attn_fused, result)

        # CLS token attention to patches (exclude CLS-to-CLS)
        cls_attn = result[:, 0, 1:]   # B x (N-1)
        return cls_attn


class ForensicViT(nn.Module):
    """
    Vision Transformer with forensic-tuned head.
    Supports: ViT-L/16, Swin-L, EVA-L.
    """

    BACKBONE_MAP = {
        "vit_large":  "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "swin_large": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        "eva_large":  "eva_large_patch14_196.mim_m38m_ft_in22k_in1k",
        "vit_base":   "vit_base_patch16_224.augreg_in21k_ft_in1k",
    }

    def __init__(
        self,
        backbone: str = "vit_large",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.1,
        embed_dim: int = 512,
        use_local_features: bool = True,
    ):
        super().__init__()
        backbone_name = self.BACKBONE_MAP.get(backbone, backbone)

        try:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained,
                num_classes=0, global_pool=""
            )
            feat_dim = self.backbone.num_features
        except Exception as e:
            print(f"[ForensicViT] Could not load {backbone_name}: {e}. Using vit_base.")
            self.backbone = timm.create_model(
                "vit_base_patch16_224.augreg_in21k_ft_in1k",
                pretrained=pretrained, num_classes=0, global_pool=""
            )
            feat_dim = self.backbone.num_features

        self.is_swin = "swin" in backbone.lower()
        self.use_local_features = use_local_features

        # ── Local patch cross-attention ──
        if use_local_features:
            self.local_attn = nn.MultiheadAttention(
                embed_dim=feat_dim, num_heads=8, dropout=dropout, batch_first=True
            )
            self.local_norm = nn.LayerNorm(feat_dim)

        # ── Projection ──
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # ── Source classifier (real/GAN/Diffusion/Deepfake) ──
        self.source_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),
        )

    def _extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_token: B x feat_dim
            patch_tokens: B x N x feat_dim
        """
        feats = self.backbone(x)  # B x N x C for ViT, B x H x W x C for Swin

        if self.is_swin:
            # Swin returns spatial feature map
            B = feats.shape[0]
            patch_tokens = feats.flatten(1, -2)  # B x N x C
            cls_token = patch_tokens.mean(dim=1)
        else:
            # ViT returns [CLS, patch, patch, ...]
            if feats.dim() == 3:
                cls_token    = feats[:, 0]       # B x C
                patch_tokens = feats[:, 1:]      # B x (N-1) x C
            else:
                patch_tokens = feats
                cls_token = feats.mean(dim=1)

        return cls_token, patch_tokens

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        cls_token, patch_tokens = self._extract_tokens(x)

        # Local cross-attention refinement
        if self.use_local_features:
            q = cls_token.unsqueeze(1)                       # B x 1 x C
            attn_out, _ = self.local_attn(q, patch_tokens, patch_tokens)
            cls_token = self.local_norm(cls_token + attn_out.squeeze(1))

        embed = self.proj(cls_token)                         # B x embed_dim
        logit = self.classifier(embed)                       # B x 1
        source_logit = self.source_head(embed)               # B x 4

        prob = torch.sigmoid(logit).squeeze(1)               # B

        if return_features:
            return prob, embed, source_logit, patch_tokens

        return prob, embed, source_logit

    def get_attention_map(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns spatial attention map at input resolution.
        B x H x W (normalized)
        """
        if self.is_swin:
            return None  # Swin attention rollout not implemented here

        self.eval()
        with torch.no_grad():
            feats = self.backbone(x)
            if feats.dim() == 3:
                patch_tokens = feats[:, 1:]   # B x N x C
                # Compute self-similarity of patches
                norm = F.normalize(patch_tokens, dim=-1)
                sim = torch.bmm(norm, norm.transpose(1, 2))  # B x N x N
                attn = sim.mean(dim=1)                        # B x N
                attn = (attn - attn.min(dim=-1, keepdim=True).values) / \
                       (attn.max(dim=-1, keepdim=True).values - attn.min(dim=-1, keepdim=True).values + 1e-8)
                # Reshape to spatial
                N = patch_tokens.shape[1]
                H = W = int(math.sqrt(N))
                if H * W == N:
                    return attn.reshape(x.shape[0], H, W)
        return None


class ViTForensicsLoss(nn.Module):
    """
    Loss for ViT model:
    - Detection BCE
    - Source classification (real/GAN/Diffusion/Deepfake) CE
    - Patch diversity regularization
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.2,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        prob: torch.Tensor,
        source_logit: torch.Tensor,
        label: torch.Tensor,
        patch_tokens: Optional[torch.Tensor] = None,
        source_label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Primary
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(prob, target)

        losses = {"bce": bce}
        total = bce

        # Source classification
        if source_label is not None:
            src_loss = F.cross_entropy(source_logit, source_label)
            losses["source"] = src_loss
            total = self.alpha * bce + (1 - self.alpha - self.beta) * src_loss

        # Patch diversity: encourage patches to differ (regularization)
        if patch_tokens is not None:
            norm_patches = F.normalize(patch_tokens, dim=-1)  # B x N x C
            sim_matrix = torch.bmm(norm_patches, norm_patches.transpose(1, 2))  # B x N x N
            eye = torch.eye(sim_matrix.shape[1], device=sim_matrix.device)
            off_diag_sim = (sim_matrix * (1 - eye)).mean()
            diversity_loss = off_diag_sim  # minimize similarity = maximize diversity
            losses["diversity"] = diversity_loss
            total = total + self.beta * diversity_loss

        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    model = ForensicViT(backbone="vit_base", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    prob, embed, source = model(x)
    print(f"prob: {prob.shape}, embed: {embed.shape}, source: {source.shape}")
    attn = model.get_attention_map(x)
    if attn is not None:
        print(f"attn map: {attn.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
