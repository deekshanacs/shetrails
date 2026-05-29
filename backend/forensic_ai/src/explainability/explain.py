"""
Explainability Module for Forensic AI Engine.
Implements: GradCAM, GradCAM++, Integrated Gradients, Attention Rollout.
Generates visual heatmaps overlaid on original images.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path


# ─────────────────────────────────────────────
#  GradCAM
# ─────────────────────────────────────────────

class GradCAM:
    """
    GradCAM: Class Activation Mapping using gradients.
    Supports any CNN with a target convolutional layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(save_gradient)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns GradCAM heatmap: H x W numpy array (0–1).
        """
        self.model.eval()
        x.requires_grad_(True)
        output = self.model(x)
        # Extract scalar output
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() > 1:
            output = output.squeeze(-1)
        score = output[0]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Pool gradients
        grads = self.gradients[0]              # C x H x W
        acts  = self.activations[0]            # C x H x W
        weights = grads.mean(dim=(-2, -1))     # C

        # Weighted combination
        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ with improved weighting for multiple instances.
    """

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        x.requires_grad_(True)
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() > 1:
            output = output.squeeze(-1)
        score = output[0]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients[0]   # C x H x W
        acts  = self.activations[0] # C x H x W

        # GradCAM++ weights
        grads_2  = grads ** 2
        grads_3  = grads ** 3
        alpha_num = grads_2
        alpha_denom = 2 * grads_2 + (acts * grads_3).sum(dim=(-2, -1), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-8)

        relu_grad = F.relu(score.exp() * grads)  # positive class contribution
        weights   = (alpha * relu_grad).sum(dim=(-2, -1))  # C

        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ─────────────────────────────────────────────
#  Integrated Gradients
# ─────────────────────────────────────────────

class IntegratedGradients:
    """
    Integrated Gradients: attribute importance to each input pixel.
    Baseline: black image (zeros).
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model   = model
        self.n_steps = n_steps

    def generate(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Returns integrated gradients map: H x W (mean over channels).
        """
        if baseline is None:
            baseline = torch.zeros_like(x)

        x        = x.detach()
        baseline = baseline.detach()

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps, device=x.device)
        integrated_grads = torch.zeros_like(x)

        for alpha in alphas:
            inp = (baseline + alpha * (x - baseline)).requires_grad_(True)
            output = self.model(inp)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.dim() > 1:
                output = output.squeeze(-1)
            score = output[0]

            self.model.zero_grad()
            score.backward(retain_graph=True)

            if inp.grad is not None:
                integrated_grads += inp.grad.detach()

        # Multiply by (input - baseline) and average
        ig = (x - baseline) * integrated_grads / self.n_steps
        ig = ig.abs().mean(dim=1).squeeze(0).cpu().numpy()  # H x W
        ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
        return ig


# ─────────────────────────────────────────────
#  Attention Rollout
# ─────────────────────────────────────────────

class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.
    Aggregates attention maps across all layers.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean",
                 discard_ratio: float = 0.9):
        self.model         = model
        self.head_fusion   = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions    = []
        self._hooks        = []

    def _get_attn_hook(self):
        def hook(module, input, output):
            # Capture softmax attention weights
            if isinstance(output, tuple):
                attn_weights = output[1]  # some modules return (value, weights)
                if attn_weights is not None:
                    self.attentions.append(attn_weights.detach())
            elif output.dim() == 4:
                self.attentions.append(output.detach())
        return hook

    def register_hooks(self):
        for name, module in self.model.named_modules():
            # Detect attention modules by common patterns
            cls_name = type(module).__name__.lower()
            if "attention" in cls_name or "attn" in cls_name:
                h = module.register_forward_hook(self._get_attn_hook())
                self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.attentions.clear()

    def generate(self, x: torch.Tensor) -> Optional[np.ndarray]:
        self.attentions.clear()
        self.register_hooks()

        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)

        self.remove_hooks()

        if not self.attentions:
            return None

        result = torch.eye(
            self.attentions[0].shape[-1],
            device=self.attentions[0].device
        ).unsqueeze(0).expand(self.attentions[0].shape[0], -1, -1)

        for attn in self.attentions:
            if attn.dim() == 4:    # B x H x N x N
                if self.head_fusion == "mean":
                    attn_fused = attn.mean(dim=1)
                elif self.head_fusion == "max":
                    attn_fused = attn.max(dim=1).values
                else:
                    attn_fused = attn.min(dim=1).values
            elif attn.dim() == 3:  # B x N x N
                attn_fused = attn
            else:
                continue

            B, N, _ = attn_fused.shape

            # Apply discard ratio
            flat = attn_fused.view(B, N * N)
            thresh = flat.quantile(self.discard_ratio, dim=-1, keepdim=True)
            attn_fused[attn_fused < thresh.view(B, 1, 1)] = 0

            # Add residual
            attn_fused = attn_fused + torch.eye(N, device=attn_fused.device)
            attn_fused = attn_fused / (attn_fused.sum(dim=-1, keepdim=True) + 1e-8)

            if result.shape[-1] != N:
                result = torch.eye(N, device=attn_fused.device).unsqueeze(0).expand(B, -1, -1)

            result = torch.bmm(attn_fused, result)

        # CLS → patches
        mask = result[0, 0, 1:]  # N-1 patches
        N_patches = mask.shape[0]
        H = W = int(math.sqrt(N_patches))
        if H * W != N_patches:
            # Non-square: just return flattened
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask.reshape(H, W).cpu().numpy()

        # Normalize
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
        return mask_np


# ─────────────────────────────────────────────
#  Heatmap Visualization
# ─────────────────────────────────────────────

def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Apply colormap to a normalized heatmap."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    image: H x W x 3 uint8
    heatmap: H x W float [0, 1]
    """
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = apply_colormap(heatmap, colormap)
    overlay = (alpha * colored + (1 - alpha) * image).astype(np.uint8)
    return overlay


def generate_explanation_panel(
    image: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    save_path: str,
    title: str = "",
) -> np.ndarray:
    """
    Create a multi-panel visualization with all heatmaps.
    image: H x W x 3 uint8
    heatmaps: dict of method_name → H x W [0,1]
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_panels = 1 + len(heatmaps)
    fig = plt.figure(figsize=(4 * n_panels, 4))
    gs  = gridspec.GridSpec(1, n_panels)

    # Original
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.set_title("Original", fontsize=10)
    ax0.axis("off")

    # Each heatmap
    for i, (name, hmap) in enumerate(heatmaps.items()):
        ax = fig.add_subplot(gs[0, i + 1])
        overlay = overlay_heatmap(image, hmap)
        ax.imshow(overlay)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    return overlay


# ─────────────────────────────────────────────
#  Unified Explainability Engine
# ─────────────────────────────────────────────

class ForensicExplainer:
    """
    Generates all explanations for a forensic model.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        methods: List[str] = ("gradcam", "gradcam_pp", "integrated_gradients"),
        device: str = "cuda",
    ):
        self.model      = model.to(device)
        self.model_name = model_name
        self.methods    = methods
        self.device     = torch.device(device)

        # Get target layer for gradient methods
        self.target_layer = self._get_target_layer()

    def _get_target_layer(self) -> Optional[nn.Module]:
        """Auto-detect last convolutional layer."""
        if hasattr(self.model, "get_cam_target_layer"):
            return self.model.get_cam_target_layer()
        # Auto-detect
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                return module
        return None

    def explain(
        self,
        image_tensor: torch.Tensor,
        image_np: np.ndarray,
        save_dir: str,
        filename: str = "explanation",
    ) -> Dict[str, np.ndarray]:
        """
        Generate all heatmaps for a single image.
        Args:
            image_tensor: 1 x 3 x H x W on CPU
            image_np:     H x W x 3 uint8 (for visualization)
        Returns:
            dict of method → heatmap array
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        x = image_tensor.to(self.device)
        heatmaps = {}

        if "gradcam" in self.methods and self.target_layer is not None:
            gcam = GradCAM(self.model, self.target_layer)
            hmap = gcam.generate(x)
            gcam.remove_hooks()
            heatmaps["GradCAM"] = hmap

        if "gradcam_pp" in self.methods and self.target_layer is not None:
            gcam_pp = GradCAMPlusPlus(self.model, self.target_layer)
            hmap = gcam_pp.generate(x)
            gcam_pp.remove_hooks()
            heatmaps["GradCAM++"] = hmap

        if "integrated_gradients" in self.methods:
            ig = IntegratedGradients(self.model, n_steps=30)
            hmap = ig.generate(x)
            heatmaps["IntGrad"] = hmap

        if "attention_rollout" in self.methods:
            ar = AttentionRollout(self.model, head_fusion="mean", discard_ratio=0.9)
            hmap = ar.generate(x)
            if hmap is not None:
                if hmap.ndim == 1:
                    # Reshape best we can
                    n = len(hmap)
                    h = w = int(math.sqrt(n))
                    if h * w == n:
                        hmap = hmap.reshape(h, w)
                    else:
                        hmap = hmap.reshape(1, -1).repeat(n, axis=0)[:n, :n]
                heatmaps["AttRollout"] = cv2.resize(
                    hmap, (image_np.shape[1], image_np.shape[0])
                )

        # Generate panel
        prob_str = ""
        panel_path = str(Path(save_dir) / f"{filename}_explanation.png")
        generate_explanation_panel(image_np, heatmaps, panel_path, title=prob_str)

        # Save individual maps
        for method, hmap in heatmaps.items():
            hmap_path = str(Path(save_dir) / f"{filename}_{method.lower()}.npy")
            np.save(hmap_path, hmap)

        return heatmaps


if __name__ == "__main__":
    from src.models.model1_rgb import RGBForensicsNet

    model = RGBForensicsNet(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    img_np = (x[0].permute(1, 2, 0).numpy() * 127 + 128).clip(0, 255).astype(np.uint8)

    explainer = ForensicExplainer(
        model, "rgb",
        methods=["gradcam", "gradcam_pp", "integrated_gradients"],
        device="cpu"
    )
    heatmaps = explainer.explain(x, img_np, save_dir="outputs/heatmaps", filename="test")
    print(f"Generated {len(heatmaps)} heatmaps: {list(heatmaps.keys())}")
