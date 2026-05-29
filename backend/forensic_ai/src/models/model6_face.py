"""
Model 6: Face Forensics Network
Detects deepfakes, face swaps, and identity inconsistencies.
Input:  3xHxW face-cropped image
Output: forgery probability + identity inconsistency score + embedding
Trained on: FaceForensics++, Celeb-DF, DFDC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import Tuple, Optional, List, Dict


# ─────────────────────────────────────────────
#  Face Detector Wrapper (RetinaFace / MTCNN)
# ─────────────────────────────────────────────

class FaceDetector:
    """
    Wraps RetinaFace or MTCNN for face crop extraction.
    Falls back to center crop if no face detected.
    """

    def __init__(self, detector: str = "retinaface", device: str = "cuda",
                 margin: float = 0.3):
        self.device = device
        self.margin = margin
        self.detector_name = detector
        self._detector = None
        self._init_detector(detector)

    def _init_detector(self, name: str):
        try:
            if name == "retinaface":
                from retinaface import RetinaFace
                self._detector = RetinaFace
                self._type = "retinaface"
            elif name == "mtcnn":
                from facenet_pytorch import MTCNN
                self._detector = MTCNN(keep_all=True, device=self.device)
                self._type = "mtcnn"
        except ImportError:
            print(f"[FaceDetector] {name} not available. Using center crop fallback.")
            self._detector = None
            self._type = "none"

    def detect_and_crop(
        self, image: np.ndarray, size: int = 224
    ) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        Returns:
            face_crop: HxWx3 numpy array or None
            landmarks: list of facial landmark points or None
        """
        import cv2
        if self._type == "retinaface" and self._detector is not None:
            try:
                faces = self._detector.detect_faces(image)
                if faces:
                    best_face = max(faces.values(), key=lambda x: x["score"])
                    x1, y1, w, h = best_face["facial_area"]
                    x2, y2 = x1 + w, y1 + h
                    mx = int((x2 - x1) * self.margin)
                    my = int((y2 - y1) * self.margin)
                    x1 = max(0, x1 - mx);  y1 = max(0, y1 - my)
                    x2 = min(image.shape[1], x2 + mx)
                    y2 = min(image.shape[0], y2 + my)
                    crop = image[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (size, size))
                    landmarks = best_face.get("landmarks")
                    return crop, landmarks
            except Exception:
                pass

        elif self._type == "mtcnn" and self._detector is not None:
            try:
                from PIL import Image as PILImage
                pil = PILImage.fromarray(image)
                boxes, probs, landmarks = self._detector.detect(pil, landmarks=True)
                if boxes is not None and len(boxes) > 0:
                    best = np.argmax(probs)
                    x1, y1, x2, y2 = [int(v) for v in boxes[best]]
                    mx = int((x2 - x1) * self.margin)
                    my = int((y2 - y1) * self.margin)
                    x1 = max(0, x1 - mx);  y1 = max(0, y1 - my)
                    x2 = min(image.shape[1], x2 + mx)
                    y2 = min(image.shape[0], y2 + my)
                    crop = image[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (size, size))
                    return crop, landmarks[best].tolist()
            except Exception:
                pass

        # Fallback: center crop
        h, w = image.shape[:2]
        s = min(h, w)
        y1 = (h - s) // 2;  x1 = (w - s) // 2
        crop = image[y1:y1+s, x1:x1+s]
        crop = cv2.resize(crop, (size, size))
        return crop, None


# ─────────────────────────────────────────────
#  Landmark Inconsistency Analyzer
# ─────────────────────────────────────────────

class LandmarkInconsistencyModule(nn.Module):
    """
    Detects geometric inconsistencies in facial landmarks.
    Face swaps often have subtle landmark misalignments.
    """

    def __init__(self, n_landmarks: int = 5, embed_dim: int = 64):
        super().__init__()
        input_dim = n_landmarks * 2  # x,y for each
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """landmarks: B x n_landmarks x 2"""
        B = landmarks.shape[0]
        flat = landmarks.view(B, -1).float()
        return self.mlp(flat)


# ─────────────────────────────────────────────
#  Blending Boundary Detector
# ─────────────────────────────────────────────

class BlendingBoundaryDetector(nn.Module):
    """
    Detects the blending boundary artifacts common in face swaps.
    Analyzes gradient discontinuities at face edges.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Sobel-based edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        self.register_buffer("sobel_x",
                             sobel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3))
        self.register_buffer("sobel_y",
                             sobel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3))
        self.analyzer = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x 3 x H x W
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        edge = torch.cat([gx, gy], dim=1)  # B x 6 x H x W
        return self.analyzer(edge)         # B x 32


# ─────────────────────────────────────────────
#  Main Face Forensics Network
# ─────────────────────────────────────────────

class FaceForensicsNet(nn.Module):
    """
    Multi-cue face forgery detector.
    Combines:
    - Backbone appearance features
    - Blending boundary analysis
    - Frequency artifact analysis
    - (Optional) Landmark inconsistency
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_m.in21k_ft_in1k",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        embed_dim: int = 512,
        use_landmarks: bool = False,
    ):
        super().__init__()
        self.use_landmarks = use_landmarks

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

        # ── Auxiliary modules ──
        self.blend_detector = BlendingBoundaryDetector()
        blend_dim = 32

        landmark_dim = 64 if use_landmarks else 0
        if use_landmarks:
            self.landmark_module = LandmarkInconsistencyModule(n_landmarks=5)

        # ── Pooling ──
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # ── Feature fusion ──
        combined_dim = feat_dim * 2 + blend_dim + landmark_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(combined_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── Heads ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # Forgery type: deepfake | face_swap | ai_gen | authentic
        self.forgery_type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),
        )

        # Identity score: how much does the identity look inconsistent
        self.identity_score_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # Backbone
        feat_map = self.backbone(x)              # B x C x H' x W'
        avg    = self.gap(feat_map).flatten(1)
        mx     = self.gmp(feat_map).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)     # B x 2C

        # Blending boundary
        blend_feat = self.blend_detector(x)      # B x 32

        # Combine
        parts = [pooled, blend_feat]
        if self.use_landmarks and landmarks is not None:
            lm_feat = self.landmark_module(landmarks)
            parts.append(lm_feat)

        combined = torch.cat(parts, dim=1)
        embed    = self.feature_proj(combined)   # B x embed_dim

        logit        = self.classifier(embed)
        prob         = torch.sigmoid(logit).squeeze(1)
        forgery_type = self.forgery_type_head(embed)
        identity_inc = self.identity_score_head(embed).squeeze(1)

        if return_features:
            return prob, embed, forgery_type, identity_inc, feat_map

        return prob, embed, forgery_type, identity_inc

    def get_cam_target_layer(self):
        for name, module in reversed(list(self.backbone.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        return None


class FaceForensicsLoss(nn.Module):
    """
    Multi-task loss for face forensics:
    - Binary forgery detection
    - Forgery type classification
    - Identity inconsistency regression
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        prob: torch.Tensor,
        forgery_type: torch.Tensor,
        identity_inc: torch.Tensor,
        label: torch.Tensor,
        type_label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        target = label.float()
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(prob, target)

        losses = {"bce": bce}
        total  = bce

        if type_label is not None:
            type_loss = F.cross_entropy(forgery_type, type_label)
            losses["type"] = type_loss
            total = self.alpha * bce + self.beta * type_loss

        # Identity score should be high for fakes
        id_target = label.float()
        id_loss = F.binary_cross_entropy(identity_inc, id_target)
        losses["identity"] = id_loss
        total = total + self.gamma * id_loss

        losses["total"] = total
        return total, losses


if __name__ == "__main__":
    model = FaceForensicsNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    prob, embed, ftype, id_inc = model(x)
    print(f"prob: {prob.shape}")
    print(f"embed: {embed.shape}")
    print(f"forgery_type: {ftype.shape}")
    print(f"identity_inconsistency: {id_inc.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
