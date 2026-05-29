"""
Forensic AI Engine — Final Inference Pipeline
Runs all 7 forensic models + fusion, produces structured forensic report.
"""

import os
import time
import json
import logging
import hashlib
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
import piexif

from ..models.model1_rgb import RGBForensicsNet
from ..models.model2_vit import ViTForensicsNet
from ..models.model3_frequency import FrequencyForensicsNet
from ..models.model4_noise import NoiseForensicsNet
from ..models.model5_ela import ELAForensicsNet
from ..models.model6_face import FaceForensicsNet
from ..models.model7_localization import ManipulationLocalizationNet
from ..fusion.fusion import FusionEnsembleComparator, NeuralMetaClassifier
from ..explainability.explain import ForensicExplainer
from ..datasets.dataset_pipeline import (
    compute_ela, compute_fft_map, compute_dct_map,
    compute_wavelet_map, compute_srm_residual, extract_metadata,
    metadata_to_vector
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

RISK_THRESHOLDS = {
    "CRITICAL": 0.85,
    "HIGH":     0.65,
    "MEDIUM":   0.40,
    "LOW":      0.20,
    "MINIMAL":  0.00,
}

MODEL_WEIGHTS = {
    "rgb":          0.20,
    "vit":          0.20,
    "frequency":    0.15,
    "noise":        0.15,
    "ela":          0.10,
    "face":         0.10,
    "localization": 0.10,
}

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────

class ModelRegistry:
    """Lazy-loads and caches all forensic models."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._models: Dict[str, torch.nn.Module] = {}
        self._fusion = None

    def _load_checkpoint(self, model: torch.nn.Module, name: str) -> torch.nn.Module:
        ckpt_path = self.checkpoint_dir / f"{name}_best.pth"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=DEVICE)
            # Support both raw state_dict and wrapped checkpoints
            sd = state.get("ema_state_dict") or state.get("model_state_dict") or state
            model.load_state_dict(sd, strict=False)
            logger.info(f"Loaded {name} from {ckpt_path}")
        else:
            logger.warning(f"No checkpoint for {name} at {ckpt_path} — using random weights")
        model.to(DEVICE).eval()
        return model

    def get(self, name: str) -> torch.nn.Module:
        if name not in self._models:
            model = self._build(name)
            self._models[name] = self._load_checkpoint(model, name)
        return self._models[name]

    def _build(self, name: str) -> torch.nn.Module:
        if name == "rgb":
            return RGBForensicsNet(backbone="convnextv2_large", num_classes=1, num_manip_classes=8)
        elif name == "vit":
            return ViTForensicsNet(backbone="vit_large_patch16_224", num_classes=1, num_source_classes=4)
        elif name == "frequency":
            return FrequencyForensicsNet(backbone="tf_efficientnetv2_m", num_classes=1)
        elif name == "noise":
            return NoiseForensicsNet(num_classes=1, learnable_srm=False)
        elif name == "ela":
            return ELAForensicsNet(backbone="tf_efficientnetv2_s", num_classes=1)
        elif name == "face":
            return FaceForensicsNet(backbone="tf_efficientnetv2_m", num_classes=1, num_forgery_classes=4)
        elif name == "localization":
            return ManipulationLocalizationNet(backbone="swin_base_patch4_window7_224")
        else:
            raise ValueError(f"Unknown model: {name}")

    def get_fusion(self) -> Optional[Any]:
        if self._fusion is None:
            fusion_path = self.checkpoint_dir / "fusion_best.pth"
            if fusion_path.exists():
                state = torch.load(fusion_path, map_location=DEVICE)
                model = NeuralMetaClassifier(num_models=7, hidden_dim=128)
                model.load_state_dict(state.get("model_state_dict", state))
                model.to(DEVICE).eval()
                self._fusion = model
            else:
                logger.warning("No fusion checkpoint found — using weighted average")
        return self._fusion

    def warmup(self, names: Optional[List[str]] = None):
        """Pre-load all models into GPU memory."""
        targets = names or ["rgb", "vit", "frequency", "noise", "ela", "face", "localization"]
        for name in targets:
            _ = self.get(name)
        logger.info(f"Warmed up {len(targets)} models on {DEVICE}")


# ──────────────────────────────────────────────
# Image Preprocessor
# ──────────────────────────────────────────────

class ForensicPreprocessor:
    """Converts raw image into all forensic modalities required by each model."""

    def __init__(self, image_size: int = IMAGE_SIZE, ela_quality: int = 90):
        self.image_size = image_size
        self.ela_quality = ela_quality
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def load(self, source: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, Dict]:
        """Load from path, numpy array, or PIL image. Returns (rgb_uint8_hwc, metadata)."""
        if isinstance(source, str):
            pil = Image.open(source).convert("RGB")
            metadata = extract_metadata(source)
        elif isinstance(source, np.ndarray):
            pil = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB) if source.shape[2] == 3 else source)
            metadata = {}
        elif isinstance(source, Image.Image):
            pil = source.convert("RGB")
            metadata = {}
        else:
            raise TypeError(f"Unsupported input type: {type(source)}")

        rgb = np.array(pil)
        return rgb, metadata

    def to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """HWC uint8 → CHW float normalized tensor."""
        resized = cv2.resize(rgb, (self.image_size, self.image_size))
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return (t - self.mean) / self.std

    def process(self, source: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Full preprocessing into all modalities."""
        rgb_orig, metadata = self.load(source)

        # Resize for model input
        rgb_resized = cv2.resize(rgb_orig, (self.image_size, self.image_size))

        # All modalities
        ela_map  = compute_ela(rgb_orig, quality=self.ela_quality)
        ela_r    = cv2.resize(ela_map, (self.image_size, self.image_size))

        fft_map  = compute_fft_map(rgb_orig)
        fft_r    = cv2.resize(fft_map, (self.image_size, self.image_size))

        dct_map  = compute_dct_map(rgb_orig)
        dct_r    = cv2.resize(dct_map, (self.image_size, self.image_size))

        wav_map  = compute_wavelet_map(rgb_orig)
        wav_r    = cv2.resize(wav_map, (self.image_size, self.image_size))

        srm_map  = compute_srm_residual(rgb_orig)
        srm_r    = cv2.resize(srm_map, (self.image_size, self.image_size))

        meta_vec = torch.tensor(metadata_to_vector(metadata), dtype=torch.float32)

        # Normalize to tensors
        def norm(arr, c=3):
            if arr.ndim == 2:
                arr = np.stack([arr]*c, axis=-1)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            return (t - self.mean[:c]) / self.std[:c]

        freq_tensor = torch.cat([
            norm(fft_r), norm(dct_r), norm(wav_r)
        ], dim=0)  # 9 × H × W

        return {
            "rgb":       self.to_tensor(rgb_resized).unsqueeze(0),        # 1,3,H,W
            "ela":       norm(ela_r).unsqueeze(0),                        # 1,3,H,W
            "freq":      freq_tensor.unsqueeze(0),                        # 1,9,H,W
            "srm":       norm(srm_map[:self.image_size, :self.image_size] if srm_map.shape[0] >= self.image_size else
                              cv2.resize(srm_map, (self.image_size, self.image_size))).unsqueeze(0),
            "meta":      meta_vec.unsqueeze(0),                           # 1,D
            "rgb_orig":  rgb_orig,
            "metadata":  metadata,
        }


# ──────────────────────────────────────────────
# Per-Model Inference Runners
# ──────────────────────────────────────────────

@torch.no_grad()
def run_rgb_model(model: torch.nn.Module, inputs: Dict) -> Dict[str, float]:
    rgb = inputs["rgb"].to(DEVICE)
    out = model(rgb)
    score = torch.sigmoid(out["logit"]).item()
    manip_probs = torch.softmax(out.get("manip_logit", torch.zeros(1, 8)), dim=-1).squeeze().tolist()
    return {"rgb_score": score, "rgb_manip_probs": manip_probs}


@torch.no_grad()
def run_vit_model(model: torch.nn.Module, inputs: Dict) -> Dict[str, float]:
    rgb = inputs["rgb"].to(DEVICE)
    out = model(rgb)
    score = torch.sigmoid(out["logit"]).item()
    source_probs = torch.softmax(out.get("source_logit", torch.zeros(1, 4)), dim=-1).squeeze().tolist()
    return {"vit_score": score, "vit_source_probs": source_probs}


@torch.no_grad()
def run_frequency_model(model: torch.nn.Module, inputs: Dict) -> Dict[str, float]:
    freq = inputs["freq"].to(DEVICE)
    out = model(freq)
    score = torch.sigmoid(out["logit"]).item()
    return {"frequency_score": score}


@torch.no_grad()
def run_noise_model(model: torch.nn.Module, inputs: Dict) -> Dict[str, float]:
    rgb = inputs["rgb"].to(DEVICE)
    out = model(rgb)
    score = torch.sigmoid(out["logit"]).item()
    noise_level = out.get("noise_level", torch.tensor(0.0)).item()
    return {"noise_score": score, "noise_level": noise_level}


@torch.no_grad()
def run_ela_model(model: torch.nn.Module, inputs: Dict) -> Dict[str, float]:
    ela = inputs["ela"].to(DEVICE)
    out = model(ela)
    score = torch.sigmoid(out["logit"]).item()
    return {"ela_score": score}


@torch.no_grad()
def run_face_model(model: torch.nn.Module, inputs: Dict) -> Dict[str, float]:
    rgb = inputs["rgb"].to(DEVICE)
    out = model(rgb)
    score = torch.sigmoid(out["logit"]).item()
    identity_score = torch.sigmoid(out.get("identity_logit", torch.tensor(0.0))).item()
    forgery_probs = torch.softmax(out.get("forgery_logit", torch.zeros(1, 4)), dim=-1).squeeze().tolist()
    return {
        "face_score": score,
        "identity_inconsistency": identity_score,
        "forgery_type_probs": forgery_probs,
    }


@torch.no_grad()
def run_localization_model(model: torch.nn.Module, inputs: Dict) -> Dict[str, Any]:
    rgb = inputs["rgb"].to(DEVICE)
    out = model(rgb)
    mask_logit = out["mask"]                                    # 1,1,H,W
    mask_prob  = torch.sigmoid(mask_logit).squeeze().cpu().numpy()  # H,W ∈ [0,1]
    image_score = torch.sigmoid(out.get("image_logit", torch.tensor(0.0))).item()
    return {"localization_score": image_score, "mask": mask_prob}


# ──────────────────────────────────────────────
# Metadata Analysis
# ──────────────────────────────────────────────

METADATA_RED_FLAGS = [
    "Adobe Photoshop", "GIMP", "Stable Diffusion", "Midjourney",
    "ComfyUI", "Automatic1111", "InvokeAI", "RunwayML",
    "FaceApp", "Reface", "DeepFaceLab",
]

def analyze_metadata(metadata: Dict) -> Dict[str, Any]:
    flags = []
    score = 0.0

    software = metadata.get("software", "") or ""
    for rf in METADATA_RED_FLAGS:
        if rf.lower() in software.lower():
            flags.append(f"Software: {software}")
            score = max(score, 0.9)
            break

    # Missing EXIF for JPEG is suspicious
    if not metadata.get("exif_present", True):
        flags.append("Missing EXIF data")
        score = max(score, 0.3)

    # GPS in portrait but no datetime = likely edited
    if metadata.get("has_gps") and not metadata.get("datetime"):
        flags.append("GPS without datetime")
        score = max(score, 0.25)

    # Detect re-encoding traces: compression count field if available
    if metadata.get("compression_count", 1) > 1:
        flags.append(f"Re-encoded {metadata['compression_count']} times")
        score = max(score, 0.4)

    # Comment / prompt embedding (AI generators often embed prompts)
    comment = str(metadata.get("image_description", "") or metadata.get("user_comment", "") or "")
    if any(kw in comment.lower() for kw in ["prompt", "steps", "sampler", "cfg scale", "seed"]):
        flags.append("AI generation prompt found in metadata")
        score = max(score, 0.95)

    return {"metadata_score": score, "metadata_flags": flags}


# ──────────────────────────────────────────────
# Fusion
# ──────────────────────────────────────────────

def weighted_average_fusion(scores: Dict[str, float]) -> float:
    total, weight_sum = 0.0, 0.0
    for key, w in MODEL_WEIGHTS.items():
        score_key = f"{key}_score"
        if score_key in scores:
            total += scores[score_key] * w
            weight_sum += w
    return total / weight_sum if weight_sum > 0 else 0.5


def neural_fusion(
    fusion_model: Optional[torch.nn.Module],
    scores: Dict[str, float],
) -> float:
    if fusion_model is None:
        return weighted_average_fusion(scores)

    keys = ["rgb_score", "vit_score", "frequency_score", "noise_score",
            "ela_score", "face_score", "localization_score"]
    vec = torch.tensor([[scores.get(k, 0.5) for k in keys]], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = fusion_model(vec)
        return torch.sigmoid(out).item()


# ──────────────────────────────────────────────
# Report Builder
# ──────────────────────────────────────────────

MANIPULATION_NAMES = [
    "pristine", "face_swap", "inpainting", "splicing",
    "copy_move", "object_insertion", "color_grading", "ai_generated"
]

SOURCE_NAMES = ["real", "gan_generated", "diffusion_generated", "deepfake"]
FORGERY_NAMES = ["pristine", "face_swap", "expression_transfer", "identity_replacement"]


def classify_risk(score: float) -> str:
    for level, threshold in RISK_THRESHOLDS.items():
        if score >= threshold:
            return level
    return "MINIMAL"


def build_report(
    inputs: Dict,
    model_outputs: Dict[str, Any],
    final_score: float,
    heatmap: Optional[np.ndarray],
    elapsed: float,
) -> Dict[str, Any]:
    scores = model_outputs

    # Determine dominant manipulation type from RGB model
    manip_probs = scores.get("rgb_manip_probs", [1.0] + [0.0]*7)
    dominant_manip = MANIPULATION_NAMES[int(np.argmax(manip_probs))]

    # Determine source type from ViT model
    source_probs = scores.get("vit_source_probs", [1.0] + [0.0]*3)
    dominant_source = SOURCE_NAMES[int(np.argmax(source_probs))]

    # Forgery type
    forgery_probs = scores.get("forgery_type_probs", [1.0] + [0.0]*3)
    dominant_forgery = FORGERY_NAMES[int(np.argmax(forgery_probs))]

    # Image hash for traceability
    img_bytes = inputs["rgb_orig"].tobytes()
    img_hash = hashlib.sha256(img_bytes).hexdigest()

    risk_level = classify_risk(final_score)

    # Compile per-detector findings
    findings = []
    if scores.get("rgb_score", 0) > 0.5:
        findings.append(f"RGB analysis detects {dominant_manip} (p={scores['rgb_score']:.3f})")
    if scores.get("vit_score", 0) > 0.5:
        findings.append(f"Transformer detects {dominant_source} artifacts (p={scores['vit_score']:.3f})")
    if scores.get("frequency_score", 0) > 0.5:
        findings.append(f"Frequency domain shows synthetic patterns (p={scores['frequency_score']:.3f})")
    if scores.get("noise_score", 0) > 0.5:
        findings.append(f"Noise residuals indicate tampering (p={scores['noise_score']:.3f})")
    if scores.get("ela_score", 0) > 0.5:
        findings.append(f"ELA reveals re-compression artifacts (p={scores['ela_score']:.3f})")
    if scores.get("face_score", 0) > 0.5:
        findings.append(f"Face analysis: {dominant_forgery} detected (p={scores['face_score']:.3f})")
    if scores.get("metadata_score", 0) > 0.3:
        for flag in scores.get("metadata_flags", []):
            findings.append(f"Metadata: {flag}")

    # Localized region statistics
    mask = scores.get("mask")
    manipulated_fraction = float(np.mean(mask > 0.5)) if mask is not None else None

    report_text = (
        f"FORENSIC ANALYSIS REPORT\n"
        f"{'='*40}\n"
        f"Risk Level  : {risk_level}\n"
        f"Confidence  : {final_score:.4f}\n"
        f"Image SHA256: {img_hash[:16]}...\n"
        f"Elapsed     : {elapsed:.2f}s\n\n"
        f"DETECTOR SCORES\n"
        f"  RGB Model       : {scores.get('rgb_score', 'N/A'):.3f}\n"
        f"  Transformer     : {scores.get('vit_score', 'N/A'):.3f}\n"
        f"  Frequency       : {scores.get('frequency_score', 'N/A'):.3f}\n"
        f"  Noise           : {scores.get('noise_score', 'N/A'):.3f}\n"
        f"  ELA             : {scores.get('ela_score', 'N/A'):.3f}\n"
        f"  Face            : {scores.get('face_score', 'N/A'):.3f}\n"
        f"  Localization    : {scores.get('localization_score', 'N/A'):.3f}\n"
        f"  Metadata        : {scores.get('metadata_score', 'N/A'):.3f}\n\n"
        f"FINDINGS\n"
    ) + "\n".join(f"  • {f}" for f in findings) + (
        f"\n\nMANIPULATED REGION\n"
        f"  {manipulated_fraction*100:.1f}% of image pixels flagged\n"
        if manipulated_fraction is not None else ""
    )

    return {
        "status":              "MANIPULATED" if final_score >= 0.5 else "AUTHENTIC",
        "confidence":          round(final_score, 6),
        "deepfake_score":      round(scores.get("face_score", 0.0), 6),
        "ai_generated_score":  round(scores.get("vit_score", 0.0), 6),
        "manipulation_score":  round(scores.get("rgb_score", 0.0), 6),
        "frequency_score":     round(scores.get("frequency_score", 0.0), 6),
        "noise_score":         round(scores.get("noise_score", 0.0), 6),
        "ela_score":           round(scores.get("ela_score", 0.0), 6),
        "metadata_score":      round(scores.get("metadata_score", 0.0), 6),
        "risk_level":          risk_level,
        "dominant_manipulation": dominant_manip,
        "dominant_source":     dominant_source,
        "manipulated_fraction": manipulated_fraction,
        "metadata_flags":      scores.get("metadata_flags", []),
        "findings":            findings,
        "image_hash":          img_hash,
        "heatmap":             heatmap,
        "mask":                scores.get("mask"),
        "report":              report_text,
        "elapsed_seconds":     round(elapsed, 3),
    }


# ──────────────────────────────────────────────
# Main ForensicEngine
# ──────────────────────────────────────────────

class ForensicEngine:
    """
    Production-grade forensic AI engine.

    Usage:
        engine = ForensicEngine(checkpoint_dir="outputs/checkpoints")
        engine.warmup()
        result = engine.analyze_image("suspect.jpg")
        print(result["report"])
    """

    def __init__(
        self,
        checkpoint_dir: str = "outputs/checkpoints",
        image_size: int = IMAGE_SIZE,
        ela_quality: int = 90,
        generate_heatmap: bool = True,
        generate_mask: bool = True,
        models_to_run: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        global DEVICE
        if device is not None:
            DEVICE = device

        self.registry      = ModelRegistry(checkpoint_dir)
        self.preprocessor  = ForensicPreprocessor(image_size, ela_quality)
        self.generate_heatmap = generate_heatmap
        self.generate_mask    = generate_mask
        self.models_to_run = models_to_run or ["rgb", "vit", "frequency", "noise", "ela", "face", "localization"]
        self._explainer: Optional[ForensicExplainer] = None

    def warmup(self):
        self.registry.warmup(self.models_to_run)
        logger.info("ForensicEngine warmed up and ready.")

    # ── Public entry point ──────────────────────

    def analyze_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
        explain: bool = False,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full forensic analysis pipeline.

        Args:
            image: File path, numpy BGR/RGB array, or PIL Image.
            explain: If True, generates GradCAM heatmap via explainability module.
            save_dir: If provided, saves heatmap + mask + JSON report here.

        Returns:
            Structured forensic report dict.
        """
        t0 = time.time()

        # ── 1. Preprocess ──
        inputs = self.preprocessor.process(image)

        # ── 2. Run each model ──
        all_scores: Dict[str, Any] = {}

        if "rgb" in self.models_to_run:
            try:
                all_scores.update(run_rgb_model(self.registry.get("rgb"), inputs))
            except Exception as e:
                logger.warning(f"RGB model failed: {e}")
                all_scores["rgb_score"] = 0.5

        if "vit" in self.models_to_run:
            try:
                all_scores.update(run_vit_model(self.registry.get("vit"), inputs))
            except Exception as e:
                logger.warning(f"ViT model failed: {e}")
                all_scores["vit_score"] = 0.5

        if "frequency" in self.models_to_run:
            try:
                all_scores.update(run_frequency_model(self.registry.get("frequency"), inputs))
            except Exception as e:
                logger.warning(f"Frequency model failed: {e}")
                all_scores["frequency_score"] = 0.5

        if "noise" in self.models_to_run:
            try:
                all_scores.update(run_noise_model(self.registry.get("noise"), inputs))
            except Exception as e:
                logger.warning(f"Noise model failed: {e}")
                all_scores["noise_score"] = 0.5

        if "ela" in self.models_to_run:
            try:
                all_scores.update(run_ela_model(self.registry.get("ela"), inputs))
            except Exception as e:
                logger.warning(f"ELA model failed: {e}")
                all_scores["ela_score"] = 0.5

        if "face" in self.models_to_run:
            try:
                all_scores.update(run_face_model(self.registry.get("face"), inputs))
            except Exception as e:
                logger.warning(f"Face model failed: {e}")
                all_scores["face_score"] = 0.5

        if "localization" in self.models_to_run:
            try:
                all_scores.update(run_localization_model(self.registry.get("localization"), inputs))
            except Exception as e:
                logger.warning(f"Localization model failed: {e}")
                all_scores["localization_score"] = 0.5
                all_scores["mask"] = None

        # ── 3. Metadata analysis ──
        all_scores.update(analyze_metadata(inputs["metadata"]))

        # ── 4. Fusion ──
        fusion_model = self.registry.get_fusion()
        final_score = neural_fusion(fusion_model, all_scores)

        # ── 5. Explainability / Heatmap ──
        heatmap = None
        if self.generate_heatmap or explain:
            try:
                rgb_model = self.registry.get("rgb")
                if self._explainer is None:
                    self._explainer = ForensicExplainer(rgb_model, method="gradcam++")
                heatmap = self._explainer.explain(
                    inputs["rgb"].to(DEVICE),
                    inputs["rgb_orig"],
                )
            except Exception as e:
                logger.warning(f"Explainability failed: {e}")

        # ── 6. Build report ──
        elapsed = time.time() - t0
        report = build_report(inputs, all_scores, final_score, heatmap, elapsed)

        # ── 7. Optionally save outputs ──
        if save_dir:
            self._save_outputs(report, save_dir)

        return report

    def _save_outputs(self, report: Dict, save_dir: str):
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        # Save heatmap
        if report["heatmap"] is not None:
            hmap_path = out / f"heatmap_{ts}.png"
            cv2.imwrite(str(hmap_path), report["heatmap"])
            logger.info(f"Heatmap saved: {hmap_path}")

        # Save mask
        if report["mask"] is not None:
            mask_vis = (report["mask"] * 255).astype(np.uint8)
            mask_path = out / f"mask_{ts}.png"
            cv2.imwrite(str(mask_path), mask_vis)
            logger.info(f"Mask saved: {mask_path}")

        # Save JSON report (without numpy arrays)
        json_report = {
            k: v for k, v in report.items()
            if not isinstance(v, np.ndarray)
        }
        json_path = out / f"report_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2, default=str)
        logger.info(f"Report saved: {json_path}")

        # Save text report
        txt_path = out / f"report_{ts}.txt"
        with open(txt_path, "w") as f:
            f.write(report["report"])
        logger.info(f"Text report saved: {txt_path}")

    def batch_analyze(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Analyze a list of images."""
        results = []
        for i, img in enumerate(images):
            if verbose:
                logger.info(f"Analyzing {i+1}/{len(images)} ...")
            r = self.analyze_image(img, save_dir=save_dir)
            results.append(r)
        return results


# ──────────────────────────────────────────────
# Module-level convenience function
# ──────────────────────────────────────────────

_global_engine: Optional[ForensicEngine] = None


def analyze_image(
    image: Union[str, np.ndarray, Image.Image],
    checkpoint_dir: str = "outputs/checkpoints",
    explain: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function — initializes global engine on first call.

    Args:
        image:          File path, numpy array, or PIL Image.
        checkpoint_dir: Path to model checkpoints.
        explain:        Generate GradCAM explanation.
        save_dir:       Directory to save heatmap, mask, JSON.

    Returns:
        Full forensic report dict.

    Example:
        from src.inference.pipeline import analyze_image
        result = analyze_image("photo.jpg", save_dir="outputs/reports")
        print(result["status"], result["confidence"])
    """
    global _global_engine
    if _global_engine is None:
        _global_engine = ForensicEngine(checkpoint_dir=checkpoint_dir)
        _global_engine.warmup()
    return _global_engine.analyze_image(image, explain=explain, save_dir=save_dir)
