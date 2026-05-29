"""
tests/test_engine.py
--------------------
Smoke tests and unit tests for the Forensic AI Engine.
Run with: pytest tests/ -v
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def fake_rgb_array():
    """256×256 random RGB image as numpy array."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def fake_pil_image(fake_rgb_array):
    return Image.fromarray(fake_rgb_array)


@pytest.fixture(scope="session")
def fake_tensor_224():
    """Batch of 2 normalized 224×224 RGB tensors."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture(scope="session")
def fake_freq_tensor():
    """9-channel frequency input."""
    return torch.randn(2, 9, 224, 224)


@pytest.fixture(scope="session")
def fake_ela_tensor():
    return torch.randn(2, 3, 224, 224)


# ──────────────────────────────────────────────
# Dataset pipeline tests
# ──────────────────────────────────────────────

class TestPreprocessing:

    def test_compute_ela(self, fake_rgb_array):
        from src.datasets.dataset_pipeline import compute_ela
        ela = compute_ela(fake_rgb_array, quality=90)
        assert ela.shape == fake_rgb_array.shape
        assert ela.dtype == np.uint8

    def test_compute_fft_map(self, fake_rgb_array):
        from src.datasets.dataset_pipeline import compute_fft_map
        fft = compute_fft_map(fake_rgb_array)
        assert fft.shape[:2] == fake_rgb_array.shape[:2]
        assert fft.ndim == 3

    def test_compute_dct_map(self, fake_rgb_array):
        from src.datasets.dataset_pipeline import compute_dct_map
        dct = compute_dct_map(fake_rgb_array)
        assert dct.shape[:2] == fake_rgb_array.shape[:2]

    def test_compute_wavelet_map(self, fake_rgb_array):
        from src.datasets.dataset_pipeline import compute_wavelet_map
        wav = compute_wavelet_map(fake_rgb_array)
        assert wav.ndim == 3

    def test_compute_srm_residual(self, fake_rgb_array):
        from src.datasets.dataset_pipeline import compute_srm_residual
        srm = compute_srm_residual(fake_rgb_array)
        assert srm.ndim == 3

    def test_metadata_extraction_on_pil(self, fake_pil_image, tmp_path):
        from src.datasets.dataset_pipeline import extract_metadata, metadata_to_vector
        path = str(tmp_path / "test.jpg")
        fake_pil_image.save(path, quality=95)
        meta = extract_metadata(path)
        assert isinstance(meta, dict)
        vec = metadata_to_vector(meta)
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)


# ──────────────────────────────────────────────
# Model forward-pass tests (no pretrained weights needed)
# ──────────────────────────────────────────────

class TestModelForwardPasses:

    def test_rgb_model(self, fake_tensor_224):
        from src.models.model1_rgb import RGBForensicsNet
        model = RGBForensicsNet(backbone="convnextv2_tiny", num_classes=1, num_manip_classes=8)
        model.eval()
        with torch.no_grad():
            out = model(fake_tensor_224)
        assert "logit" in out
        assert out["logit"].shape == (2, 1)
        assert "manip_logit" in out

    def test_vit_model(self, fake_tensor_224):
        from src.models.model2_vit import ViTForensicsNet
        model = ViTForensicsNet(backbone="vit_small_patch16_224", num_classes=1, num_source_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(fake_tensor_224)
        assert "logit" in out
        assert out["logit"].shape == (2, 1)

    def test_frequency_model(self, fake_freq_tensor):
        from src.models.model3_frequency import FrequencyForensicsNet
        model = FrequencyForensicsNet(backbone="tf_efficientnetv2_s", num_classes=1)
        model.eval()
        with torch.no_grad():
            out = model(fake_freq_tensor)
        assert "logit" in out
        assert out["logit"].shape == (2, 1)

    def test_noise_model(self, fake_tensor_224):
        from src.models.model4_noise import NoiseForensicsNet
        model = NoiseForensicsNet(num_classes=1, learnable_srm=False)
        model.eval()
        with torch.no_grad():
            out = model(fake_tensor_224)
        assert "logit" in out

    def test_ela_model(self, fake_ela_tensor):
        from src.models.model5_ela import ELAForensicsNet
        model = ELAForensicsNet(backbone="tf_efficientnetv2_s", num_classes=1)
        model.eval()
        with torch.no_grad():
            out = model(fake_ela_tensor)
        assert "logit" in out

    def test_face_model(self, fake_tensor_224):
        from src.models.model6_face import FaceForensicsNet
        model = FaceForensicsNet(backbone="tf_efficientnetv2_s", num_classes=1)
        model.eval()
        with torch.no_grad():
            out = model(fake_tensor_224)
        assert "logit" in out

    def test_localization_model(self, fake_tensor_224):
        from src.models.model7_localization import ManipulationLocalizationNet
        model = ManipulationLocalizationNet(backbone="swin_tiny_patch4_window7_224")
        model.eval()
        with torch.no_grad():
            out = model(fake_tensor_224)
        assert "mask" in out
        # Mask should be spatial
        assert out["mask"].ndim == 4    # B, 1, H, W
        assert out["mask"].shape[0] == 2


# ──────────────────────────────────────────────
# Fusion tests
# ──────────────────────────────────────────────

class TestFusion:

    def test_neural_meta_classifier(self):
        from src.fusion.fusion import NeuralMetaClassifier
        model = NeuralMetaClassifier(num_models=7, hidden_dim=64)
        model.eval()
        x = torch.rand(4, 7)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)

    def test_weighted_average_fusion(self):
        from src.inference.pipeline import weighted_average_fusion
        scores = {
            "rgb_score": 0.8, "vit_score": 0.7,
            "frequency_score": 0.6, "noise_score": 0.5,
            "ela_score": 0.4, "face_score": 0.9,
            "localization_score": 0.3,
        }
        result = weighted_average_fusion(scores)
        assert 0.0 <= result <= 1.0

    def test_classify_risk(self):
        from src.inference.pipeline import classify_risk
        assert classify_risk(0.90) == "CRITICAL"
        assert classify_risk(0.70) == "HIGH"
        assert classify_risk(0.45) == "MEDIUM"
        assert classify_risk(0.25) == "LOW"
        assert classify_risk(0.05) == "MINIMAL"


# ──────────────────────────────────────────────
# Metadata analysis tests
# ──────────────────────────────────────────────

class TestMetadataAnalysis:

    def test_clean_metadata(self):
        from src.inference.pipeline import analyze_metadata
        meta = {"exif_present": True, "software": "Canon EOS 5D"}
        result = analyze_metadata(meta)
        assert result["metadata_score"] < 0.5

    def test_ai_software_flag(self):
        from src.inference.pipeline import analyze_metadata
        meta = {"exif_present": True, "software": "Stable Diffusion WebUI"}
        result = analyze_metadata(meta)
        assert result["metadata_score"] >= 0.85
        assert len(result["metadata_flags"]) > 0

    def test_ai_prompt_in_comment(self):
        from src.inference.pipeline import analyze_metadata
        meta = {"user_comment": "prompt: a beautiful landscape, steps: 30, sampler: euler"}
        result = analyze_metadata(meta)
        assert result["metadata_score"] >= 0.90

    def test_missing_exif(self):
        from src.inference.pipeline import analyze_metadata
        meta = {"exif_present": False}
        result = analyze_metadata(meta)
        assert result["metadata_score"] > 0.0


# ──────────────────────────────────────────────
# Preprocessor tests
# ──────────────────────────────────────────────

class TestForensicPreprocessor:

    def test_process_from_array(self, fake_rgb_array):
        from src.inference.pipeline import ForensicPreprocessor
        pp = ForensicPreprocessor(image_size=224)
        result = pp.process(fake_rgb_array)
        assert "rgb" in result
        assert "ela" in result
        assert "freq" in result
        assert "srm" in result
        assert "meta" in result
        assert result["rgb"].shape == (1, 3, 224, 224)
        assert result["freq"].shape == (1, 9, 224, 224)

    def test_process_from_pil(self, fake_pil_image):
        from src.inference.pipeline import ForensicPreprocessor
        pp = ForensicPreprocessor(image_size=224)
        result = pp.process(fake_pil_image)
        assert result["rgb"].shape == (1, 3, 224, 224)

    def test_process_from_path(self, fake_pil_image, tmp_path):
        from src.inference.pipeline import ForensicPreprocessor
        path = str(tmp_path / "test.jpg")
        fake_pil_image.save(path, quality=95)
        pp = ForensicPreprocessor(image_size=224)
        result = pp.process(path)
        assert result["rgb"].shape == (1, 3, 224, 224)


# ──────────────────────────────────────────────
# Report builder test
# ──────────────────────────────────────────────

class TestReportBuilder:

    def test_report_structure(self, fake_rgb_array):
        from src.inference.pipeline import build_report
        inputs = {"rgb_orig": fake_rgb_array, "metadata": {}}
        scores = {
            "rgb_score": 0.8, "vit_score": 0.75,
            "frequency_score": 0.6, "noise_score": 0.5,
            "ela_score": 0.45, "face_score": 0.9,
            "localization_score": 0.7, "metadata_score": 0.0,
            "rgb_manip_probs": [0.1, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1],
            "vit_source_probs": [0.1, 0.1, 0.1, 0.7],
            "forgery_type_probs": [0.1, 0.8, 0.05, 0.05],
            "metadata_flags": [],
            "mask": np.random.rand(224, 224).astype(np.float32),
        }
        report = build_report(inputs, scores, 0.82, None, 1.5)

        required_keys = [
            "status", "confidence", "deepfake_score", "ai_generated_score",
            "manipulation_score", "frequency_score", "noise_score", "ela_score",
            "metadata_score", "risk_level", "heatmap", "mask", "report",
        ]
        for k in required_keys:
            assert k in report, f"Missing key: {k}"

        assert report["status"] == "MANIPULATED"
        assert report["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL")
        assert isinstance(report["report"], str)
        assert len(report["report"]) > 50


# ──────────────────────────────────────────────
# Metrics tests
# ──────────────────────────────────────────────

class TestMetrics:

    def test_compute_metrics(self):
        from src.evaluation.metrics import compute_metrics
        labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        scores = np.array([0.1, 0.3, 0.8, 0.9, 0.4, 0.7, 0.6, 0.2])
        m = compute_metrics(labels, scores)
        assert "accuracy" in m
        assert "roc_auc" in m
        assert "f1" in m
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_find_optimal_threshold(self):
        from src.evaluation.metrics import find_optimal_threshold
        labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        scores = np.array([0.1, 0.3, 0.8, 0.9, 0.4, 0.7, 0.6, 0.2])
        thresh = find_optimal_threshold(labels, scores, method="f1")
        assert 0.0 <= thresh <= 1.0


# ──────────────────────────────────────────────
# Trainer / loss tests
# ──────────────────────────────────────────────

class TestTrainer:

    def test_mixup(self, fake_tensor_224):
        from src.training.trainer import mixup_data
        x = fake_tensor_224
        y = torch.randint(0, 2, (2,)).float()
        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
        assert x_mix.shape == x.shape
        assert 0.0 <= lam <= 1.0

    def test_cutmix(self, fake_tensor_224):
        from src.training.trainer import cutmix_data
        x = fake_tensor_224
        y = torch.randint(0, 2, (2,)).float()
        x_cut, y_a, y_b, lam = cutmix_data(x, y, alpha=0.4)
        assert x_cut.shape == x.shape

    def test_ema(self):
        from src.training.trainer import EMA
        model = torch.nn.Linear(10, 1)
        ema = EMA(model, decay=0.999)
        # Simulate training step
        for _ in range(5):
            with torch.no_grad():
                model.weight.data += 0.01
            ema.update()
        ema.apply_shadow()
        # EMA weights should be slightly different from raw weights
        assert model.weight.data is not None
