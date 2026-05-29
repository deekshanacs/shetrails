# Forensic AI Engine

A research-grade, production-quality multi-model image forensics and deepfake detection system.

## Detection Capabilities

| Category | Examples |
|---|---|
| **Deepfakes** | Face swaps, identity replacement, expression transfer |
| **AI-Generated** | Stable Diffusion, Midjourney, DALL-E, Flux, GAN outputs |
| **Image Manipulation** | Splicing, inpainting, copy-move, object insertion/removal |
| **Photoshop Edits** | Color grading, retouching, composite images |
| **Metadata Tampering** | EXIF stripping, software traces, AI prompt embedding |
| **Compression Artifacts** | Re-encoding traces, JPEG inconsistencies |

---

## Architecture

Seven specialized forensic models fused into a single decision:

```
Input Image
    │
    ├──► Model 1: RGB Forensics       (ConvNeXt V2-L)
    ├──► Model 2: Vision Transformer  (ViT-L / Swin-L / EVA)
    ├──► Model 3: Frequency Domain    (FFT + DCT + Wavelet → EfficientNetV2)
    ├──► Model 4: Noise Forensics     (SRM filters → ResNet50)
    ├──► Model 5: ELA Network         (Error Level Analysis → EfficientNetV2)
    ├──► Model 6: Face Forensics      (EfficientNetV2 + boundary detection)
    ├──► Model 7: Localization        (Swin + SegFormer → pixel mask)
    └──► Metadata Analysis            (EXIF, software, GPS, compression)
             │
             └──► Fusion Layer (XGBoost / LightGBM / Neural Meta-Classifier)
                       │
                       └──► Forensic Report JSON
```

---

## Project Structure

```
forensic_ai/
├── configs/
│   └── config.yaml                  # All hyperparameters & dataset paths
├── data/
│   ├── raw/                         # Original datasets
│   ├── processed/                   # Extracted frames, ELA maps
│   └── splits/                      # Train/val/test manifests
├── src/
│   ├── datasets/
│   │   └── dataset_pipeline.py      # Unified dataset builder + all modalities
│   ├── models/
│   │   ├── model1_rgb.py            # ConvNeXt V2 + dual pooling
│   │   ├── model2_vit.py            # ViT/Swin/EVA + patch attention
│   │   ├── model3_frequency.py      # FFT/DCT/Wavelet CNN
│   │   ├── model4_noise.py          # SRM filter bank + noise consistency
│   │   ├── model5_ela.py            # Multi-quality ELA CNN
│   │   ├── model6_face.py           # Face boundary + landmark forensics
│   │   └── model7_localization.py   # Pixel-level segmentation mask
│   ├── training/
│   │   └── trainer.py               # Universal trainer: AMP, EMA, mixup, SimCLR
│   ├── evaluation/
│   │   └── metrics.py               # Full metrics, cross-dataset, OOD eval
│   ├── explainability/
│   │   └── explain.py               # GradCAM, GradCAM++, IG, AttentionRollout
│   ├── fusion/
│   │   └── fusion.py                # XGBoost / LightGBM / Neural meta-classifier
│   └── inference/
│       └── pipeline.py              # analyze_image() — full forensic report
├── scripts/
│   ├── train_models.py              # CLI training for all models
│   └── run_inference.py             # CLI inference
├── tests/
├── outputs/
│   ├── checkpoints/                 # Saved model weights
│   ├── logs/                        # TensorBoard / WandB logs
│   ├── reports/                     # JSON + text forensic reports
│   ├── heatmaps/                    # GradCAM visualizations
│   └── masks/                       # Pixel-level manipulation masks
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/forensic-ai-engine
cd forensic-ai-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install (full dependencies including face detection, grad-cam, etc.)
pip install -e ".[full]"

# Or minimal install
pip install -r requirements.txt
```

**CUDA setup (recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Datasets

Place raw datasets under `data/raw/` with the following folder names:

| Dataset | Folder | Description |
|---|---|---|
| FaceForensics++ | `data/raw/faceforensics` | Deepfake, face swap, neural textures |
| Celeb-DF v2 | `data/raw/celebdf` | High-quality deepfake faces |
| DFDC | `data/raw/dfdc` | Facebook DFDC challenge set |
| CASIA v2 | `data/raw/casia` | Splicing, copy-move, image manipulation |
| Columbia | `data/raw/columbia` | Uncompressed splicing |
| Coverage | `data/raw/coverage` | Copy-move with similar regions |
| IMD2020 | `data/raw/imd2020` | Internet image manipulation |
| Synthetic AI | `data/raw/synthetic_ai` | Stable Diffusion, Midjourney, DALL-E, Flux |

---

## Training

### Train all models sequentially
```bash
python scripts/train_models.py --model all --config configs/config.yaml
```

### Train a single model
```bash
python scripts/train_models.py --model rgb --config configs/config.yaml
```

### With SimCLR contrastive pre-training
```bash
python scripts/train_models.py --model vit --pretrain
```

### Resume interrupted training
```bash
python scripts/train_models.py --model rgb --resume
```

### Train with cross-dataset validation
```bash
python scripts/train_models.py --model all --cross_validate
```

### Train fusion layer only (after all 7 models are trained)
```bash
python scripts/train_models.py --fusion_only
```

---

## Inference

### Single image
```bash
python scripts/run_inference.py \
    --image suspect.jpg \
    --checkpoint_dir outputs/checkpoints \
    --save_dir outputs/reports \
    --explain
```

### Batch directory
```bash
python scripts/run_inference.py \
    --batch_dir folder_of_images/ \
    --save_dir outputs/reports
```

### Python API
```python
from src.inference.pipeline import ForensicEngine

engine = ForensicEngine(checkpoint_dir="outputs/checkpoints")
engine.warmup()

result = engine.analyze_image("suspect.jpg", save_dir="outputs/reports")

print(result["status"])        # "MANIPULATED" or "AUTHENTIC"
print(result["risk_level"])    # "CRITICAL" / "HIGH" / "MEDIUM" / "LOW" / "MINIMAL"
print(result["confidence"])    # 0.0 – 1.0
print(result["report"])        # Full text report
```

### Convenience function
```python
from src.inference.pipeline import analyze_image

result = analyze_image("suspect.jpg")
```

### Output format
```json
{
  "status": "MANIPULATED",
  "confidence": 0.923,
  "deepfake_score": 0.941,
  "ai_generated_score": 0.887,
  "manipulation_score": 0.856,
  "frequency_score": 0.712,
  "noise_score": 0.634,
  "ela_score": 0.589,
  "metadata_score": 0.950,
  "risk_level": "CRITICAL",
  "dominant_manipulation": "face_swap",
  "dominant_source": "deepfake",
  "manipulated_fraction": 0.31,
  "metadata_flags": ["Software: Adobe Photoshop"],
  "findings": ["..."],
  "image_hash": "a1b2c3d4e5f6...",
  "heatmap": "<ndarray H×W×3>",
  "mask": "<ndarray H×W>",
  "report": "FORENSIC ANALYSIS REPORT\n...",
  "elapsed_seconds": 1.24
}
```

---

## Evaluation

Models are evaluated on:

- **Per-dataset**: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR, EER
- **Cross-dataset generalization**: train on FaceForensics++ → test on Celeb-DF, DFDC
- **Out-of-distribution**: never-seen generators (Flux, DALL-E 3, etc.)
- **Generator generalization**: performance grouped by GAN / diffusion / deepfake / face-swap

---

## Training Strategy

| Technique | Details |
|---|---|
| **Transfer Learning** | ImageNet-22k pre-trained backbones via `timm` |
| **Contrastive Pre-training** | SimCLR with NT-Xent loss, projection head |
| **Mixup / CutMix** | Alpha=0.4, applied at batch level |
| **Hard Negative Mining** | Online selection of top-k hardest negatives per batch |
| **Label Smoothing** | ε=0.1 for all classification heads |
| **EMA** | Decay=0.9998 for stable inference weights |
| **AMP** | FP16 mixed precision on all models |
| **Layer-wise LR Decay** | Backbone LR × 0.1, head LR × 1.0 |

---

## Explainability

GradCAM, GradCAM++, Integrated Gradients, and Attention Rollout visualizations are generated automatically with `--explain`:

```python
result = engine.analyze_image("suspect.jpg", explain=True)
# result["heatmap"] = H×W×3 BGR overlay ready for cv2.imwrite
```

---

## Hardware Requirements

| Configuration | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | 8 GB | 24–40 GB |
| **RAM** | 16 GB | 64 GB |
| **Storage** | 100 GB | 500 GB+ |
| **CUDA** | 11.8 | 12.1+ |

For multi-GPU training, set `CUDA_VISIBLE_DEVICES=0,1,2,3` and use `torch.nn.DataParallel` or enable DeepSpeed in config.

---

## License

MIT License. Research use only. Not for deployment in surveillance, law enforcement, or court proceedings without human expert review.
