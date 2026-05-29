"""
Unified Dataset Pipeline for Forensic AI Engine.
Handles: FaceForensics++, Celeb-DF, DFDC, CASIA, Columbia,
         Coverage, IMD2020, NIST Nimble, Synthetic (MJ/SD/DALLE/Flux).
"""

import os
import cv2
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import piexif
from scipy.fft import fft2, fftshift
import pywt

# ─────────────────────────────────────────────
#  Data Record
# ─────────────────────────────────────────────

@dataclass
class ForensicSample:
    image_path: str
    label: int                         # 0 = real, 1 = fake/manipulated
    dataset: str
    manipulation_type: str = "none"    # deepfake | face_swap | ai_gen | splice | copymove | ...
    mask_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────
#  Dataset Loaders per source
# ─────────────────────────────────────────────

class FaceForensicsLoader:
    """
    FaceForensics++ directory structure:
    root/
      manipulated_sequences/{method}/{compression}/frames/{id}/*.png
      original_sequences/youtube/{compression}/frames/{id}/*.png
    """
    METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
    COMPRESSIONS = ["c0", "c23", "c40"]

    def __init__(self, root: str, compressions: List[str] = None):
        self.root = Path(root)
        self.compressions = compressions or self.COMPRESSIONS

    def collect(self) -> List[ForensicSample]:
        samples = []
        for comp in self.compressions:
            # Real
            real_base = self.root / "original_sequences" / "youtube" / comp / "frames"
            if real_base.exists():
                for vid_dir in real_base.iterdir():
                    for img in sorted(vid_dir.glob("*.png"))[:30]:  # cap per video
                        samples.append(ForensicSample(
                            image_path=str(img), label=0,
                            dataset="faceforensics", manipulation_type="none",
                            metadata={"compression": comp, "video_id": vid_dir.name}
                        ))
            # Fake
            for method in self.METHODS:
                fake_base = self.root / "manipulated_sequences" / method / comp / "frames"
                if fake_base.exists():
                    for vid_dir in fake_base.iterdir():
                        for img in sorted(vid_dir.glob("*.png"))[:30]:
                            samples.append(ForensicSample(
                                image_path=str(img), label=1,
                                dataset="faceforensics",
                                manipulation_type=method.lower(),
                                metadata={"compression": comp, "method": method, "video_id": vid_dir.name}
                            ))
        return samples


class CelebDFLoader:
    """
    Celeb-DF-v2 structure:
    root/
      Celeb-real/videos/*.mp4  (or extracted frames)
      Celeb-synthesis/videos/*.mp4
      YouTube-real/videos/*.mp4
    """
    def __init__(self, root: str):
        self.root = Path(root)

    def collect(self) -> List[ForensicSample]:
        samples = []
        for split, label in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
            frames_dir = self.root / split / "frames"
            if not frames_dir.exists():
                frames_dir = self.root / split / "images"
            if frames_dir.exists():
                for img in sorted(frames_dir.rglob("*.png"))[:5000]:
                    samples.append(ForensicSample(
                        image_path=str(img),
                        label=label,
                        dataset="celebdf",
                        manipulation_type="none" if label == 0 else "deepfake",
                        metadata={"split": split}
                    ))
        return samples


class DFDCLoader:
    """
    DFDC structure:
    root/
      train_part_XX/
        metadata.json
        *.mp4 (or extracted frames)
    """
    def __init__(self, root: str):
        self.root = Path(root)

    def collect(self) -> List[ForensicSample]:
        samples = []
        for part_dir in sorted(self.root.glob("train_part_*")):
            meta_file = part_dir / "metadata.json"
            if not meta_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            frames_dir = part_dir / "frames"
            for vid_name, info in meta.items():
                label = 0 if info["label"] == "REAL" else 1
                vid_frames = frames_dir / Path(vid_name).stem
                if vid_frames.exists():
                    for img in sorted(vid_frames.glob("*.jpg"))[:20]:
                        samples.append(ForensicSample(
                            image_path=str(img),
                            label=label,
                            dataset="dfdc",
                            manipulation_type="none" if label == 0 else "deepfake",
                            metadata={"original": info.get("original", "")}
                        ))
        return samples


class CASIALoader:
    """
    CASIA v2 structure:
    root/
      Au/   (authentic)
      Tp/   (tampered)
      CASIA 2 Groundtruth/  (binary masks)
    """
    def __init__(self, root: str):
        self.root = Path(root)

    def collect(self) -> List[ForensicSample]:
        samples = []
        auth_dir = self.root / "Au"
        if auth_dir.exists():
            for img in auth_dir.glob("*.*"):
                if img.suffix.lower() in [".jpg", ".png", ".bmp", ".tif"]:
                    samples.append(ForensicSample(
                        image_path=str(img), label=0,
                        dataset="casia", manipulation_type="none"
                    ))
        tp_dir = self.root / "Tp"
        mask_dir = self.root / "CASIA 2 Groundtruth"
        if tp_dir.exists():
            for img in tp_dir.glob("*.*"):
                if img.suffix.lower() in [".jpg", ".png", ".bmp", ".tif"]:
                    mask_path = None
                    if mask_dir.exists():
                        m = mask_dir / (img.stem + ".png")
                        if m.exists():
                            mask_path = str(m)
                    manip = self._parse_manipulation(img.stem)
                    samples.append(ForensicSample(
                        image_path=str(img), label=1,
                        dataset="casia", manipulation_type=manip,
                        mask_path=mask_path
                    ))
        return samples

    @staticmethod
    def _parse_manipulation(stem: str) -> str:
        s = stem.upper()
        if "CM" in s: return "copy_move"
        if "SP" in s: return "splice"
        if "RE" in s: return "removal"
        return "tampered"


class ColumbiaLoader:
    """
    Columbia Uncompressed Image Splicing Dataset:
    root/
      4cam_auth/  (authentic)
      4cam_splc/  (spliced)
    """
    def __init__(self, root: str):
        self.root = Path(root)

    def collect(self) -> List[ForensicSample]:
        samples = []
        for d, label, mtype in [("4cam_auth", 0, "none"), ("4cam_splc", 1, "splice")]:
            d_path = self.root / d
            if d_path.exists():
                for img in d_path.glob("*.tif"):
                    samples.append(ForensicSample(
                        image_path=str(img), label=label,
                        dataset="columbia", manipulation_type=mtype
                    ))
        return samples


class CoverageLoader:
    """Coverage dataset for copy-move forgery."""
    def __init__(self, root: str):
        self.root = Path(root)

    def collect(self) -> List[ForensicSample]:
        samples = []
        img_dir = self.root / "image"
        mask_dir = self.root / "mask"
        if not img_dir.exists():
            return samples
        for img in img_dir.glob("*.tif"):
            label = 0 if img.stem.endswith("t") else 1
            mask_path = None
            if label == 1 and mask_dir.exists():
                m = mask_dir / (img.stem + "forged.tif")
                if m.exists():
                    mask_path = str(m)
            samples.append(ForensicSample(
                image_path=str(img), label=label,
                dataset="coverage",
                manipulation_type="none" if label == 0 else "copy_move",
                mask_path=mask_path
            ))
        return samples


class SyntheticAILoader:
    """
    Generic loader for synthetic AI datasets (Midjourney, SD, DALLE, Flux).
    Expects:
      root/
        real/
        fake/ (or ai_generated/)
    """
    def __init__(self, root: str, source: str, manipulation_type: str = "ai_generated"):
        self.root = Path(root)
        self.source = source
        self.manipulation_type = manipulation_type

    def collect(self) -> List[ForensicSample]:
        samples = []
        for folder, label in [("real", 0), ("fake", 1), ("ai_generated", 1),
                               ("authentic", 0), ("synthetic", 1)]:
            d = self.root / folder
            if d.exists():
                for img in d.rglob("*.*"):
                    if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        samples.append(ForensicSample(
                            image_path=str(img), label=label,
                            dataset=self.source,
                            manipulation_type=self.manipulation_type if label == 1 else "none",
                            metadata={"source": self.source}
                        ))
        return samples


class IMD2020Loader:
    """IMD2020 splicing detection dataset."""
    def __init__(self, root: str):
        self.root = Path(root)

    def collect(self) -> List[ForensicSample]:
        samples = []
        for label_dir, label in [("authentic", 0), ("tampered", 1)]:
            d = self.root / label_dir
            if d.exists():
                for img in d.rglob("*.jpg"):
                    mask_path = None
                    mask = d.parent / "masks" / img.relative_to(d).with_suffix(".png")
                    if mask.exists():
                        mask_path = str(mask)
                    samples.append(ForensicSample(
                        image_path=str(img), label=label,
                        dataset="imd2020",
                        manipulation_type="none" if label == 0 else "splice",
                        mask_path=mask_path
                    ))
        return samples


# ─────────────────────────────────────────────
#  Unified Dataset Builder
# ─────────────────────────────────────────────

class ForensicDatasetBuilder:
    """Aggregates all dataset loaders into one unified manifest."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.loaders = []

    def register(self, loader) -> "ForensicDatasetBuilder":
        self.loaders.append(loader)
        return self

    def build_from_config(self, data_cfg: dict) -> List[ForensicSample]:
        ds_cfg = data_cfg.get("datasets", {})
        if "faceforensics" in ds_cfg:
            self.register(FaceForensicsLoader(
                ds_cfg["faceforensics"]["path"],
                ds_cfg["faceforensics"].get("compression", ["c23"])
            ))
        if "celebdf" in ds_cfg:
            self.register(CelebDFLoader(ds_cfg["celebdf"]["path"]))
        if "dfdc" in ds_cfg:
            self.register(DFDCLoader(ds_cfg["dfdc"]["path"]))
        if "casia" in ds_cfg:
            self.register(CASIALoader(ds_cfg["casia"]["path"]))
        if "columbia" in ds_cfg:
            self.register(ColumbiaLoader(ds_cfg["columbia"]["path"]))
        if "coverage" in ds_cfg:
            self.register(CoverageLoader(ds_cfg["coverage"]["path"]))
        if "imd2020" in ds_cfg:
            self.register(IMD2020Loader(ds_cfg["imd2020"]["path"]))
        synth = ds_cfg.get("synthetic", {})
        for name in ["midjourney", "stable_diffusion", "dalle", "flux"]:
            if name in synth:
                self.register(SyntheticAILoader(synth[name], name))
        return self.collect_all()

    def collect_all(self) -> List[ForensicSample]:
        all_samples = []
        for loader in self.loaders:
            samples = loader.collect()
            print(f"[{loader.__class__.__name__}] Collected {len(samples)} samples")
            all_samples.append(samples)
        flat = [s for group in all_samples for s in group]
        print(f"\n[DatasetBuilder] Total samples: {len(flat)}")
        print(f"  Real: {sum(1 for s in flat if s.label == 0)}")
        print(f"  Fake: {sum(1 for s in flat if s.label == 1)}")
        return flat

    @staticmethod
    def split(samples: List[ForensicSample],
              ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
              seed: int = 42) -> Tuple[List, List, List]:
        """Stratified split by label."""
        rng = np.random.default_rng(seed)
        real = [s for s in samples if s.label == 0]
        fake = [s for s in samples if s.label == 1]
        splits = []
        for subset in [real, fake]:
            idx = rng.permutation(len(subset))
            n_train = int(len(subset) * ratios[0])
            n_val   = int(len(subset) * ratios[1])
            splits.append((
                [subset[i] for i in idx[:n_train]],
                [subset[i] for i in idx[n_train:n_train+n_val]],
                [subset[i] for i in idx[n_train+n_val:]]
            ))
        train = splits[0][0] + splits[1][0]
        val   = splits[0][1] + splits[1][1]
        test  = splits[0][2] + splits[1][2]
        rng.shuffle(train)
        return train, val, test

    @staticmethod
    def save_manifest(samples: List[ForensicSample], path: str):
        records = [
            {"image_path": s.image_path, "label": s.label,
             "dataset": s.dataset, "manipulation_type": s.manipulation_type,
             "mask_path": s.mask_path, "metadata": s.metadata}
            for s in samples
        ]
        pd.DataFrame(records).to_csv(path, index=False)
        print(f"Saved manifest: {path} ({len(records)} records)")

    @staticmethod
    def load_manifest(path: str) -> List[ForensicSample]:
        df = pd.read_csv(path)
        samples = []
        for _, row in df.iterrows():
            meta = json.loads(row["metadata"]) if isinstance(row.get("metadata"), str) else {}
            samples.append(ForensicSample(
                image_path=row["image_path"], label=int(row["label"]),
                dataset=row["dataset"], manipulation_type=row["manipulation_type"],
                mask_path=row.get("mask_path") if pd.notna(row.get("mask_path")) else None,
                metadata=meta
            ))
        return samples


# ─────────────────────────────────────────────
#  ELA Utility
# ─────────────────────────────────────────────

def compute_ela(image_path: str, quality: int = 90, scale: float = 10.0) -> np.ndarray:
    """Error Level Analysis map."""
    import io
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    ela = np.abs(np.array(img, dtype=np.float32) - np.array(recompressed, dtype=np.float32))
    ela = np.clip(ela * scale, 0, 255).astype(np.uint8)
    return ela


def compute_fft_map(image: np.ndarray) -> np.ndarray:
    """FFT magnitude spectrum (log-scaled, 3-channel)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    return np.stack([magnitude] * 3, axis=-1)


def compute_dct_map(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Block DCT coefficient map."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    out = np.zeros_like(gray)
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            dct = cv2.dct(block)
            out[i:i+block_size, j:j+block_size] = np.abs(dct)
    out = np.log1p(out)
    out = (out / out.max() * 255).astype(np.uint8) if out.max() > 0 else out.astype(np.uint8)
    return np.stack([out] * 3, axis=-1)


def compute_wavelet_map(image: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """Wavelet detail coefficients map."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    _, (cH, cV, cD) = pywt.dwt2(gray, wavelet)
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    detail = (detail / (detail.max() + 1e-8) * 255).astype(np.uint8)
    detail_resized = cv2.resize(detail, (image.shape[1], image.shape[0]))
    return np.stack([detail_resized] * 3, axis=-1)


def compute_srm_residual(image: np.ndarray) -> np.ndarray:
    """
    SRM (Steganalysis Rich Model) noise residual.
    Uses a set of high-pass filters to extract noise residuals.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # 3x3 and 5x5 high-pass filters
    kernel_3 = np.array([[-1, 2, -1],
                          [2, -4, 2],
                          [-1, 2, -1]], dtype=np.float32) / 4
    kernel_5 = np.array([[0, 0, 0, 0, 0],
                          [0,-1, 2,-1, 0],
                          [0, 2,-4, 2, 0],
                          [0,-1, 2,-1, 0],
                          [0, 0, 0, 0, 0]], dtype=np.float32) / 4
    r3 = cv2.filter2D(gray, -1, kernel_3)
    r5 = cv2.filter2D(gray, -1, kernel_5)
    # Combine channels
    diff = image.astype(np.float32)
    r_rgb = np.stack([r3, r5, r3 - r5], axis=-1)
    r_rgb = np.clip((r_rgb + 128), 0, 255).astype(np.uint8)
    return r_rgb


def extract_metadata(image_path: str) -> Dict:
    """Extract EXIF and metadata features."""
    meta = {
        "has_exif": False,
        "make": None, "model": None,
        "software": None, "datetime": None,
        "gps": False, "thumbnail": False,
        "modified_datetime": None,
        "exif_consistent": True,
        "file_size": 0,
        "format": None,
    }
    try:
        meta["file_size"] = os.path.getsize(image_path)
        img = Image.open(image_path)
        meta["format"] = img.format
        raw_exif = img.info.get("exif", b"")
        if raw_exif:
            meta["has_exif"] = True
            exif = piexif.load(raw_exif)
            ifd0 = exif.get("0th", {})
            meta["make"]     = ifd0.get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore").strip("\x00")
            meta["model"]    = ifd0.get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore").strip("\x00")
            meta["software"] = ifd0.get(piexif.ImageIFD.Software, b"").decode("utf-8", errors="ignore").strip("\x00")
            gps_ifd = exif.get("GPS", {})
            meta["gps"] = len(gps_ifd) > 0
            meta["thumbnail"] = exif.get("thumbnail") is not None
            # Check for common inconsistencies
            software = meta["software"].lower()
            if any(x in software for x in ["photoshop", "gimp", "lightroom", "affinity",
                                            "stable diffusion", "midjourney", "dall-e"]):
                meta["exif_consistent"] = False
    except Exception:
        pass
    return meta


def metadata_to_vector(meta: Dict) -> np.ndarray:
    """Convert metadata dict to fixed-length feature vector."""
    vec = np.zeros(10, dtype=np.float32)
    vec[0] = float(meta.get("has_exif", False))
    vec[1] = float(meta.get("gps", False))
    vec[2] = float(meta.get("thumbnail", False))
    vec[3] = float(not meta.get("exif_consistent", True))
    make = (meta.get("make") or "").lower()
    model = (meta.get("model") or "").lower()
    software = (meta.get("software") or "").lower()
    vec[4] = float(any(x in software for x in ["photoshop", "gimp", "lightroom"]))
    vec[5] = float(any(x in software for x in ["stable diffusion", "midjourney", "dalle", "flux"]))
    vec[6] = float(meta.get("file_size", 0) == 0)
    vec[7] = float(meta.get("format") in [None, "PNG"])  # PNG = no JPEG compression
    vec[8] = float(len(make) == 0 and meta.get("has_exif", False))  # missing make with exif
    vec[9] = float(len(model) == 0 and meta.get("has_exif", False))
    return vec


# ─────────────────────────────────────────────
#  Core PyTorch Dataset
# ─────────────────────────────────────────────

class ForensicDataset(Dataset):
    """
    Main dataset returning:
      - RGB image tensor
      - ELA tensor
      - Frequency (FFT/DCT/Wavelet) tensor
      - Noise residual tensor
      - Metadata vector
      - Segmentation mask (if available)
      - Label
    """

    def __init__(
        self,
        samples: List[ForensicSample],
        image_size: int = 224,
        augment: bool = True,
        ela_quality: int = 90,
        mode: str = "train",
    ):
        self.samples = [s for s in samples if os.path.exists(s.image_path)]
        print(f"[ForensicDataset] {mode}: {len(self.samples)} valid samples "
              f"(dropped {len(samples) - len(self.samples)} missing files)")
        self.image_size = image_size
        self.augment = augment
        self.ela_quality = ela_quality
        self.mode = mode
        self._build_transforms()

    def _build_transforms(self):
        spatial = [
            A.Resize(self.image_size, self.image_size),
        ]
        if self.augment and self.mode == "train":
            spatial += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15, p=0.4),
                A.OneOf([
                    A.ImageCompression(quality_lower=60, quality_upper=95, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                ], p=0.4),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.05, p=0.3),
                A.RandomGamma(p=0.2),
            ]
        self.spatial_aug = A.Compose(spatial)
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        # Load RGB
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            img_bgr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ELA (computed before spatial augmentation to preserve JPEG artifacts)
        ela_map = compute_ela(sample.image_path, quality=self.ela_quality)

        # Spatial augmentation (same transform applied to RGB and ELA)
        aug = self.spatial_aug(image=img_rgb, mask=ela_map)
        img_aug  = aug["image"]       # H x W x 3
        ela_aug  = aug["mask"]        # H x W x 3

        # Frequency and noise (on augmented RGB)
        fft_map  = compute_fft_map(img_aug)
        dct_map  = compute_dct_map(img_aug)
        wav_map  = compute_wavelet_map(img_aug)
        srm_map  = compute_srm_residual(img_aug)

        # Normalize and convert
        rgb_tensor  = self.normalize(image=img_aug)["image"]
        ela_tensor  = self.normalize(image=ela_aug)["image"]
        fft_tensor  = self.normalize(image=fft_map)["image"]
        dct_tensor  = self.normalize(image=dct_map)["image"]
        wav_tensor  = self.normalize(image=wav_map)["image"]
        srm_tensor  = self.normalize(image=srm_map)["image"]

        # Frequency combined: stack FFT, DCT, Wavelet into 9-channel
        freq_tensor = torch.cat([fft_tensor, dct_tensor, wav_tensor], dim=0)  # 9xHxW

        # Metadata
        meta_vec = torch.tensor(
            metadata_to_vector(extract_metadata(sample.image_path)),
            dtype=torch.float32
        )

        # Segmentation mask
        if sample.mask_path and os.path.exists(sample.mask_path):
            mask_img = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (self.image_size, self.image_size))
            mask_tensor = torch.from_numpy((mask_img > 127).astype(np.float32)).unsqueeze(0)
        else:
            mask_tensor = torch.zeros(1, self.image_size, self.image_size)

        return {
            "rgb":      rgb_tensor,            # 3xHxW
            "ela":      ela_tensor,            # 3xHxW
            "freq":     freq_tensor,           # 9xHxW
            "srm":      srm_tensor,            # 3xHxW
            "metadata": meta_vec,              # 10
            "mask":     mask_tensor,           # 1xHxW
            "label":    torch.tensor(sample.label, dtype=torch.long),
            "image_path": sample.image_path,
            "manipulation_type": sample.manipulation_type,
            "dataset":  sample.dataset,
        }


def build_weighted_sampler(samples: List[ForensicSample]) -> WeightedRandomSampler:
    """Class-balanced sampler."""
    labels = np.array([s.label for s in samples])
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor([class_weights[l] for l in labels], dtype=torch.float32)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def build_dataloaders(
    train_samples: List[ForensicSample],
    val_samples: List[ForensicSample],
    test_samples: List[ForensicSample],
    cfg: dict,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ForensicDataset(train_samples, image_size=image_size, augment=True, mode="train")
    val_ds   = ForensicDataset(val_samples,   image_size=image_size, augment=False, mode="val")
    test_ds  = ForensicDataset(test_samples,  image_size=image_size, augment=False, mode="test")
    sampler = build_weighted_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  Frame Extraction Utility (video → frames)
# ─────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str,
                   max_frames: int = 30, interval: int = 1) -> List[str]:
    """Extract frames from video at given interval."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved = []
    idx = 0
    while cap.isOpened() and len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(path)
            frame_count += 1
        idx += 1
    cap.release()
    return saved


def batch_extract_frames(video_dir: str, output_root: str,
                         max_frames: int = 30, workers: int = 8):
    """Parallel frame extraction for a directory of videos."""
    video_paths = list(Path(video_dir).rglob("*.mp4")) + list(Path(video_dir).rglob("*.avi"))
    print(f"Extracting frames from {len(video_paths)} videos...")
    def _extract_one(vp):
        out_dir = os.path.join(output_root, Path(vp).stem)
        return extract_frames(str(vp), out_dir, max_frames=max_frames)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, vp): vp for vp in video_paths}
        total = 0
        for fut in as_completed(futures):
            total += len(fut.result())
    print(f"Extracted {total} total frames.")


# ─────────────────────────────────────────────
#  Dataset Preparation Script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--extract_frames", action="store_true")
    parser.add_argument("--build_manifest", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.extract_frames:
        for ds_name in ["faceforensics", "celebdf", "dfdc"]:
            ds_cfg = cfg["datasets"].get(ds_name, {})
            ds_path = ds_cfg.get("path", "")
            if os.path.exists(ds_path):
                print(f"\nExtracting frames for {ds_name}...")
                batch_extract_frames(
                    ds_path,
                    os.path.join(ds_path, "frames"),
                    max_frames=30
                )

    if args.build_manifest:
        builder = ForensicDatasetBuilder(cfg)
        all_samples = builder.build_from_config(cfg)
        train, val, test = ForensicDatasetBuilder.split(all_samples, seed=cfg["project"]["seed"])

        splits_dir = cfg["paths"]["splits_dir"]
        os.makedirs(splits_dir, exist_ok=True)
        ForensicDatasetBuilder.save_manifest(train, os.path.join(splits_dir, "train.csv"))
        ForensicDatasetBuilder.save_manifest(val,   os.path.join(splits_dir, "val.csv"))
        ForensicDatasetBuilder.save_manifest(test,  os.path.join(splits_dir, "test.csv"))
        print("\nManifests saved.")
