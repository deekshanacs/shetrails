"""
Model Fusion Engine for Forensic AI.
Combines scores from all 7 forensic models.
Methods: XGBoost, LightGBM, Neural Meta-Classifier.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


# ─────────────────────────────────────────────
#  Fusion Feature Schema
# ─────────────────────────────────────────────

SCORE_KEYS = [
    "rgb_score",
    "vit_score",
    "frequency_score",
    "noise_score",
    "ela_score",
    "face_score",
    "localization_score",
    "metadata_score",
]

N_SCORES = len(SCORE_KEYS)


def build_fusion_features(scores: Dict[str, float]) -> np.ndarray:
    """
    Convert score dict to feature vector.
    Also adds interaction features.
    """
    base = np.array([scores.get(k, 0.5) for k in SCORE_KEYS], dtype=np.float32)

    # Pairwise interactions (max, min, diff)
    interactions = []
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            interactions.append(base[i] * base[j])  # product
            interactions.append(abs(base[i] - base[j]))  # diff

    # Statistics
    stats = [
        base.mean(),
        base.std(),
        base.max(),
        base.min(),
        base.max() - base.min(),
        np.median(base),
        (base > 0.5).sum() / len(base),   # fraction of models saying fake
        (base > 0.7).sum() / len(base),
    ]

    return np.concatenate([base, interactions, stats]).astype(np.float32)


# ─────────────────────────────────────────────
#  Fusion Dataset
# ─────────────────────────────────────────────

class FusionDataset:
    """
    Dataset for training the fusion model.
    Collects predictions from all forensic models.
    """

    def __init__(self):
        self.features = []
        self.labels   = []
        self.metadata = []

    def add(self, scores: Dict[str, float], label: int, meta: dict = None):
        feat = build_fusion_features(scores)
        self.features.append(feat)
        self.labels.append(label)
        self.metadata.append(meta or {})

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.stack(self.features), np.array(self.labels)

    def save(self, path: str):
        X, y = self.to_numpy()
        np.savez(path, features=X, labels=y)

    @classmethod
    def load(cls, path: str) -> "FusionDataset":
        data = np.load(path, allow_pickle=True)
        ds = cls()
        ds.features = list(data["features"])
        ds.labels   = list(data["labels"])
        return ds


# ─────────────────────────────────────────────
#  XGBoost Fusion
# ─────────────────────────────────────────────

class XGBoostFusion:
    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self.params = {
            "n_estimators":     500,
            "max_depth":        6,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric":      "auc",
            "tree_method":      "gpu_hist",
            "random_state":     42,
            **kwargs
        }
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        self.model = self.xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=50,
        )
        print(f"[XGBoost] Best iteration: {self.model.best_iteration}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        imp = self.model.feature_importances_
        return {f"feat_{i}": float(v) for i, v in enumerate(imp)}


# ─────────────────────────────────────────────
#  LightGBM Fusion
# ─────────────────────────────────────────────

class LightGBMFusion:
    def __init__(self, **kwargs):
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        self.params = {
            "n_estimators":    500,
            "max_depth":       8,
            "learning_rate":   0.05,
            "num_leaves":      63,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "objective":       "binary",
            "metric":          "auc",
            "device":          "gpu",
            "verbose":         -1,
            "random_state":    42,
            **kwargs
        }
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        callbacks = [
            self.lgb.early_stopping(30, verbose=True),
            self.lgb.log_evaluation(50),
        ]
        self.model = self.lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)


# ─────────────────────────────────────────────
#  Neural Meta-Classifier
# ─────────────────────────────────────────────

class NeuralMetaClassifier(nn.Module):
    """
    MLP that fuses all model scores into a final verdict.
    Input: N_SCORES + interaction features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

        # Attention over individual scores (for interpretability)
        self.score_attn = nn.Sequential(
            nn.Linear(N_SCORES, N_SCORES),
            nn.Softmax(dim=-1),
        )

        # Per-model reliability weights (learned)
        self.reliability = nn.Parameter(torch.ones(N_SCORES))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: B x input_dim
        Returns: (prob: B, attn_weights: B x N_SCORES)
        """
        # Attended individual scores
        scores = x[:, :N_SCORES]
        rel    = F.softmax(self.reliability, dim=0)
        weighted_scores = scores * rel

        # Attention weights (for explainability)
        attn   = self.score_attn(weighted_scores)     # B x N_SCORES

        embed  = self.mlp(x)
        logit  = self.head(embed)
        prob   = torch.sigmoid(logit).squeeze(1)

        return prob, attn


class NeuralFusion:
    """Trainer for the Neural Meta-Classifier."""

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        device: str = "cuda",
        lr: float = 1e-3,
    ):
        input_dim = len(build_fusion_features({k: 0.5 for k in SCORE_KEYS}))
        self.model  = NeuralMetaClassifier(input_dim, list(hidden_dims), dropout)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.opt    = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = GradScaler()

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(X).float().to(self.device)

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
        epochs: int = 50, batch_size: int = 512,
    ):
        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(
            self._to_tensor(X_train),
            torch.from_numpy(y_train).long().to(self.device)
        )
        val_ds = TensorDataset(
            self._to_tensor(X_val),
            torch.from_numpy(y_val).long().to(self.device)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=1024, shuffle=False)

        best_auc = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            for X_b, y_b in train_loader:
                self.opt.zero_grad()
                with autocast():
                    prob, _ = self.model(X_b)
                    loss = F.binary_cross_entropy(prob, y_b.float())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                total_loss += loss.item()

            # Validate
            self.model.eval()
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    prob, _ = self.model(X_b)
                    all_probs.extend(prob.cpu().numpy())
                    all_labels.extend(y_b.cpu().numpy())

            metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
            auc = metrics["roc_auc"]

            if epoch % 10 == 0 or auc > best_auc:
                print(f"  Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
                      f"Val AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state:
            self.model.load_state_dict(best_state)
        print(f"[NeuralFusion] Best Val AUC: {best_auc:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = self._to_tensor(X)
            prob, _ = self.model(t)
        return prob.cpu().numpy()

    def get_attention(self, X: np.ndarray) -> np.ndarray:
        """Returns attention weights over individual models: N x N_SCORES"""
        self.model.eval()
        with torch.no_grad():
            t = self._to_tensor(X)
            _, attn = self.model(t)
        return attn.cpu().numpy()

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": {
                "input_dim": list(self.model.mlp.children())[0].in_features if hasattr(self.model.mlp, "children") else 0,
            }
        }, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])


# ─────────────────────────────────────────────
#  Ensemble Comparator
# ─────────────────────────────────────────────

class FusionEnsembleComparator:
    """
    Trains and compares XGBoost, LightGBM, Neural fusion.
    Selects best-performing method.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
        self.best_method = None
        self.models = {}

    def fit_all(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ):
        print("\n[Fusion] Training XGBoost...")
        try:
            xgb = XGBoostFusion()
            xgb.fit(X_train, y_train, X_val, y_val)
            probs = xgb.predict_proba(X_val)
            self.results["xgboost"] = compute_metrics(y_val, probs)
            self.models["xgboost"] = xgb
            print(f"  XGBoost Val AUC: {self.results['xgboost']['roc_auc']:.4f}")
        except Exception as e:
            print(f"  XGBoost failed: {e}")

        print("\n[Fusion] Training LightGBM...")
        try:
            lgb = LightGBMFusion()
            lgb.fit(X_train, y_train, X_val, y_val)
            probs = lgb.predict_proba(X_val)
            self.results["lightgbm"] = compute_metrics(y_val, probs)
            self.models["lightgbm"] = lgb
            print(f"  LightGBM Val AUC: {self.results['lightgbm']['roc_auc']:.4f}")
        except Exception as e:
            print(f"  LightGBM failed: {e}")

        print("\n[Fusion] Training Neural Meta-Classifier...")
        neural = NeuralFusion(device=self.device)
        neural.fit(X_train, y_train, X_val, y_val, epochs=50)
        probs = neural.predict_proba(X_val)
        self.results["neural"] = compute_metrics(y_val, probs)
        self.models["neural"] = neural
        print(f"  Neural Val AUC: {self.results['neural']['roc_auc']:.4f}")

        # Select best
        self.best_method = max(
            self.results, key=lambda k: self.results[k]["roc_auc"]
        )
        print(f"\n[Fusion] Best method: {self.best_method} "
              f"(AUC={self.results[self.best_method]['roc_auc']:.4f})")

    def predict(self, X: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        method = method or self.best_method
        return self.models[method].predict_proba(X)

    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Average predictions from all available methods."""
        all_probs = []
        for name, model in self.models.items():
            all_probs.append(model.predict_proba(X))
        return np.stack(all_probs).mean(axis=0)

    def save_all(self, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            ext = ".pt" if name == "neural" else ".pkl"
            model.save(str(Path(save_dir) / f"fusion_{name}{ext}"))
        with open(str(Path(save_dir) / "fusion_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        with open(str(Path(save_dir) / "best_method.txt"), "w") as f:
            f.write(self.best_method or "neural")
        print(f"[Fusion] All models saved to {save_dir}")

    def print_comparison(self):
        print("\n" + "="*70)
        print("FUSION MODEL COMPARISON")
        print("="*70)
        metrics = ["roc_auc", "f1", "accuracy", "fpr", "fnr"]
        header = f"{'Method':<15}" + "".join(f"{m:>12}" for m in metrics)
        print(header)
        print("-"*70)
        for method, res in self.results.items():
            marker = " ← BEST" if method == self.best_method else ""
            row = f"{method:<15}" + "".join(f"{res.get(m,0)*100:>11.2f}%" for m in metrics)
            print(row + marker)
        print("="*70)


# ─────────────────────────────────────────────
#  Score Collector (runs all models on a dataset)
# ─────────────────────────────────────────────

class ModelScoreCollector:
    """
    Runs all 7 forensic models on a DataLoader and collects scores.
    Used to build the fusion training set.
    """

    def __init__(self, models: Dict[str, nn.Module], device: str = "cuda"):
        self.models = models
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        for m in models.values():
            m.to(self.device)
            m.eval()

    @torch.no_grad()
    def collect(self, loader: DataLoader) -> FusionDataset:
        fusion_ds = FusionDataset()

        for batch in tqdm(loader, desc="Collecting model scores"):
            label = batch["label"].numpy()
            B = len(label)

            # Default all scores to 0.5
            batch_scores = [{k: 0.5 for k in SCORE_KEYS} for _ in range(B)]

            # RGB
            if "rgb" in self.models:
                inp  = batch["rgb"].to(self.device)
                prob, _, _ = self.models["rgb"](inp)
                for i, p in enumerate(prob.cpu().numpy()):
                    batch_scores[i]["rgb_score"] = float(p)

            # ViT
            if "vit" in self.models:
                inp  = batch["rgb"].to(self.device)
                prob, _, _ = self.models["vit"](inp)
                for i, p in enumerate(prob.cpu().numpy()):
                    batch_scores[i]["vit_score"] = float(p)

            # Frequency
            if "frequency" in self.models:
                inp  = batch["freq"].to(self.device)
                prob, _ = self.models["frequency"](inp)
                for i, p in enumerate(prob.cpu().numpy()):
                    batch_scores[i]["frequency_score"] = float(p)

            # Noise
            if "noise" in self.models:
                inp  = batch["rgb"].to(self.device)
                prob, _, _ = self.models["noise"](inp)
                for i, p in enumerate(prob.cpu().numpy()):
                    batch_scores[i]["noise_score"] = float(p)

            # ELA
            if "ela" in self.models:
                inp  = batch["ela"].to(self.device)
                prob, _, _ = self.models["ela"](inp)
                for i, p in enumerate(prob.cpu().numpy()):
                    batch_scores[i]["ela_score"] = float(p)

            # Face
            if "face" in self.models:
                inp  = batch["rgb"].to(self.device)
                prob, _, _, _ = self.models["face"](inp)
                for i, p in enumerate(prob.cpu().numpy()):
                    batch_scores[i]["face_score"] = float(p)

            # Localization
            if "localization" in self.models:
                inp  = batch["rgb"].to(self.device)
                _, img_prob = self.models["localization"](inp)
                for i, p in enumerate(img_prob.cpu().numpy()):
                    batch_scores[i]["localization_score"] = float(p)

            # Metadata score (precomputed)
            meta_vecs = batch["metadata"].numpy()  # B x 10
            for i in range(B):
                meta_score = float(meta_vecs[i][3])  # exif inconsistency flag
                meta_score = max(meta_score, float(meta_vecs[i][5]))  # AI software
                batch_scores[i]["metadata_score"] = meta_score

            for i in range(B):
                fusion_ds.add(batch_scores[i], int(label[i]))

        return fusion_ds


if __name__ == "__main__":
    # Demo: train fusion on dummy data
    np.random.seed(42)
    N = 2000
    X = np.random.rand(N, len(build_fusion_features({k: 0.5 for k in SCORE_KEYS}))).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    comparator = FusionEnsembleComparator(device="cpu")
    comparator.fit_all(X_train, y_train, X_val, y_val)
    comparator.print_comparison()
