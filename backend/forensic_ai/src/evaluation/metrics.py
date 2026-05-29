"""
Forensic AI Evaluation Engine.
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR.
Evaluations: Cross-dataset, OOD, Generator generalization.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


# ─────────────────────────────────────────────
#  Core Metrics
# ─────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute full forensic evaluation metrics.
    Args:
        labels: ground truth binary labels (0=real, 1=fake)
        probs:  predicted probabilities (0–1)
        threshold: decision threshold
    Returns:
        dict with all metrics
    """
    preds = (probs >= threshold).astype(int)

    # Basic metrics
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    # AUC
    try:
        roc_auc = roc_auc_score(labels, probs)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(labels, probs)
    except Exception:
        pr_auc = 0.0

    # FPR / FNR
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel() \
        if len(np.unique(labels)) == 2 else (0, 0, 0, 0)

    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)

    # EER (Equal Error Rate)
    fprs, tprs, thresholds = roc_curve(labels, probs) if len(np.unique(labels)) == 2 else \
        ([0, 1], [0, 1], [0.5])
    fnrs_curve = 1 - np.array(tprs)
    eer_idx = np.argmin(np.abs(np.array(fprs) - fnrs_curve))
    eer = (fprs[eer_idx] + fnrs_curve[eer_idx]) / 2

    return {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "roc_auc":   float(roc_auc),
        "pr_auc":    float(pr_auc),
        "fpr":       float(fpr),
        "fnr":       float(fnr),
        "eer":       float(eer),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def find_optimal_threshold(
    labels: np.ndarray, probs: np.ndarray, metric: str = "f1"
) -> Tuple[float, float]:
    """Find threshold that maximizes F1 or minimizes EER."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_val = -1

    for t in thresholds:
        preds = (probs >= t).astype(int)
        if metric == "f1":
            val = f1_score(labels, preds, zero_division=0)
        elif metric == "eer":
            fprs, tprs, _ = roc_curve(labels, probs)
            fnrs = 1 - tprs
            eer_idx = np.argmin(np.abs(fprs - fnrs))
            val = -abs(fprs[eer_idx] - fnrs[eer_idx])
        else:
            val = accuracy_score(labels, preds)

        if val > best_val:
            best_val = val
            best_thresh = t

    return best_thresh, best_val


# ─────────────────────────────────────────────
#  Model Evaluator
# ─────────────────────────────────────────────

class ForensicEvaluator:
    """
    Evaluates a single forensic model on a DataLoader.
    Returns full metrics + per-dataset breakdown.
    """

    def __init__(self, model: nn.Module, model_name: str, device: str = "cuda"):
        self.model      = model
        self.model_name = model_name
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Returns (labels, probs, dataset_names)."""
        all_labels  = []
        all_probs   = []
        all_datasets= []

        for batch in tqdm(loader, desc=f"Eval [{self.model_name}]"):
            label   = batch["label"].to(self.device)

            if self.model_name == "rgb":
                inp = batch["rgb"].to(self.device)
                prob, _, _ = self.model(inp)
            elif self.model_name == "vit":
                inp = batch["rgb"].to(self.device)
                prob, _, _ = self.model(inp)
            elif self.model_name == "frequency":
                inp = batch["freq"].to(self.device)
                prob, _ = self.model(inp)
            elif self.model_name == "noise":
                inp = batch["rgb"].to(self.device)
                prob, _, _ = self.model(inp)
            elif self.model_name == "ela":
                inp = batch["ela"].to(self.device)
                prob, _, _ = self.model(inp)
            elif self.model_name == "face":
                inp = batch["rgb"].to(self.device)
                prob, _, _, _ = self.model(inp)
            elif self.model_name == "localization":
                inp = batch["rgb"].to(self.device)
                _, prob = self.model(inp)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            all_labels.extend(label.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
            all_datasets.extend(batch.get("dataset", ["unknown"] * len(label)))

        return np.array(all_labels), np.array(all_probs), all_datasets

    def evaluate(self, loader: DataLoader, threshold: float = 0.5) -> Dict:
        labels, probs, datasets = self.predict(loader)
        overall = compute_metrics(labels, probs, threshold)
        opt_thresh, _ = find_optimal_threshold(labels, probs)
        overall["optimal_threshold"] = float(opt_thresh)

        # Per-dataset breakdown
        per_dataset = {}
        unique_ds = set(datasets)
        for ds in unique_ds:
            mask = np.array([d == ds for d in datasets])
            if mask.sum() < 10:
                continue
            per_dataset[ds] = compute_metrics(labels[mask], probs[mask], threshold)

        return {"overall": overall, "per_dataset": per_dataset,
                "labels": labels.tolist(), "probs": probs.tolist()}


# ─────────────────────────────────────────────
#  Cross-Dataset Evaluation
# ─────────────────────────────────────────────

class CrossDatasetEvaluator:
    """
    Tests generalization: train on dataset A, test on dataset B.
    Critical for measuring real-world robustness.
    """

    def __init__(self, evaluator: ForensicEvaluator):
        self.evaluator = evaluator

    def evaluate_all(
        self,
        test_loaders: Dict[str, DataLoader],
        threshold: float = 0.5,
    ) -> Dict[str, Dict]:
        results = {}
        for ds_name, loader in test_loaders.items():
            print(f"\n[CrossDataset] Testing on: {ds_name}")
            results[ds_name] = self.evaluator.evaluate(loader, threshold)
        return results

    def print_summary(self, results: Dict):
        print("\n" + "="*80)
        print("CROSS-DATASET EVALUATION SUMMARY")
        print("="*80)
        metrics_to_show = ["accuracy", "roc_auc", "pr_auc", "f1", "fpr", "fnr", "eer"]
        header = f"{'Dataset':<20}" + "".join(f"{m:>10}" for m in metrics_to_show)
        print(header)
        print("-"*80)
        for ds, result in results.items():
            ov = result["overall"]
            row = f"{ds:<20}" + "".join(f"{ov.get(m, 0)*100:>9.2f}%" for m in metrics_to_show)
            print(row)
        print("="*80)


# ─────────────────────────────────────────────
#  Generator Generalization Test
# ─────────────────────────────────────────────

class GeneratorGeneralizationEvaluator:
    """
    Tests whether model generalizes to unseen generation methods.
    Groups results by: Deepfake | GAN | Diffusion | Face Swap.
    """

    GENERATOR_GROUPS = {
        "deepfake_methods": ["deepfakes", "neuraltextures", "face2face", "faceswap", "faceshifter"],
        "gan_methods":      ["progan", "stylegan", "biggan", "cyclegan", "gaugan"],
        "diffusion_methods": ["stable_diffusion", "midjourney", "dalle", "flux", "imagen"],
        "face_swap_methods": ["faceswap", "deepfacelab", "simswap"],
    }

    def __init__(self, evaluator: ForensicEvaluator):
        self.evaluator = evaluator

    def evaluate(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
    ) -> Dict:
        labels, probs, _ = self.evaluator.predict(loader)

        # Get manipulation types from loader
        manip_types = []
        for batch in loader:
            manip_types.extend(batch.get("manipulation_type", ["unknown"] * len(batch["label"])))

        results = {}
        for group_name, methods in self.GENERATOR_GROUPS.items():
            mask = np.array([
                any(m in t.lower() for m in methods) or t.lower() in methods
                for t in manip_types
            ])
            if mask.sum() < 5:
                continue
            results[group_name] = compute_metrics(labels[mask], probs[mask], threshold)

        return results


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def plot_roc_curves(
    results_by_model: Dict[str, Dict],
    save_path: str,
    title: str = "ROC Curves",
):
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = cm.Set1(np.linspace(0, 1, len(results_by_model)))

    for (model_name, result), color in zip(results_by_model.items(), colors):
        labels = np.array(result["labels"])
        probs  = np.array(result["probs"])
        if len(np.unique(labels)) < 2:
            continue
        fprs, tprs, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fprs, tprs, color=color, lw=2,
                label=f"{model_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall_curves(
    results_by_model: Dict[str, Dict],
    save_path: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = cm.Set1(np.linspace(0, 1, len(results_by_model)))

    for (model_name, result), color in zip(results_by_model.items(), colors):
        labels = np.array(result["labels"])
        probs  = np.array(result["probs"])
        if len(np.unique(labels)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{model_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    labels: np.ndarray, probs: np.ndarray,
    threshold: float, save_path: str, title: str = "",
):
    preds = (probs >= threshold).astype(int)
    cm_arr = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_evaluation_report(
    results: Dict,
    save_path: str,
    model_name: str,
    dataset_name: str,
):
    """Save full evaluation report as JSON + text."""
    report = {
        "model": model_name,
        "dataset": dataset_name,
        "overall": results["overall"],
        "per_dataset": results.get("per_dataset", {}),
    }
    json_path = save_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(save_path, "w") as f:
        f.write(f"Forensic AI Evaluation Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model:   {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write("OVERALL METRICS\n")
        f.write(f"{'-'*40}\n")
        for k, v in results["overall"].items():
            if k in ["tp", "tn", "fp", "fn"]:
                f.write(f"  {k:20s}: {v}\n")
            elif isinstance(v, float):
                f.write(f"  {k:20s}: {v*100:.2f}%\n")
        f.write(f"\nPER-DATASET BREAKDOWN\n{'-'*40}\n")
        for ds, m in results.get("per_dataset", {}).items():
            f.write(f"\n  [{ds}]\n")
            f.write(f"    AUC={m.get('roc_auc', 0)*100:.2f}%  "
                    f"F1={m.get('f1', 0)*100:.2f}%  "
                    f"FPR={m.get('fpr', 0)*100:.2f}%\n")


# ─────────────────────────────────────────────
#  Evaluation Script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, argparse
    from src.datasets.dataset_pipeline import ForensicDatasetBuilder, build_dataloaders
    from src.training.trainer import build_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--model",    default="rgb",
                        choices=["rgb", "vit", "frequency", "noise", "ela", "face", "localization"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--save_dir", default="outputs/reports")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.save_dir, exist_ok=True)

    # Load test data
    splits_dir = cfg["paths"]["splits_dir"]
    test_samples = ForensicDatasetBuilder.load_manifest(f"{splits_dir}/test.csv")
    _, _, test_loader = build_dataloaders(
        test_samples, test_samples, test_samples,
        cfg=cfg, image_size=cfg["image"]["size"],
        batch_size=64, num_workers=4,
    )

    # Load model
    model = build_model(args.model, cfg, pretrained=False)
    state = torch.load(args.checkpoint, map_location=args.device)
    if "ema_state" in state:
        model.load_state_dict(state["ema_state"])
    else:
        model.load_state_dict(state["state_dict"])

    evaluator = ForensicEvaluator(model, args.model, device=args.device)
    results   = evaluator.evaluate(test_loader, threshold=cfg["evaluation"]["threshold"])

    # Save
    txt_path = os.path.join(args.save_dir, f"{args.model}_eval.txt")
    save_evaluation_report(results, txt_path, args.model, "test_set")

    # Plot
    plot_roc_curves({args.model: results}, os.path.join(args.save_dir, f"{args.model}_roc.png"))
    plot_precision_recall_curves({args.model: results},
                                  os.path.join(args.save_dir, f"{args.model}_pr.png"))
    plot_confusion_matrix(
        np.array(results["labels"]), np.array(results["probs"]),
        threshold=results["overall"]["optimal_threshold"],
        save_path=os.path.join(args.save_dir, f"{args.model}_cm.png"),
        title=f"{args.model} Confusion Matrix",
    )

    print(f"\n{'='*60}")
    print(f"[{args.model}] Test Results:")
    for k in ["accuracy", "roc_auc", "f1", "fpr", "fnr", "eer"]:
        print(f"  {k:15s}: {results['overall'].get(k, 0)*100:.2f}%")
