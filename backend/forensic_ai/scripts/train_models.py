#!/usr/bin/env python3
"""
train_models.py
---------------
CLI training script for all 7 forensic models.

Usage:
    python scripts/train_models.py --model rgb --config configs/config.yaml
    python scripts/train_models.py --model all --config configs/config.yaml
    python scripts/train_models.py --model rgb --pretrain       # SimCLR pre-train first
    python scripts/train_models.py --model all --cross_validate
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.dataset_pipeline import ForensicDatasetBuilder, build_dataloaders
from src.training.trainer import (
    ForensicTrainer, ContrastivePretrainer,
    build_model, build_criterion
)
from src.evaluation.metrics import ForensicEvaluator, CrossDatasetEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_models")

ALL_MODELS = ["rgb", "vit", "frequency", "noise", "ela", "face", "localization"]


# ──────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def model_config(cfg: dict, model_name: str) -> dict:
    """Extract per-model overrides merged with global training config."""
    base = cfg.get("training", {}).copy()
    override = cfg.get("models", {}).get(model_name, {})
    base.update(override)
    return base


# ──────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────

def prepare_datasets(cfg: dict, model_name: str):
    """Build or load cached dataset manifests and return dataloaders."""
    paths_cfg = cfg.get("paths", {})
    data_root  = paths_cfg.get("data_raw", "data/raw")
    split_dir  = paths_cfg.get("splits", "data/splits")
    split_file = Path(split_dir) / f"manifest_{model_name}.json"

    builder = ForensicDatasetBuilder(
        root_dir=data_root,
        datasets=cfg.get("datasets", {}).get("enabled", []),
    )

    if split_file.exists():
        logger.info(f"Loading cached manifest: {split_file}")
        builder.load_manifest(str(split_file))
    else:
        logger.info("Building dataset manifest ...")
        builder.build()
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        builder.save_manifest(str(split_file))

    mcfg = model_config(cfg, model_name)
    loaders = build_dataloaders(
        builder=builder,
        image_size=cfg["image"]["size"],
        ela_quality=cfg["image"]["ela_quality"],
        batch_size=mcfg.get("batch_size", 32),
        num_workers=mcfg.get("num_workers", 4),
        augment_train=True,
    )
    return loaders


# ──────────────────────────────────────────────
# Core train function
# ──────────────────────────────────────────────

def train_one_model(
    model_name: str,
    cfg: dict,
    pretrain: bool = False,
    resume: bool = False,
):
    logger.info(f"{'='*50}")
    logger.info(f"Training model: {model_name.upper()}")
    logger.info(f"{'='*50}")

    mcfg = model_config(cfg, model_name)
    paths_cfg = cfg.get("paths", {})
    ckpt_dir = paths_cfg.get("checkpoints", "outputs/checkpoints")

    # Build model and criterion
    model = build_model(model_name, mcfg)
    criterion = build_criterion(model_name, mcfg)

    # Optional contrastive pre-training
    if pretrain:
        logger.info(f"Running SimCLR pre-training for {model_name} ...")
        # Build unlabeled loader (use train split, labels ignored in contrastive)
        loaders = prepare_datasets(cfg, model_name)
        pretrain_loader = loaders["train"]

        pre = ContrastivePretrainer(
            model=model,
            dataloader=pretrain_loader,
            projection_dim=128,
            temperature=0.07,
            lr=mcfg.get("lr", 1e-4),
            epochs=mcfg.get("pretrain_epochs", 10),
            checkpoint_dir=ckpt_dir,
            model_name=model_name,
        )
        pre.run()
        logger.info("Pre-training complete. Loading pre-trained backbone.")

    # Build dataloaders
    loaders = prepare_datasets(cfg, model_name)

    # Build trainer
    trainer = ForensicTrainer(
        model=model,
        criterion=criterion,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=mcfg,
        model_name=model_name,
        checkpoint_dir=ckpt_dir,
        log_dir=paths_cfg.get("logs", "outputs/logs"),
    )

    # Resume if checkpoint exists
    if resume:
        ckpt_path = Path(ckpt_dir) / f"{model_name}_latest.pth"
        if ckpt_path.exists():
            trainer.load_checkpoint(str(ckpt_path))
            logger.info(f"Resumed from {ckpt_path}")

    # Train
    history = trainer.train(epochs=mcfg.get("epochs", 50))

    # Evaluate on test set
    logger.info(f"Evaluating {model_name} on test set ...")
    evaluator = ForensicEvaluator(
        model=trainer.get_best_model(),
        model_name=model_name,
        checkpoint_dir=ckpt_dir,
    )
    metrics = evaluator.evaluate(loaders["test"])
    logger.info(f"{model_name} test metrics: {json.dumps(metrics, indent=2)}")

    return history, metrics


# ──────────────────────────────────────────────
# Cross-dataset validation
# ──────────────────────────────────────────────

def run_cross_validation(model_name: str, cfg: dict):
    """Train on one dataset, evaluate on all others."""
    logger.info(f"Cross-dataset validation for {model_name}")

    paths_cfg = cfg.get("paths", {})
    ckpt_dir  = paths_cfg.get("checkpoints", "outputs/checkpoints")
    mcfg = model_config(cfg, model_name)

    # Load best trained model
    model = build_model(model_name, mcfg)
    ckpt_path = Path(ckpt_dir) / f"{model_name}_best.pth"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("ema_state_dict") or state.get("model_state_dict") or state
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded {model_name} from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found for {model_name}; using untrained weights")

    # Build per-dataset loaders
    dataset_names = cfg.get("datasets", {}).get("enabled", [])
    loaders_by_dataset = {}
    for ds_name in dataset_names:
        try:
            builder = ForensicDatasetBuilder(
                root_dir=cfg["paths"]["data_raw"],
                datasets=[ds_name],
            )
            builder.build()
            loaders = build_dataloaders(
                builder=builder,
                image_size=cfg["image"]["size"],
                ela_quality=cfg["image"]["ela_quality"],
                batch_size=mcfg.get("batch_size", 32),
                num_workers=4,
                augment_train=False,
            )
            loaders_by_dataset[ds_name] = loaders["test"]
        except Exception as e:
            logger.warning(f"Could not build loader for {ds_name}: {e}")

    evaluator = CrossDatasetEvaluator(
        model=model,
        model_name=model_name,
        checkpoint_dir=ckpt_dir,
    )
    results = evaluator.evaluate_all(loaders_by_dataset)
    logger.info(f"Cross-dataset results:\n{json.dumps(results, indent=2)}")
    return results


# ──────────────────────────────────────────────
# Fusion training
# ──────────────────────────────────────────────

def train_fusion(cfg: dict):
    """Collect model scores on val set, then train all 3 fusion strategies."""
    from src.fusion.fusion import (
        ModelScoreCollector, FusionEnsembleComparator
    )

    logger.info("Training fusion layer ...")
    paths_cfg = cfg.get("paths", {})
    ckpt_dir  = paths_cfg.get("checkpoints", "outputs/checkpoints")

    # Load all models
    from src.inference.pipeline import ModelRegistry
    registry = ModelRegistry(ckpt_dir)
    registry.warmup()

    # Load val dataloader
    mcfg = model_config(cfg, "rgb")
    loaders = prepare_datasets(cfg, "rgb")
    val_loader  = loaders["val"]
    test_loader = loaders["test"]

    # Collect scores
    collector = ModelScoreCollector(registry)
    train_scores, train_labels = collector.collect(val_loader)
    test_scores,  test_labels  = collector.collect(test_loader)

    # Train comparator
    comparator = FusionEnsembleComparator(checkpoint_dir=ckpt_dir)
    comparator.train(train_scores, train_labels)
    metrics = comparator.evaluate(test_scores, test_labels)
    logger.info(f"Fusion evaluation:\n{json.dumps(metrics, indent=2)}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Forensic AI — Model Trainer")
    p.add_argument("--model",          type=str, default="all",
                   help=f"Model to train: {ALL_MODELS + ['all', 'fusion']}")
    p.add_argument("--config",         type=str, default="configs/config.yaml",
                   help="Path to YAML config")
    p.add_argument("--pretrain",       action="store_true",
                   help="Run SimCLR contrastive pre-training first")
    p.add_argument("--resume",         action="store_true",
                   help="Resume from latest checkpoint")
    p.add_argument("--cross_validate", action="store_true",
                   help="Run cross-dataset validation after training")
    p.add_argument("--fusion_only",    action="store_true",
                   help="Skip model training; only train fusion layer")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)

    targets = ALL_MODELS if args.model == "all" else [args.model]

    if not args.fusion_only and args.model != "fusion":
        for model_name in targets:
            train_one_model(
                model_name,
                cfg,
                pretrain=args.pretrain,
                resume=args.resume,
            )
            if args.cross_validate:
                run_cross_validation(model_name, cfg)

    if args.model in ("all", "fusion") or args.fusion_only:
        train_fusion(cfg)

    logger.info("All training complete.")


if __name__ == "__main__":
    main()
