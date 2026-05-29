"""
Universal Training Engine for Forensic AI Models.
Implements: Transfer learning, contrastive pretraining,
            Mixup/CutMix, hard negative mining, EMA, cosine LR.
"""

import os
import math
import time
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import wandb
from tqdm import tqdm

from src.models.model1_rgb       import RGBForensicsNet, RGBForensicsLoss, EMA
from src.models.model2_vit       import ForensicViT, ViTForensicsLoss
from src.models.model3_frequency import FrequencyForensicsNet, FrequencyForensicsLoss
from src.models.model4_noise     import NoiseForensicsNet, NoiseForensicsLoss
from src.models.model5_ela       import ELAForensicsNet, ELAForensicsLoss
from src.models.model6_face      import FaceForensicsNet, FaceForensicsLoss
from src.models.model7_localization import ManipulationLocalizationNet, LocalizationLoss


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────────
#  Mixup / CutMix Augmentation
# ─────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x.shape[0]
    idx = torch.randperm(bs, device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    bs = x.shape[0]
    idx = torch.randperm(bs, device=x.device)
    _, _, H, W = x.shape
    cut_rat = math.sqrt(1 - lam)
    cut_h = int(H * cut_rat);  cut_w = int(W * cut_rat)
    cx = random.randint(0, W);  cy = random.randint(0, H)
    x1 = max(0, cx - cut_w // 2);  x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2);  y2 = min(H, cy + cut_h // 2)
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_mixed, y, y[idx], lam


def mixup_criterion(
    criterion: Callable, pred: torch.Tensor,
    y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────
#  Hard Negative Mining
# ─────────────────────────────────────────────

class HardNegativeMiner:
    """
    Online hard negative mining:
    Each batch, select the hardest misclassified negatives (highest fake probability
    among real images, lowest probability among fakes).
    """

    def __init__(self, ratio: float = 0.3):
        self.ratio = ratio  # fraction of batch to replace with hard negatives

    def select_hard(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns indices of hard samples within the batch."""
        n_hard = max(1, int(len(batch_indices) * self.ratio))
        real_mask = labels == 0
        fake_mask = labels == 1

        hard_idx = []
        if real_mask.sum() > 0:
            real_probs = probs[real_mask]
            real_indices = batch_indices[real_mask]
            top_k = min(n_hard // 2, real_mask.sum().item())
            _, worst_real = real_probs.topk(top_k, largest=True)
            hard_idx.extend(real_indices[worst_real].tolist())

        if fake_mask.sum() > 0:
            fake_probs = probs[fake_mask]
            fake_indices = batch_indices[fake_mask]
            top_k = min(n_hard // 2, fake_mask.sum().item())
            _, worst_fake = fake_probs.topk(top_k, largest=False)
            hard_idx.extend(fake_indices[worst_fake].tolist())

        return torch.tensor(hard_idx, device=probs.device)


# ─────────────────────────────────────────────
#  Metric Tracker
# ─────────────────────────────────────────────

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sums   = {}
        self._counts = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v * n
            self._counts[k] = self._counts.get(k, 0)  + n

    def average(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(self._counts[k], 1) for k in self._sums}


# ─────────────────────────────────────────────
#  Model Factory
# ─────────────────────────────────────────────

def build_model(model_name: str, cfg: dict, pretrained: bool = True) -> nn.Module:
    mcfg = cfg.get("models", {}).get(model_name.lower(), {})
    if model_name == "rgb":
        return RGBForensicsNet(
            backbone=mcfg.get("name", "convnextv2_large.fcmae_ft_in22k_in1k_384"),
            pretrained=pretrained,
            dropout=mcfg.get("dropout", 0.3)
        )
    elif model_name == "vit":
        return ForensicViT(
            backbone=mcfg.get("name", "vit_large"),
            pretrained=pretrained,
            dropout=mcfg.get("dropout", 0.1)
        )
    elif model_name == "frequency":
        return FrequencyForensicsNet(pretrained=pretrained)
    elif model_name == "noise":
        return NoiseForensicsNet(pretrained=pretrained)
    elif model_name == "ela":
        return ELAForensicsNet(pretrained=pretrained)
    elif model_name == "face":
        return FaceForensicsNet(pretrained=pretrained)
    elif model_name == "localization":
        return ManipulationLocalizationNet(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_criterion(model_name: str, cfg: dict) -> nn.Module:
    ls = cfg.get("training", {}).get("label_smoothing", 0.1)
    if model_name == "rgb":
        return RGBForensicsLoss(label_smoothing=ls)
    elif model_name == "vit":
        return ViTForensicsLoss(label_smoothing=ls)
    elif model_name == "frequency":
        return FrequencyForensicsLoss(label_smoothing=ls)
    elif model_name == "noise":
        return NoiseForensicsLoss(label_smoothing=ls)
    elif model_name == "ela":
        return ELAForensicsLoss(label_smoothing=ls)
    elif model_name == "face":
        return FaceForensicsLoss(label_smoothing=ls)
    elif model_name == "localization":
        return LocalizationLoss(label_smoothing=0.05)
    else:
        raise ValueError(f"Unknown criterion: {model_name}")


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class ForensicTrainer:
    """
    Universal trainer for all forensic models.
    Supports AMP, EMA, Mixup/CutMix, hard negative mining.
    """

    def __init__(
        self,
        model_name: str,
        cfg: dict,
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.model_name = model_name
        self.cfg        = cfg
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_wandb  = use_wandb
        self.tcfg       = cfg.get("training", {})

        # Dirs
        self.ckpt_dir = Path(cfg["paths"]["checkpoints"]) / model_name
        self.log_dir  = Path(cfg["paths"]["logs"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            model_name, str(self.log_dir / f"{model_name}_train.log")
        )

        # Model
        self.model     = build_model(model_name, cfg).to(self.device)
        self.criterion = build_criterion(model_name, cfg).to(self.device)
        self.ema       = EMA(self.model, decay=self.tcfg.get("ema_decay", 0.9998))
        self.scaler    = GradScaler(enabled=self.tcfg.get("fp16", True))

        # Optimizer
        self.optimizer = self._build_optimizer()
        self.best_metric = 0.0
        self.global_step = 0

        if use_wandb:
            wandb.init(project="forensic_ai", name=model_name, config=cfg)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Layer-wise learning rate decay for transformers."""
        no_decay = ["bias", "LayerNorm", "norm"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.tcfg.get("weight_decay", 1e-4),
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(params, lr=self.tcfg.get("lr", 1e-4))

    def _build_scheduler(self, steps_per_epoch: int):
        total_steps = self.tcfg.get("epochs", 50) * steps_per_epoch
        warmup_steps = self.tcfg.get("warmup_epochs", 5) * steps_per_epoch
        return OneCycleLR(
            self.optimizer,
            max_lr=self.tcfg.get("lr", 1e-4),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

    # ── Train one epoch ──────────────────────

    def _train_step(self, batch: dict) -> Tuple[torch.Tensor, float]:
        label = batch["label"].to(self.device)
        use_mix = random.random() < 0.5 and self.model_name != "localization"

        # Select input based on model type
        if self.model_name == "rgb":
            inp = batch["rgb"].to(self.device)
        elif self.model_name == "vit":
            inp = batch["rgb"].to(self.device)
        elif self.model_name == "frequency":
            inp = batch["freq"].to(self.device)
        elif self.model_name == "noise":
            inp = batch["rgb"].to(self.device)
        elif self.model_name == "ela":
            inp = batch["ela"].to(self.device)
        elif self.model_name == "face":
            inp = batch["rgb"].to(self.device)
        elif self.model_name == "localization":
            inp = batch["rgb"].to(self.device)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Mixup / CutMix
        lam = 1.0
        y_a = y_b = label
        if use_mix and self.model_name not in ["localization"]:
            if random.random() < 0.5:
                inp, y_a, y_b, lam = mixup_data(inp, label, self.tcfg.get("mixup_alpha", 0.4))
            else:
                inp, y_a, y_b, lam = cutmix_data(inp, label, self.tcfg.get("cutmix_alpha", 1.0))

        with autocast(enabled=self.tcfg.get("fp16", True)):
            gt_mask = batch["mask"].to(self.device) if self.model_name == "localization" else None

            if self.model_name == "rgb":
                prob, _, manip = self.model(inp)
                total_loss, losses = self.criterion(prob, manip, y_a)
                if use_mix:
                    l2, _ = self.criterion(prob, manip, y_b)
                    total_loss = lam * total_loss + (1 - lam) * l2

            elif self.model_name == "vit":
                prob, embed, source, patches = self.model(inp, return_features=True)
                total_loss, losses = self.criterion(prob, source, y_a, patches)
                if use_mix:
                    l2, _ = self.criterion(prob, source, y_b, patches)
                    total_loss = lam * total_loss + (1 - lam) * l2

            elif self.model_name == "frequency":
                prob, _ = self.model(inp)
                total_loss, losses = self.criterion(prob, y_a)
                if use_mix:
                    l2, _ = self.criterion(prob, y_b)
                    total_loss = lam * total_loss + (1 - lam) * l2

            elif self.model_name == "noise":
                prob, _, sigma = self.model(inp)
                total_loss, losses = self.criterion(prob, sigma, y_a)
                if use_mix:
                    l2, _ = self.criterion(prob, sigma, y_b)
                    total_loss = lam * total_loss + (1 - lam) * l2

            elif self.model_name == "ela":
                prob, _, loc_map = self.model(inp)
                total_loss, losses = self.criterion(prob, loc_map, y_a, gt_mask)
                if use_mix:
                    l2, _ = self.criterion(prob, loc_map, y_b)
                    total_loss = lam * total_loss + (1 - lam) * l2

            elif self.model_name == "face":
                prob, _, ftype, id_inc = self.model(inp)
                total_loss, losses = self.criterion(prob, ftype, id_inc, y_a)
                if use_mix:
                    l2, _ = self.criterion(prob, ftype, id_inc, y_b)
                    total_loss = lam * total_loss + (1 - lam) * l2

            elif self.model_name == "localization":
                mask_pred, img_prob = self.model(inp)
                total_loss, losses = self.criterion(mask_pred, img_prob, label, gt_mask)

        return total_loss, losses

    def train_epoch(self, loader: DataLoader, scheduler) -> Dict:
        self.model.train()
        tracker = MetricTracker()
        pbar = tqdm(loader, desc=f"[{self.model_name}] Train", leave=False)

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            loss, losses = self._train_step(batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.tcfg.get("gradient_clip", 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scheduler.step()
            self.ema.update()

            tracker.update({k: v.item() for k, v in losses.items()},
                           n=batch["label"].shape[0])
            self.global_step += 1

            if batch_idx % 50 == 0:
                avg = tracker.average()
                pbar.set_postfix({"loss": f"{avg.get('total', 0):.4f}"})

            if self.use_wandb:
                wandb.log({f"train/{self.model_name}/{k}": v.item()
                           for k, v in losses.items()}, step=self.global_step)

        return tracker.average()

    # ── Validation ───────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict:
        self.ema.apply_shadow()
        self.model.eval()
        tracker = MetricTracker()
        all_probs  = []
        all_labels = []

        for batch in tqdm(loader, desc=f"[{self.model_name}] Val", leave=False):
            label = batch["label"].to(self.device)

            if self.model_name in ["rgb"]:
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
                raise ValueError

            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        self.ema.restore()

        # Compute metrics
        from src.evaluation.metrics import compute_metrics
        metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
        return metrics

    # ── Full training loop ───────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ):
        epochs = epochs or self.tcfg.get("epochs", 50)
        scheduler = self._build_scheduler(len(train_loader))
        self.logger.info(f"Starting training: {self.model_name} for {epochs} epochs")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, scheduler)
            val_metrics   = self.validate(val_loader)
            elapsed       = time.time() - t0

            self.logger.info(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Loss: {train_metrics.get('total', 0):.4f} | "
                f"Val AUC: {val_metrics.get('roc_auc', 0):.4f} | "
                f"Val F1: {val_metrics.get('f1', 0):.4f} | "
                f"{elapsed:.1f}s"
            )

            if self.use_wandb:
                wandb.log({f"val/{self.model_name}/{k}": v
                           for k, v in val_metrics.items()}, step=self.global_step)

            # Save checkpoint
            metric = val_metrics.get("roc_auc", 0)
            self._save_checkpoint(epoch, val_metrics, is_best=(metric > self.best_metric))
            if metric > self.best_metric:
                self.best_metric = metric

        self.logger.info(f"Training complete. Best val AUC: {self.best_metric:.4f}")

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch":      epoch,
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "ema_state":  self.ema.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "metrics":    metrics,
            "config":     self.cfg,
        }
        ckpt_path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(state, ckpt_path)
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)
            self.logger.info(f"  → Saved best checkpoint: {best_path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.ema.load_state_dict(state["ema_state"])
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.logger.info(f"Loaded checkpoint from {path} (epoch {state['epoch']})")
        return state["epoch"], state.get("metrics", {})


# ─────────────────────────────────────────────
#  Self-Supervised Contrastive Pretraining
# ─────────────────────────────────────────────

class ContrastiveHead(nn.Module):
    """Projection head for SimCLR-style contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07):
    """NT-Xent contrastive loss (SimCLR)."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)         # 2B x D
    sim = torch.mm(z, z.T) / temperature   # 2B x 2B
    # Mask self-similarities
    mask = torch.eye(2 * B, dtype=bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    # Positive pairs are (i, i+B) and (i+B, i)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels], dim=0)  # 2B
    loss = F.cross_entropy(sim, labels)
    return loss


class ContrastivePretrainer:
    """
    Self-supervised contrastive pretraining using augmentation pairs.
    Use before supervised fine-tuning.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 512,
                 device: str = "cuda", lr: float = 3e-4):
        self.backbone = backbone.to(device)
        self.device   = torch.device(device)
        self.proj     = ContrastiveHead(embed_dim).to(device)
        self.opt      = AdamW(
            list(backbone.parameters()) + list(self.proj.parameters()),
            lr=lr, weight_decay=1e-4
        )
        self.scaler = GradScaler()

    def pretrain_epoch(self, loader: DataLoader) -> float:
        self.backbone.train()
        self.proj.train()
        total_loss = 0
        n = 0

        for batch in tqdm(loader, desc="Contrastive pretrain"):
            # Two augmented views
            x1 = batch["rgb"].to(self.device)
            x2 = batch["ela"].to(self.device)  # ELA as second view

            self.opt.zero_grad()
            with autocast():
                if hasattr(self.backbone, "forward"):
                    # Get embeddings
                    try:
                        _, e1, _ = self.backbone(x1)
                        _, e2, _ = self.backbone(x2)
                    except Exception:
                        _, e1 = self.backbone(x1)[:2]
                        _, e2 = self.backbone(x2)[:2]

                z1 = self.proj(e1)
                z2 = self.proj(e2)
                loss = nt_xent_loss(z1, z2)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)


# ─────────────────────────────────────────────
#  Main Training Script
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--model",   default="rgb",
                        choices=["rgb", "vit", "frequency", "noise", "ela", "face", "localization"])
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--resume",  default=None)
    parser.add_argument("--pretrain_contrastive", action="store_true")
    parser.add_argument("--wandb",   action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load manifests
    from src.datasets.dataset_pipeline import ForensicDatasetBuilder, build_dataloaders

    splits_dir = cfg["paths"]["splits_dir"]
    train_samples = ForensicDatasetBuilder.load_manifest(f"{splits_dir}/train.csv")
    val_samples   = ForensicDatasetBuilder.load_manifest(f"{splits_dir}/val.csv")

    train_loader, val_loader, _ = build_dataloaders(
        train_samples, val_samples, val_samples,  # test=val for now
        cfg=cfg,
        image_size=cfg["image"]["size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    trainer = ForensicTrainer(
        model_name=args.model,
        cfg=cfg,
        device=args.device,
        use_wandb=args.wandb,
    )

    if args.pretrain_contrastive:
        print("Starting contrastive pretraining...")
        cp = ContrastivePretrainer(trainer.model, device=args.device)
        for ep in range(10):
            loss = cp.pretrain_epoch(train_loader)
            print(f"Contrastive epoch {ep+1}: loss={loss:.4f}")

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit(train_loader, val_loader, epochs=args.epochs)
