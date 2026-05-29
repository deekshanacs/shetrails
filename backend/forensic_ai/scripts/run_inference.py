#!/usr/bin/env python3
"""
run_inference.py
----------------
CLI inference script for the Forensic AI Engine.

Usage:
    python scripts/run_inference.py --image suspect.jpg
    python scripts/run_inference.py --image suspect.jpg --save_dir outputs/reports --explain
    python scripts/run_inference.py --batch_dir folder_of_images/ --save_dir outputs/reports
    python scripts/run_inference.py --image suspect.jpg --models rgb vit frequency
"""

import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import ForensicEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("run_inference")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Forensic AI — Inference")
    p.add_argument("--image",        type=str, default=None,
                   help="Path to a single image")
    p.add_argument("--batch_dir",    type=str, default=None,
                   help="Directory of images for batch inference")
    p.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints",
                   help="Path to model checkpoints")
    p.add_argument("--save_dir",     type=str, default=None,
                   help="Where to save report/heatmap/mask outputs")
    p.add_argument("--explain",      action="store_true",
                   help="Generate GradCAM explanation heatmap")
    p.add_argument("--models",       nargs="+",
                   default=["rgb", "vit", "frequency", "noise", "ela", "face", "localization"],
                   help="Subset of models to run")
    p.add_argument("--json_only",    action="store_true",
                   help="Print only JSON output (no text report)")
    return p.parse_args()


def print_result(result: dict, json_only: bool = False):
    if json_only:
        safe = {k: v for k, v in result.items() if not hasattr(v, "tolist")}
        # Convert numpy to list where needed
        for k in ("mask", "heatmap"):
            if result.get(k) is not None:
                safe[k] = f"<ndarray shape={result[k].shape}>"
        print(json.dumps(safe, indent=2, default=str))
    else:
        print(result["report"])
        print(f"\nStatus    : {result['status']}")
        print(f"Risk      : {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Elapsed   : {result['elapsed_seconds']:.2f}s")


def main():
    args = parse_args()

    engine = ForensicEngine(
        checkpoint_dir=args.checkpoint_dir,
        generate_heatmap=args.explain,
        generate_mask=True,
        models_to_run=args.models,
    )
    engine.warmup()

    if args.image:
        result = engine.analyze_image(
            args.image,
            explain=args.explain,
            save_dir=args.save_dir,
        )
        print_result(result, args.json_only)

    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        images = [p for p in batch_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        logger.info(f"Found {len(images)} images in {batch_dir}")

        results = []
        for img_path in images:
            logger.info(f"Processing: {img_path.name}")
            r = engine.analyze_image(str(img_path), save_dir=args.save_dir)
            results.append({
                "file":       str(img_path.name),
                "status":     r["status"],
                "risk_level": r["risk_level"],
                "confidence": r["confidence"],
                "elapsed":    r["elapsed_seconds"],
            })

        # Summary table
        print(f"\n{'File':<40} {'Status':<12} {'Risk':<10} {'Confidence'}")
        print("-" * 80)
        for r in results:
            print(f"{r['file']:<40} {r['status']:<12} {r['risk_level']:<10} {r['confidence']:.4f}")

        if args.save_dir:
            summary_path = Path(args.save_dir) / "batch_summary.json"
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nBatch summary saved: {summary_path}")
    else:
        print("Provide --image or --batch_dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
