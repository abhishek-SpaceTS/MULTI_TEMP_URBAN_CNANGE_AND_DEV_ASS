"""
segmentation_F/evaluate.py
===========================
Fast evaluation on test set — no panel saving, just metrics.

Usage:
    python3 segmentation_F/evaluate.py \
        --checkpoint segmentation_F/checkpoints_v5/best.pt \
        --patches_dir data/processed/tiling/patches \
        --fbc_dir     data/processed/mask_engineering/watershed \
        --threshold   0.45
"""

import argparse
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from model   import TE_UNet
from dataset import TEST_CITIES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NORM_MEAN = np.array([0.4736, 0.4196, 0.3154], dtype=np.float32)
NORM_STD  = np.array([0.2381, 0.1894, 0.1756], dtype=np.float32)
N_MONTHS  = 5
PATCH_SIZE = 256


# ── Model ─────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    raw       = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = raw.get("model_cfg", {}) if isinstance(raw, dict) else {}
    fbc_ch    = model_cfg.get("fbc_ch", 1)

    val = raw.get("val", {})
    logger.info(f"Checkpoint: epoch={raw.get('epoch','?')}  fbc_ch={fbc_ch}  "
                f"val_seg_f1={val.get('seg_f1',0):.4f}  "
                f"val_chg_iou={val.get('change_iou',0):.4f}")

    model = TE_UNet(
        in_ch_per_frame  = model_cfg.get("in_ch_per_frame", 3),
        n_months         = model_cfg.get("n_months", N_MONTHS),
        fbc_ch           = fbc_ch,
        topology         = model_cfg.get("topology", [32,64,128,256,512]),
        n_heads          = model_cfg.get("n_heads", 4),
        n_layers         = model_cfg.get("n_layers", 2),
        deep_supervision = False,
    )
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, fbc_ch


# ── Data helpers ──────────────────────────────────────────────
def _load_tif_rgb(path):
    try:
        import rasterio
        with rasterio.open(path) as src:
            return np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
    except:
        return np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)


def _normalize(img):
    return (img.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD


def _load_mask(path, fbc_ch):
    try:
        img = Image.open(path)
        if fbc_ch == 3:
            return (np.array(img.convert("RGB"))[:,:,0] > 127).astype(np.float32)
        else:
            return (np.array(img.convert("L")) > 127).astype(np.float32)
    except:
        return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


def _parse_stem(stem):
    m = re.match(r"^(.+?)_m(\d{2})_r(\d{2})_c(\d{2})$", stem)
    if not m: return None
    return {"location": m.group(1), "month_idx": int(m.group(2)),
            "row": int(m.group(3)), "col": int(m.group(4))}


def discover_positions(patches_dir, fbc_dir, fbc_ch, allowed_cities=None):
    mask_suffix = "_fbc.png" if fbc_ch == 3 else "_watershed.png"
    positions   = []
    city_dirs   = sorted([d for d in patches_dir.iterdir() if d.is_dir()])
    if allowed_cities:
        city_dirs = [d for d in city_dirs if d.name in set(allowed_cities)]
    logger.info(f"Cities: {len(city_dirs)}")

    for city_dir in city_dirs:
        city     = city_dir.name
        mask_dir = fbc_dir / city
        if not mask_dir.exists(): continue

        pos_months = defaultdict(dict)
        for tif in sorted(city_dir.glob("*.tif")):
            info = _parse_stem(tif.stem)
            if info is None: continue
            pos_months[(info["row"], info["col"])][info["month_idx"]] = tif

        for (row, col), month_dict in pos_months.items():
            if len(month_dict) < N_MONTHS: continue
            sorted_months = sorted(month_dict.keys())[:N_MONTHS]
            image_paths   = [month_dict[m] for m in sorted_months]

            mask_paths = []
            valid = True
            for img_path in image_paths:
                mp = mask_dir / f"{img_path.stem}{mask_suffix}"
                if not mp.exists(): valid = False; break
                mask_paths.append(mp)
            if not valid: continue

            positions.append({
                "city": city, "row": row, "col": col,
                "image_paths": image_paths,
                "mask_paths":  mask_paths,
                "months":      sorted_months,
            })

    logger.info(f"Positions: {len(positions)}")
    return positions


# ── Evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, fbc_ch, positions, device, seg_thresh, chg_thresh):
    # Accumulators
    seg_tp = seg_fp = seg_fn = 0.0
    chg_tp = chg_fp = chg_fn = 0.0
    n = len(positions)
    fp_idx = list(range(0, fbc_ch * N_MONTHS, fbc_ch))

    print(f"\nEvaluating {n} positions...")
    for i, pos in enumerate(positions):
        if i % 100 == 0:
            logger.info(f"  {i}/{n}...")

        # Build input
        channels = []
        for p in pos["image_paths"]:
            norm = _normalize(_load_tif_rgb(p))
            channels.append(norm.transpose(2, 0, 1))
        x = torch.from_numpy(
            np.concatenate(channels, axis=0)
        ).float().unsqueeze(0).to(device)

        # Forward — duplicate for GroupNorm batch=1 safety
        x2  = torch.cat([x, x], dim=0)
        out = model(x2)

        # Predictions
        fbc_prob    = torch.sigmoid(out["fbc"])[0].cpu().numpy()
        change_prob = torch.sigmoid(out["change"])[0, 0].cpu().numpy()

        pred_seg = (fbc_prob[fp_idx[-1]] > seg_thresh)  # last month footprint
        pred_chg = (change_prob > chg_thresh)

        # GT
        gt_masks = [_load_mask(p, fbc_ch) for p in pos["mask_paths"]]
        gt_seg   = gt_masks[-1].astype(bool)  # last month

        # Change GT: pixels that changed between first and last month
        gt_chg = (gt_masks[0] != gt_masks[-1])

        # Segmentation metrics
        seg_tp += (pred_seg & gt_seg).sum()
        seg_fp += (pred_seg & ~gt_seg).sum()
        seg_fn += (~pred_seg & gt_seg).sum()

        # Change metrics
        chg_tp += (pred_chg & gt_chg).sum()
        chg_fp += (pred_chg & ~gt_chg).sum()
        chg_fn += (~pred_chg & gt_chg).sum()

    eps = 1e-6

    seg_iou  = seg_tp / (seg_tp + seg_fp + seg_fn + eps)
    seg_f1   = 2*seg_tp / (2*seg_tp + seg_fp + seg_fn + eps)
    seg_prec = seg_tp / (seg_tp + seg_fp + eps)
    seg_rec  = seg_tp / (seg_tp + seg_fn + eps)

    chg_iou  = chg_tp / (chg_tp + chg_fp + chg_fn + eps)
    chg_f1   = 2*chg_tp / (2*chg_tp + chg_fp + chg_fn + eps)
    chg_prec = chg_tp / (chg_tp + chg_fp + eps)
    chg_rec  = chg_tp / (chg_tp + chg_fn + eps)

    print("\n" + "="*55)
    print("  TEST SET EVALUATION RESULTS")
    print("="*55)
    print(f"  Positions evaluated : {n}")
    print(f"  Seg threshold       : {seg_thresh}")
    print(f"  Chg threshold       : {chg_thresh}")
    print(f"  {'─'*50}")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Seg IoU':<25} {seg_iou:>10.4f}")
    print(f"  {'Seg F1':<25} {seg_f1:>10.4f}")
    print(f"  {'Seg Precision':<25} {seg_prec:>10.4f}")
    print(f"  {'Seg Recall':<25} {seg_rec:>10.4f}")
    print(f"  {'─'*50}")
    print(f"  {'Chg IoU':<25} {chg_iou:>10.4f}")
    print(f"  {'Chg F1':<25} {chg_f1:>10.4f}")
    print(f"  {'Chg Precision':<25} {chg_prec:>10.4f}")
    print(f"  {'Chg Recall':<25} {chg_rec:>10.4f}")
    print("="*55)

    return {
        "seg_iou": seg_iou, "seg_f1": seg_f1,
        "seg_prec": seg_prec, "seg_rec": seg_rec,
        "chg_iou": chg_iou, "chg_f1": chg_f1,
        "chg_prec": chg_prec, "chg_rec": chg_rec,
    }


# ── CLI ───────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",    required=True)
    ap.add_argument("--patches_dir",   required=True)
    ap.add_argument("--fbc_dir",       required=True)
    ap.add_argument("--seg_threshold", type=float, default=0.45)
    ap.add_argument("--chg_threshold", type=float, default=0.35)
    ap.add_argument("--device",        default="auto")
    ap.add_argument("--all_cities",    action="store_true")
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    logger.info(f"Device: {device}")

    model, fbc_ch = load_model(args.checkpoint, device)
    allowed       = None if args.all_cities else TEST_CITIES

    positions = discover_positions(
        Path(args.patches_dir), Path(args.fbc_dir),
        fbc_ch=fbc_ch, allowed_cities=allowed,
    )
    if not positions:
        logger.error("No positions found.")
        return

    evaluate(model, fbc_ch, positions, device,
             args.seg_threshold, args.chg_threshold)


if __name__ == "__main__":
    main()