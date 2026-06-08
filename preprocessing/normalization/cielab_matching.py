"""
preprocessing/normalization/cielab_matching.py
================================================

SUB-STEP 3.3 — CIELab Colour Consistency Matching
---------------------------------------------------

PURPOSE:
    Final colour consistency pass after IR-MAD (3.1) and PIF (3.2).
    Converts images to CIELab colour space, matches the L* (lightness)
    and a*, b* (chrominance) channels independently to the reference,
    then converts back to RGB. Eliminates residual hue shifts and
    lightness drifts that linear band-wise methods miss.

WHY CIELab:
    - CIELab is perceptually uniform: equal distance = equal perceived change
    - L* separates lightness from colour → can match luminance without
      distorting hue (important for vegetation/soil discrimination)
    - a* (green-red axis) and b* (blue-yellow axis) capture colour casts
      from atmospheric haze — matching these removes the "blue haze" of
      distant months without overcorrecting saturated surfaces

METHOD:
    For each month (using PIF output as input):
      1. Convert RGB → CIELab using skimage.color
      2. Compute mean/std of L*, a*, b* on USABLE pixels only
      3. Apply z-score rescaling per channel to match reference
      4. Convert back to RGB
      5. Clamp to [0,255], write uint8

    The Alpha band is NEVER touched.

INPUT:
    pif_dir  : data/processed/normalization/pif/{city}/*.tif
    mask_dir : combined quality masks

OUTPUT:
    data/processed/normalization/cielab/{city}/*.tif
    Final normalised images — input to Step 4 tiling.

NEXT STEP → 4.1 chipping.py
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from skimage import color as skcolor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_mask_bool(mask_dir: Path, stem: str, shape) -> np.ndarray:
    p = mask_dir / f"{stem}_combined.tif"
    if not p.exists():
        return np.zeros(shape, dtype=bool)
    with rasterio.open(p) as src:
        return src.read(1).astype(bool)


def _channel_stats(channel: np.ndarray, usable: np.ndarray):
    vals = channel[usable]
    if vals.size < 100:
        return float(channel.mean()), float(channel.std()) + 1e-8
    return float(vals.mean()), float(vals.std()) + 1e-8


def _match_channel(src_ch, src_mu, src_std, ref_mu, ref_std):
    corrected = (src_ch - src_mu) / src_std * ref_std + ref_mu
    return corrected


def run_cielab_matching(
    pif_dir:   Path,
    mask_dir:  Path,
    out_dir:   Path,
) -> Dict:
    """
    CIELab colour consistency matching for ONE city.

    Called by run_pipeline.py → run_step3_normalisation()

    ARGS:
        pif_dir  : PIF regression output (step 3.2)
        mask_dir : combined quality masks (step 2.3)
        out_dir  : final normalised images
    """
    city = pif_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 3.3 : CIELab COLOUR CONSISTENCY MATCHING")
    logger.info("=" * 60)
    logger.info(f"City     : {city}")
    logger.info(f"Method   : RGB → CIELab → match L*, a*, b* → RGB")
    logger.info(f"Output   : {out_dir}")

    tif_files: List[Path] = sorted(pif_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif in {pif_dir}")

    n_months = len(tif_files)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Reference month ───────────────────────────────────────────────────
    ref_path = tif_files[0]
    with rasterio.open(ref_path) as src:
        r = src.read(1).astype(np.float32) / 255.0
        g = src.read(2).astype(np.float32) / 255.0
        b = src.read(3).astype(np.float32) / 255.0
        h, w = src.height, src.width

    ref_rgb_norm = np.stack([r, g, b], axis=-1)      # (H, W, 3) float [0,1]
    ref_lab      = skcolor.rgb2lab(ref_rgb_norm)      # (H, W, 3) CIELab

    ref_mask = _load_mask_bool(mask_dir, ref_path.stem, (h, w))
    ref_use  = ~ref_mask

    # Per-channel reference stats
    ref_mu_L,  ref_std_L  = _channel_stats(ref_lab[:,:,0], ref_use)
    ref_mu_a,  ref_std_a  = _channel_stats(ref_lab[:,:,1], ref_use)
    ref_mu_b,  ref_std_b  = _channel_stats(ref_lab[:,:,2], ref_use)

    shutil.copy2(ref_path, out_dir / ref_path.name)
    logger.info(f"  [1/{n_months}] {ref_path.name[:45]}  [REFERENCE]")
    logger.info(f"    L* ref: mean={ref_mu_L:.1f} std={ref_std_L:.1f}")

    # ── Match remaining months ────────────────────────────────────────────
    for idx, tif_path in enumerate(tif_files[1:], start=2):
        with rasterio.open(tif_path) as src:
            meta  = src.meta.copy()
            r     = src.read(1).astype(np.float32) / 255.0
            g     = src.read(2).astype(np.float32) / 255.0
            b     = src.read(3).astype(np.float32) / 255.0
            alpha = src.read(4)

        mov_rgb = np.stack([r, g, b], axis=-1)
        mov_lab = skcolor.rgb2lab(mov_rgb)

        mov_mask = _load_mask_bool(mask_dir, tif_path.stem, (h, w))
        mov_use  = ~mov_mask

        # Match each CIELab channel
        out_lab = mov_lab.copy()
        for ch_idx, (ref_mu, ref_std) in enumerate([
            (ref_mu_L, ref_std_L),
            (ref_mu_a, ref_std_a),
            (ref_mu_b, ref_std_b),
        ]):
            ch = mov_lab[:, :, ch_idx]
            mu, std = _channel_stats(ch, mov_use)
            out_lab[:, :, ch_idx] = _match_channel(ch, mu, std, ref_mu, ref_std)

        # Back to RGB
        out_rgb = skcolor.lab2rgb(out_lab)                  # float [0,1]
        out_rgb = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)

        dst_path = out_dir / tif_path.name
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(out_rgb[:, :, 0], 1)   # R
            dst.write(out_rgb[:, :, 1], 2)   # G
            dst.write(out_rgb[:, :, 2], 3)   # B
            dst.write(alpha, 4)              # Alpha unchanged

        logger.info(f"  [{idx}/{n_months}] {tif_path.name[:45]}  CIELab matched ✓")

    logger.info("STATUS : CIELab matching complete ✓")
    logger.info("NEXT   : Step 4.1 → Chipping (256×256 tiling)")
    return {"city": city, "n_months": n_months,
            "output_dir": str(out_dir), "status": "complete"}


if __name__ == "__main__":
    C = "L15-0331E-1257N_1327_3160_13"
    run_cielab_matching(
        pif_dir  = Path(f"data/processed/normalization/pif/{C}"),
        mask_dir = Path(f"data/processed/masking/vegetation/{C}"),
        out_dir  = Path(f"data/processed/normalization/cielab/{C}"),
    )