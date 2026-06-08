"""
preprocessing/normalization/pif_regression.py
===============================================

SUB-STEP 3.2 — Pseudo-Invariant Feature (PIF) Regression
----------------------------------------------------------

PURPOSE:
    PIF regression is a FALLBACK / COMPLEMENT to IR-MAD.
    It handles months that IR-MAD skipped (too cloudy) or
    that still show residual radiometric drift after step 3.1.

WHAT ARE PIFs:
    Pseudo-Invariant Features = pixels whose reflectance
    should NOT change between months. Examples:
      - Concrete roads (stable grey)
      - Large flat rooftops
      - Bare soil in arid regions
      - Airport tarmac

    Method:
      1. Use the combined quality mask to find "usable" pixels
      2. Further filter to find "stable" pixels:
         - Low temporal variance across all months
         - Brightness in a mid-range (not cloud-bright, not shadow-dark)
      3. Fit OLS regression for each band:
         target_band = slope × source_band + intercept
         (where target = reference month, source = moving month)
      4. Apply correction to all pixels

WHY PIF AFTER IR-MAD:
    IR-MAD is computationally robust but assumes Gaussian
    radiometric shift. PIF regression provides a data-driven
    correction using real stable surfaces — complementary.
    Together they handle: atmospheric haze, sun angle, and
    sensor response variation.

SKIPPED MONTHS:
    If irmad_dir contains a file already (IR-MAD ran on it),
    PIF refines it. If IR-MAD skipped it, PIF processes it
    from the upsampled version directly.

INPUT:
    upsamp_dir : original upsampled images (step 1.3)
    irmad_dir  : IR-MAD corrected images (step 3.1)
    mask_dir   : combined quality masks (step 2.3)

OUTPUT:
    data/processed/normalization/pif/{city}/*.tif
    PIF-refined normalisation

NEXT STEP → 3.3 cielab_matching.py
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from scipy import stats   # for OLS linear regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

NORM_BANDS = [1, 2, 3]          # R, G, B — Alpha untouched
# PIF brightness range: exclude very dark (shadow) and very bright (cloud)
PIF_MIN_BRIGHTNESS = 60          # lower bound (uint8)
PIF_MAX_BRIGHTNESS = 190         # upper bound (uint8)
# Minimum PIF pixels for reliable regression
MIN_PIF_PIXELS = 500


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_mask(mask_dir: Path, stem: str, shape: Tuple) -> np.ndarray:
    """Loads combined mask (0=usable) or returns all-usable if missing."""
    p = mask_dir / f"{stem}_combined.tif"
    if not p.exists():
        return np.zeros(shape, dtype=bool)
    with rasterio.open(p) as src:
        return src.read(1).astype(bool)


def _find_pif_pixels(
    ref_data: np.ndarray,
    mov_data: np.ndarray,
    usable:   np.ndarray,
) -> np.ndarray:
    """
    Finds Pseudo-Invariant Feature pixels between reference and
    moving image.

    PIFs are usable pixels where:
      1. Brightness is in the mid-range (not cloud, not shadow)
      2. Absolute difference between ref and moving is small
         (likely stable surface, not a changed/seasonal pixel)

    ARGS:
        ref_data : (H, W) float32, one band of reference image
        mov_data : (H, W) float32, one band of moving image
        usable   : (H, W) bool, True = usable (not masked)

    RETURNS:
        (H, W) boolean mask, True = PIF pixel
    """
    mean_brightness = (ref_data + mov_data) / 2.0
    diff = np.abs(ref_data - mov_data)

    # Median absolute difference — robust measure of "small change"
    usable_diff = diff[usable]
    if usable_diff.size == 0:
        return np.zeros_like(usable)

    diff_threshold = np.percentile(usable_diff, 30)  # bottom 30% = most stable

    pif = (
        usable
        & (mean_brightness >= PIF_MIN_BRIGHTNESS)
        & (mean_brightness <= PIF_MAX_BRIGHTNESS)
        & (diff <= diff_threshold)
    )
    return pif


def _ols_correction(
    ref_vals: np.ndarray,
    mov_vals: np.ndarray,
    all_mov:  np.ndarray,
) -> np.ndarray:
    """
    Fits OLS regression with safety checks for zero-variance and math singularities.
    """
    # 1. Standard check for minimum pixel count
    if len(mov_vals) < MIN_PIF_PIXELS:
        return np.clip(all_mov, 0, 255).astype(np.uint8)

    # 2. ZERO-VARIANCE CHECK: Peak-to-Peak (ptp) variation
    # If mov_vals are all the same (e.g., all 0 or all 120), ptp is 0.
    if np.ptp(mov_vals) == 0:
        logger.warning("  ⚠️ Identical PIF values detected (zero variance). Skipping OLS correction.")
        return np.clip(all_mov, 0, 255).astype(np.uint8)

    # 3. ROBUST REGRESSION ATTEMPT
    try:
        slope, intercept, r, p, se = stats.linregress(mov_vals, ref_vals)
        
        # Check for NaN results which can happen with near-zero variance
        if np.isnan(slope) or np.isnan(intercept):
            logger.warning("  ⚠️ Regression resulted in NaN. Falling back to original data.")
            return np.clip(all_mov, 0, 255).astype(np.uint8)
            
        corrected = slope * all_mov + intercept
        return np.clip(corrected, 0, 255).astype(np.uint8)
        
    except ValueError as e:
        # Catch the specific scipy error and allow the pipeline to continue
        logger.error(f"  ❌ Regression error: {e}. Skipping band.")
        return np.clip(all_mov, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pif_regression(
    upsamp_dir: Path,
    irmad_dir:  Path,
    mask_dir:   Path,
    out_dir:    Path,
) -> Dict:
    """
    PIF regression normalisation for ALL months of ONE city.

    Input is irmad_dir (use IR-MAD output as source when available,
    fall back to upsamp_dir for skipped months).

    Called by run_pipeline.py → run_step3_normalisation()
    """
    city = upsamp_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 3.2 : PIF REGRESSION NORMALISATION")
    logger.info("=" * 60)
    logger.info(f"City     : {city}")

    tif_files: List[Path] = sorted(upsamp_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif in {upsamp_dir}")

    n_months = len(tif_files)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reference (month 0) ──────────────────────────────────────────
    ref_src_path = irmad_dir / tif_files[0].name
    if not ref_src_path.exists():
        ref_src_path = tif_files[0]

    with rasterio.open(ref_src_path) as ref:
        ref_bands = {b: ref.read(b).astype(np.float32) for b in NORM_BANDS}
        h, w = ref.height, ref.width

    ref_mask = _load_mask(mask_dir, tif_files[0].stem, (h, w))
    ref_usable = ~ref_mask

    shutil.copy2(ref_src_path, out_dir / tif_files[0].name)
    logger.info(f"  [1/{n_months}] {tif_files[0].name[:45]}  [REFERENCE]")

    # ── PIF-correct remaining months ──────────────────────────────────────
    for idx, orig_path in enumerate(tif_files[1:], start=2):
        # Prefer IR-MAD output as source; fall back to upsampled
        src_path = irmad_dir / orig_path.name
        if not src_path.exists():
            src_path = orig_path

        with rasterio.open(src_path) as src:
            meta  = src.meta.copy()
            alpha = src.read(4)
            mov_bands = {b: src.read(b).astype(np.float32) for b in NORM_BANDS}

        mov_mask   = _load_mask(mask_dir, orig_path.stem, (h, w))
        usable     = ref_usable & ~mov_mask

        corrected_bands = {}
        pif_counts = []

        for b in NORM_BANDS:
            pif_mask = _find_pif_pixels(ref_bands[b], mov_bands[b], usable)
            pif_counts.append(int(pif_mask.sum()))

            corrected = _ols_correction(
                ref_bands[b][pif_mask],
                mov_bands[b][pif_mask],
                mov_bands[b],
            )
            corrected_bands[b] = corrected

        dst_path = out_dir / orig_path.name
        with rasterio.open(dst_path, "w", **meta) as dst:
            for b in NORM_BANDS:
                dst.write(corrected_bands[b], b)
            dst.write(alpha, 4)

        logger.info(
            f"  [{idx}/{n_months}] {orig_path.name[:40]}  "
            f"PIF px: R={pif_counts[0]:,} G={pif_counts[1]:,} B={pif_counts[2]:,}"
        )

    logger.info("STATUS : PIF regression complete ✓")
    logger.info("NEXT   : Sub-step 3.3 → CIELab Matching")
    return {"city": city, "n_months": n_months,
            "output_dir": str(out_dir), "status": "complete"}


if __name__ == "__main__":
    C = "L15-0331E-1257N_1327_3160_13"
    run_pif_regression(
        upsamp_dir = Path(f"data/processed/alignment/upsampled/{C}"),
        irmad_dir  = Path(f"data/processed/normalization/irmad/{C}"),
        mask_dir   = Path(f"data/processed/masking/vegetation/{C}"),
        out_dir    = Path(f"data/processed/normalization/pif/{C}"),
    )
