"""
preprocessing/masking/shadow_mask.py
======================================

SUB-STEP 2.2 — Shadow Mask Generation
---------------------------------------

PURPOSE:
    Identify shadow pixels in each monthly image and add them
    to the combined quality mask. Shadow pixels have suppressed
    reflectance and should be excluded from normalisation
    statistics (Step 3) just like clouds.

WHY SHADOWS MATTER FOR NORMALISATION:
    IR-MAD and PIF regression (Step 3) compute radiometric
    shifts between time steps. If shadow pixels are included,
    they pull the statistics towards artificially dark values,
    causing the correction to over-brighten legitimate dark
    surfaces (rooftops, water, asphalt).

METHOD — Dark-Pixel + Context Heuristic:
    True shadow detection in 4-band RGBA (no NIR) is hard.
    We use a practical two-criterion approach:

    1. LUMINANCE THRESHOLD:
       Shadow pixels are dark across all visible bands.
       mean(R, G, B) < SHADOW_DARK_THRESH  (default: 40)
       This catches deep shadows from buildings/trees.

    2. COLOUR RATIO (hue check):
       Shadows cast by the atmosphere or clouds are slightly
       blue-shifted (more sky illumination).
       blue_ratio = B / (mean(R,G,B) + ε) > 1.1
       Helps separate shadow from actual dark objects (asphalt).

    3. NOT CLOUD (logical AND NOT):
       We load the cloud mask from step 2.1 and explicitly
       exclude cloud pixels — they are already handled.
       shadow = dark AND (blue_ratio > 1.1) AND NOT cloud

LIMITATION:
    - Cannot distinguish building shadow from dark rooftop
    - No DSM/DEM → shadow direction cannot be computed
    - Tall structures cast shadows not caught by luminance alone
    This is a best-effort approach for 4-band imagery.

INPUT:
    upsamp_dir  : data/processed/alignment/upsampled/{city}/*.tif
    cloud_dir   : data/processed/masking/cloud/{city}/*_cloud.tif

OUTPUT:
    data/processed/masking/shadow/{city}/*_shadow.tif
    uint8, 0=clear, 1=shadow

NEXT STEP → 2.3 vegetation_mask.py
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SHADOW_DARK_THRESH  = 40    # mean(R,G,B) below this → candidate shadow
BLUE_RATIO_THRESH   = 1.1   # B / mean(R,G,B) above this → shadow (blue-shift)
EPSILON             = 1e-6  # prevents division by zero in ratio


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_cloud_mask(cloud_dir: Path, img_stem: str) -> np.ndarray:
    """
    Loads the cloud mask for a given image stem.

    Cloud mask filename convention: {img_stem}_cloud.tif
    Returns all-zero mask if not found (safe fallback).
    """
    cloud_path = cloud_dir / f"{img_stem}_cloud.tif"
    if not cloud_path.exists():
        logger.warning(f"    Cloud mask not found: {cloud_path.name} — using zero cloud mask")
        return None   # handled by caller → zero array

    with rasterio.open(cloud_path) as src:
        return src.read(1).astype(bool)


def _compute_shadow_mask(
    img_path: Path,
    cloud_mask: np.ndarray,
) -> np.ndarray:
    """
    Computes shadow mask for one monthly image.

    ARGS:
        img_path   : upsampled 4096×4096 GeoTIFF
        cloud_mask : boolean array (H, W), True=cloud

    RETURNS:
        uint8 array (H, W), 0=clear, 1=shadow
    """
    with rasterio.open(img_path) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)

    luminance  = (r + g + b) / 3.0                    # mean brightness
    blue_ratio = b / (luminance + EPSILON)             # blue shift indicator

    # Criterion 1: dark pixels
    is_dark = luminance < SHADOW_DARK_THRESH

    # Criterion 2: blue-shifted (shadow from sky illumination)
    is_blue = blue_ratio > BLUE_RATIO_THRESH

    # Criterion 3: not already flagged as cloud
    not_cloud = ~cloud_mask if cloud_mask is not None else np.ones(r.shape, bool)

    shadow = (is_dark & is_blue & not_cloud).astype(np.uint8)
    return shadow


def _write_mask(mask: np.ndarray, dst_path: Path, reference_img: Path) -> None:
    """Writes binary uint8 mask aligned to reference_img's spatial metadata."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_img) as ref:
        meta = ref.meta.copy()
    meta.update({"count": 1, "dtype": "uint8", "nodata": None})
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(mask[np.newaxis, :, :])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_shadow_masking(
    upsamp_dir: Path,
    cloud_dir: Path,
    out_dir: Path,
) -> Dict:
    """
    Generates shadow masks for all monthly images of ONE city.

    Called by run_pipeline.py → run_step2_masking()

    ARGS:
        upsamp_dir : upsampled images (step 1.3 output)
        cloud_dir  : cloud masks (step 2.1 output)
        out_dir    : destination for shadow masks
    """
    city = upsamp_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 2.2 : SHADOW MASK GENERATION")
    logger.info("=" * 60)
    logger.info(f"City     : {city}")
    logger.info(f"Method   : Dark-pixel + Blue-shift heuristic")
    logger.info(f"Output   : {out_dir}")

    tif_files: List[Path] = sorted(upsamp_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {upsamp_dir}")

    n_months = len(tif_files)
    logger.info(f"Time steps: {n_months} monthly images")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(tif_files):
        cloud_mask = _load_cloud_mask(cloud_dir, img_path.stem)
        if cloud_mask is None:
            h, w = 4096, 4096
            cloud_mask = np.zeros((h, w), dtype=bool)

        shadow = _compute_shadow_mask(img_path, cloud_mask)
        dst    = out_dir / f"{img_path.stem}_shadow.tif"
        _write_mask(shadow, dst, img_path)

        pct = 100.0 * shadow.mean()
        logger.info(f"  [{idx+1}/{n_months}] {img_path.name[:45]}  shadow={pct:.1f}%")

    logger.info("STATUS : Shadow masking complete ✓")
    logger.info("NEXT   : Sub-step 2.3 → Vegetation Mask")
    return {
        "city": city, "n_months": n_months,
        "output_dir": str(out_dir), "status": "complete"
    }


if __name__ == "__main__":
    run_shadow_masking(
        upsamp_dir = Path("data/processed/alignment_1/upsampled/L15-0358E-1220N_1433_3310_13"),
        cloud_dir  = Path("data/processed/masking_1/cloud/L15-0358E-1220N_1433_3310_13"),
        out_dir    = Path("data/processed/masking_1/shadow/L15-0358E-1220N_1433_3310_13"),
    )
