"""
preprocessing/masking/vegetation_mask.py
==========================================

SUB-STEP 2.3 — Vegetation Mask + Combined Quality Mask
--------------------------------------------------------

PURPOSE:
    1. Identify vegetation pixels using a proxy NDVI
       (computed from R and G bands — no NIR in the RGBA stack)
    2. Combine cloud + shadow + vegetation into a single
       QUALITY MASK that all downstream steps consume

WHY MASK VEGETATION FOR NORMALISATION:
    Vegetation pixels change reflectance seasonally:
      Summer → dense green canopy (dark R, bright G)
      Winter → bare branches or dry grass (bright R, darker G)
    Including seasonal vegetation in the normalisation statistics
    (IR-MAD, PIF, CIELab) creates a false radiometric shift
    that mimics "change" when in fact it is just phenology.
    Excluding vegetation makes the normalisation focus on
    urban surfaces (concrete, asphalt, rooftops) which are
    more stable over time — exactly what the model cares about.

PROXY NDVI (WITHOUT NIR):
    True NDVI = (NIR - R) / (NIR + R)
    SpaceNet 7 has R, G, B, Alpha — no NIR band.

    We approximate using the Visible Atmospherically Resistant
    Index proxy:
        vNDVI_proxy = (G - R) / (G + R + ε)
    Plants absorb red light (photosynthesis) and reflect green.
    This proxy is weaker than true NDVI but sufficient for
    masking obviously vegetated areas.

    Threshold: vNDVI_proxy > VEG_THRESH (default: 0.1)
    Pixels above threshold → vegetation.

COMBINED QUALITY MASK OUTPUT:
    combined = cloud OR shadow OR vegetation
    Values: 0 = usable pixel, 1 = masked (cloud/shadow/vegetation)

    This combined mask is the "udm" used by:
      - Step 3 normalisation (statistics computed on combined==0)
      - Step 5 FBC mask (masked patches get lower training weight)
      - Step 6 augmentation (mask is synced with image)

INPUT:
    upsamp_dir  : upsampled images (for green/red band reading)
    cloud_dir   : *_cloud.tif files from step 2.1
    shadow_dir  : *_shadow.tif files from step 2.2

OUTPUT:
    out_dir/{stem}_veg.tif      — vegetation-only mask
    out_dir/{stem}_combined.tif — cloud + shadow + vegetation

NEXT STEP → 3.1 irmad.py  (receives upsamp_dir + combined masks)
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

VEG_THRESH = 0.10    # vNDVI_proxy threshold; >0.10 = vegetation
EPSILON    = 1e-6    # prevents division by zero


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_binary_mask(mask_path: Path, fallback_shape=(4096, 4096)) -> np.ndarray:
    """
    Loads a single-band uint8 GeoTIFF mask as a boolean array.
    Returns all-False (no masking) if file does not exist.
    """
    if not mask_path.exists():
        logger.warning(f"    Mask not found: {mask_path.name} — using zero fallback")
        return np.zeros(fallback_shape, dtype=bool)

    with rasterio.open(mask_path) as src:
        return src.read(1).astype(bool)


def _compute_veg_mask(img_path: Path, threshold: float = VEG_THRESH) -> np.ndarray:
    """
    Computes proxy-NDVI vegetation mask for one image.

    vNDVI_proxy = (Green - Red) / (Green + Red + ε)
    Pixels above threshold are classified as vegetation.

    ARGS:
        img_path  : upsampled 4096×4096 GeoTIFF (R=band1, G=band2)
        threshold : vNDVI_proxy cutoff (default 0.10)

    RETURNS:
        uint8 (H, W), 0=non-veg, 1=vegetation
    """
    with rasterio.open(img_path) as src:
        r = src.read(1).astype(np.float32)   # Band 1 = Red
        g = src.read(2).astype(np.float32)   # Band 2 = Green

    vndvi = (g - r) / (g + r + EPSILON)
    veg   = (vndvi > threshold).astype(np.uint8)
    return veg


def _write_mask(mask: np.ndarray, dst_path: Path, reference_img: Path) -> None:
    """Writes binary uint8 mask, spatially registered to reference_img."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_img) as ref:
        meta = ref.meta.copy()
    meta.update({"count": 1, "dtype": "uint8", "nodata": None})
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(mask[np.newaxis, :, :])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_vegetation_masking(
    upsamp_dir: Path,
    cloud_dir:  Path,
    shadow_dir: Path,
    out_dir:    Path,
) -> Dict:
    """
    Generates vegetation masks AND combined quality masks for
    all monthly images of ONE city.

    The combined mask is the primary product — it is passed to
    all downstream normalisation steps (Step 3).

    Called by run_pipeline.py → run_step2_masking()

    ARGS:
        upsamp_dir : upsampled images (step 1.3 output)
        cloud_dir  : cloud masks from step 2.1
        shadow_dir : shadow masks from step 2.2
        out_dir    : destination (vegetation + combined masks written here)
    """
    city = upsamp_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 2.3 : VEGETATION MASK + COMBINED QUALITY MASK")
    logger.info("=" * 60)
    logger.info(f"City     : {city}")
    logger.info(f"Method   : proxy-NDVI  (G-R)/(G+R)  threshold={VEG_THRESH}")
    logger.info(f"Output   : {out_dir}")

    tif_files: List[Path] = sorted(upsamp_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {upsamp_dir}")

    n_months = len(tif_files)
    logger.info(f"Time steps: {n_months} monthly images")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(tif_files):
        stem = img_path.stem

        # ── Load component masks ──────────────────────────────────────────
        cloud   = _load_binary_mask(cloud_dir  / f"{stem}_cloud.tif")
        shadow  = _load_binary_mask(shadow_dir / f"{stem}_shadow.tif")
        veg     = _compute_veg_mask(img_path)

        # ── Combine: any masking criterion → pixel is masked ──────────────
        combined = (cloud | shadow | veg.astype(bool)).astype(np.uint8)

        # ── Write outputs ─────────────────────────────────────────────────
        _write_mask(veg,      out_dir / f"{stem}_veg.tif",      img_path)
        _write_mask(combined, out_dir / f"{stem}_combined.tif",  img_path)

        veg_pct  = 100.0 * veg.mean()
        comb_pct = 100.0 * combined.mean()
        usable   = 100.0 - comb_pct
        logger.info(
            f"  [{idx+1}/{n_months}] {img_path.name[:40]}  "
            f"veg={veg_pct:.1f}%  combined={comb_pct:.1f}%  usable={usable:.1f}%"
        )

    logger.info("STATUS : Vegetation masking + combined quality mask complete ✓")
    logger.info("NEXT   : Step 3.1 → IR-MAD Normalisation")
    return {
        "city": city, "n_months": n_months,
        "output_dir": str(out_dir), "status": "complete"
    }


if __name__ == "__main__":
    run_vegetation_masking(
        upsamp_dir  = Path("data/processed/alignment_1/upsampled/L15-0358E-1220N_1433_3310_13"),
        cloud_dir   = Path("data/processed/masking_1/cloud/L15-0358E-1220N_1433_3310_13"),
        shadow_dir  = Path("data/processed/masking_1/shadow/L15-0358E-1220N_1433_3310_13"),
        out_dir     = Path("data/processed/masking_1/vegetation/L15-0358E-1220N_1433_3310_13"),
    )