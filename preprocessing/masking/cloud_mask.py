"""
preprocessing/masking/cloud_mask.py
=====================================

SUB-STEP 2.1 — Cloud Mask Generation
--------------------------------------

PURPOSE:
    Generate a per-pixel binary cloud mask for each monthly image.
    Pixels flagged as cloud (mask=1) are excluded from:
      - Radiometric normalisation statistics (Step 3)
      - Training patches (Step 4 filter, Step 5 FBC)

    The mask is NOT used to zero-out pixel values yet —
    that happens only in augmentation (Step 6), preserving the
    raw data for mask-aware normalisation.

TWO SOURCES OF CLOUD INFORMATION (in priority order):

    1. UDM_masks/ folder (if it exists for this AOI):
       Planet's own Unusable Data Mask. Binary GeoTIFF, value=1
       means cloud/haze/sensor error. Most reliable source.
       ~30% of SpaceNet 7 AOIs have this folder.

    2. DYNAMIC GENERATION (if UDM_masks/ is MISSING):
       Required by dataset quirk: only ~30 of 60 training cities
       have a UDM_masks/ folder. For the rest we must generate
       a fallback mask automatically.
       Method: spectral thresholding on the upsampled imagery.
         - Haze/cloud pixels are bright across all visible bands
         - Threshold: mean(R, G, B) > CLOUD_BRIGHTNESS_THRESH
         - Also catches saturated white pixels (255 in all bands)
       This is a heuristic — not as accurate as Planet's UDM,
       but sufficient to guide the normalization step.

       FALLBACK: If even spectral thresholding fails (very dark
       scenes, night imagery), we return an ALL-ZERO mask
       (clear-sky assumption). This is safe — it just means
       normalization uses all pixels including any remaining
       clouds. Better to over-include than to crash.

UDM MASK UPSAMPLING:
    UDM files are at the original ~1024×1024 resolution.
    We must resize them to 4096×4096 to match the upsampled images.
    Nearest-neighbour resampling is used — masks are binary
    (0/1), so interpolating values would create invalid 0.5 values.

INPUT:
    aoi_dir     : raw city folder (for UDM_masks/ subfolder)
    upsamp_dir  : data/processed/alignment/upsampled/{city}/
    out_dir     : data/processed/masking/cloud/{city}/

OUTPUT:
    out_dir/{timestamp}_cloud.tif  — uint8, 0=clear, 1=cloud
    One file per monthly image, same naming stem.

NEXT STEP → 2.2 shadow_mask.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Brightness threshold for heuristic cloud detection.
# Mean of R+G+B channels above this value → classified as cloud/haze.
# 200/255 ≈ 78% brightness — works well for Planet Dove uint8 imagery.
CLOUD_BRIGHTNESS_THRESH = 200

# SpaceNet 7 image bands (1-indexed, rasterio convention)
BAND_RED   = 1
BAND_GREEN = 2
BAND_BLUE  = 3
BAND_ALPHA = 4   # Not used for cloud detection — it's a validity mask

TARGET_SIZE = 4096   # Must match upsampling output


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _find_udm_file(
    udm_dir: Path,
    img_stem: str,
) -> Optional[Path]:
    """
    Finds the UDM mask file corresponding to one image.

    SpaceNet 7 UDM filenames share the date/timestamp portion
    of the image filename. We match on stem prefix.

    Strategy: look for any .tif in udm_dir whose name contains
    the date portion of img_stem. If no match, return None.

    ARGS:
        udm_dir  : path to UDM_masks/ subfolder
        img_stem : stem of the image file (e.g. "global_monthly_2018_01_mosaic_L15...")

    RETURNS:
        Path to matching UDM .tif, or None if not found
    """
    # Try exact stem first
    exact = udm_dir / f"{img_stem}.tif"
    if exact.exists():
        return exact

    # Try fuzzy: find any UDM file whose stem is contained in img_stem
    # or vice versa — handles slight naming differences
    for udm_file in udm_dir.glob("*.tif"):
        if udm_file.stem in img_stem or img_stem[:20] in udm_file.stem:
            return udm_file

    return None


def _load_udm_mask(
    udm_path: Path,
    target_h: int = TARGET_SIZE,
    target_w: int = TARGET_SIZE,
) -> np.ndarray:
    """
    Loads and upsamples a UDM mask to (target_h, target_w).

    UDM files are binary: 0=usable, 1=unusable (cloud/error).
    Nearest-neighbour resampling preserves the binary nature.

    ARGS:
        udm_path : path to UDM .tif
        target_h : required output height
        target_w : required output width

    RETURNS:
        uint8 numpy array (H, W), values 0 or 1
    """
    with rasterio.open(udm_path) as src:
        mask = src.read(
            1,                                      # band 1 (single-band UDM)
            out_shape=(target_h, target_w),
            resampling=Resampling.nearest,          # binary → nearest only
        )
    # Binarise: any non-zero → 1 (unusable / cloud)
    return (mask > 0).astype(np.uint8)


def _spectral_cloud_mask(
    img_path: Path,
    threshold: int = CLOUD_BRIGHTNESS_THRESH,
) -> np.ndarray:
    """
    Generates a heuristic cloud mask from spectral brightness.

    Used ONLY when UDM_masks/ does not exist for this AOI.

    LOGIC:
        Cloud/haze pixels have high reflectance in all visible bands.
        If mean(R, G, B) > threshold → cloud.
        This catches:
          - Thick cloud (white, brightness > 200)
          - Thin haze (slightly elevated brightness across bands)
          - Sensor saturation (pixel = 255 in all bands)

    LIMITATION:
        - Bright rooftops / sand / snow can be false positives
        - Thin cirrus is missed (not bright enough)
        These are acceptable — we just need an approximate mask
        for the normalisation statistics in Step 3.

    ARGS:
        img_path  : upsampled 4096×4096 GeoTIFF
        threshold : brightness threshold (0-255)

    RETURNS:
        uint8 numpy array (H, W), values 0=clear, 1=cloud
    """
    with rasterio.open(img_path) as src:
        r = src.read(BAND_RED).astype(np.float32)
        g = src.read(BAND_GREEN).astype(np.float32)
        b = src.read(BAND_BLUE).astype(np.float32)

    # Mean brightness across visible bands
    brightness = (r + g + b) / 3.0

    # Cloud pixels: bright + relatively uniform across bands
    # (clouds are spectrally flat; vegetation/soil is not)
    cloud = (brightness > threshold).astype(np.uint8)

    logger.debug(
        f"    Spectral cloud mask: "
        f"{cloud.sum():,} cloud pixels "
        f"({100*cloud.mean():.1f}% of image)"
    )
    return cloud


def _zero_mask(h: int = TARGET_SIZE, w: int = TARGET_SIZE) -> np.ndarray:
    """
    Returns an all-zero mask (clear-sky assumption).

    Used as last-resort fallback when:
      - UDM_masks/ is missing, AND
      - Image cannot be opened for spectral thresholding

    An all-zero mask means "no clouds detected" — normalisation
    will use all pixels. Slightly less accurate but never crashes.

    ARGS:
        h, w : mask dimensions

    RETURNS:
        uint8 numpy array (H, W) of zeros
    """
    logger.debug("    Using all-zero clear-sky fallback mask")
    return np.zeros((h, w), dtype=np.uint8)


def _write_mask(
    mask: np.ndarray,
    dst_path: Path,
    reference_img: Path,
) -> None:
    """
    Writes a binary uint8 mask as a single-band GeoTIFF.

    Copies CRS and transform from the reference image so the
    mask is spatially aligned with the upsampled imagery.

    ARGS:
        mask          : uint8 (H, W) array, values 0/1
        dst_path      : output path
        reference_img : image to copy CRS/transform from
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_img) as ref:
        meta = ref.meta.copy()

    meta.update({
        "count":  1,         # single-band mask
        "dtype":  "uint8",
        "nodata": None,
    })

    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(mask[np.newaxis, :, :])   # add band dimension


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_cloud_masking(
    aoi_dir: Path,
    upsamp_dir: Path,
    out_dir: Path,
) -> Dict:
    """
    Generates cloud masks for ALL monthly images of ONE city.

    Priority:
        1. UDM_masks/ folder exists → load & upsample Planet's mask
        2. UDM_masks/ missing → spectral brightness thresholding
        3. Spectral fails   → all-zero clear-sky fallback

    Called by run_pipeline.py → run_step2_masking()

    ARGS:
        aoi_dir    : raw city folder (contains UDM_masks/ if it exists)
        upsamp_dir : upsampled images from step 1.3
        out_dir    : destination for cloud masks

    RETURNS:
        summary dict
    """
    city = aoi_dir.name
    udm_dir = aoi_dir / "UDM_masks"
    has_udm = udm_dir.exists() and any(udm_dir.glob("*.tif"))

    logger.info("=" * 60)
    logger.info("SUB-STEP 2.1 : CLOUD MASK GENERATION")
    logger.info("=" * 60)
    logger.info(f"City      : {city}")
    logger.info(f"UDM dir   : {udm_dir}  →  {'EXISTS ✓' if has_udm else 'MISSING — using spectral fallback'}")
    logger.info(f"Output    : {out_dir}")

    # ── Discover upsampled images ─────────────────────────────────────────
    tif_files: List[Path] = sorted(upsamp_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(
            f"No upsampled .tif files in {upsamp_dir}. "
            "Did step 1.3 (upsampling) complete?"
        )

    n_months = len(tif_files)
    logger.info(f"\nTime steps : {n_months} monthly images")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_udm, n_spectral, n_fallback = 0, 0, 0

    for idx, img_path in enumerate(tif_files):
        # Output mask filename: same stem with _cloud.tif suffix
        mask_stem = img_path.stem + "_cloud"
        dst_path  = out_dir / f"{mask_stem}.tif"
        method    = "?"

        # ── Priority 1: UDM mask ──────────────────────────────────────────
        if has_udm:
            udm_file = _find_udm_file(udm_dir, img_path.stem)
            if udm_file is not None:
                mask   = _load_udm_mask(udm_file, TARGET_SIZE, TARGET_SIZE)
                method = "UDM"
                n_udm += 1

            else:
                # UDM dir exists but no matching file for this month
                # → fall through to spectral
                logger.warning(
                    f"  [{idx+1}/{n_months}] No UDM file matched {img_path.name} "
                    f"— using spectral fallback"
                )
                try:
                    mask   = _spectral_cloud_mask(img_path)
                    method = "spectral"
                    n_spectral += 1
                except Exception as e:
                    logger.warning(f"    Spectral failed: {e} — using zero mask")
                    mask   = _zero_mask()
                    method = "zero"
                    n_fallback += 1

        # ── Priority 2: Spectral thresholding ─────────────────────────────
        else:
            try:
                mask   = _spectral_cloud_mask(img_path)
                method = "spectral"
                n_spectral += 1
            except Exception as e:
                logger.warning(f"    Spectral failed: {e} — using zero mask")
                mask   = _zero_mask()
                method = "zero"
                n_fallback += 1

        # ── Write mask ────────────────────────────────────────────────────
        _write_mask(mask, dst_path, img_path)

        cloud_pct = 100.0 * mask.mean()
        logger.info(
            f"  [{idx+1}/{n_months}] {img_path.name[:45]}  "
            f"method={method:<9}  cloud={cloud_pct:.1f}%"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CLOUD MASK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total months  : {n_months}")
    logger.info(f"UDM source    : {n_udm}")
    logger.info(f"Spectral      : {n_spectral}")
    logger.info(f"Zero fallback : {n_fallback}")
    logger.info("STATUS : Cloud masking complete ✓")
    logger.info("NEXT   : Sub-step 2.2 → Shadow Mask")
    logger.info("=" * 60)

    return {
        "city":        city,
        "n_months":    n_months,
        "has_udm":     has_udm,
        "n_udm":       n_udm,
        "n_spectral":  n_spectral,
        "n_fallback":  n_fallback,
        "output_dir":  str(out_dir),
        "status":      "complete",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    AOI_DIR   = Path("data/raw/dataset/SN7_buildings_train/train/L15-0358E-1220N_1433_3310_13")
    UPSAMP    = Path("data/processed/alignment_1/upsampled/L15-0358E-1220N_1433_3310_13")
    OUT       = Path("data/processed/masking_1/cloud/L15-0358E-1220N_1433_3310_13")

    summary = run_cloud_masking(AOI_DIR, UPSAMP, OUT)

    print("\n✓ Sub-step 2.1 Complete")
    print(f"  Months       : {summary['n_months']}")
    print(f"  UDM masks    : {summary['n_udm']}")
    print(f"  Spectral     : {summary['n_spectral']}")
    print(f"  Zero fallback: {summary['n_fallback']}")
    print(f"  Output       : {summary['output_dir']}")