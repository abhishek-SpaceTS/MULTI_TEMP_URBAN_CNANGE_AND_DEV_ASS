"""
preprocessing/normalization/irmad.py
======================================

SUB-STEP 3.1 — IR-MAD Radiometric Normalisation
-------------------------------------------------

PURPOSE:
    Correct for radiometric differences between monthly images
    caused by atmospheric conditions, sun angle, and sensor
    response variation — using ONLY pixels not masked by
    cloud/shadow/vegetation (combined mask from Step 2).

WHY RADIOMETRIC NORMALISATION:
    Planet Dove produces "visually consistent" mosaics but does
    NOT guarantee calibrated reflectance across months. An image
    of the same rooftop in January vs July can differ by 30-50
    counts in uint8 (0-255) due to:
      - Atmospheric haze (variable per month)
      - Sun elevation angle (different shadow length)
      - Bi-directional reflectance effects (sun angle)
    Without normalisation, the model learns "it is winter"
    instead of "this building is new".

WHAT IS IR-MAD:
    Iteratively Re-weighted Multivariate Alteration Detection.
    A statistical method that finds "no-change" pixels between
    two images using Principal Component Analysis on the
    difference image, then downweights probable-change pixels
    in each iteration.

    Reference:
        Nielsen, A.A. (2007). The Regularized Iteratively
        Reweighted MAD Method for Change Detection in
        Multi- and Hyperspectral Data. IEEE Trans. Image Process.

SIMPLIFIED IMPLEMENTATION:
    Full IR-MAD requires iterative eigenvector decomposition.
    We implement the core statistical correction it produces:
      1. Identify invariant pixels (low change probability)
         using the combined quality mask
      2. Fit per-band linear regression: target = a * source + b
         (where target = reference month, source = moving month)
      3. Apply to ALL pixels of the moving image

    The result is equivalent to IR-MAD's final output when
    change is sparse (urban areas where most pixels are stable).

MASK-AWARE ARCHITECTURE (CRITICAL):
    Statistics (mean, std) are computed ONLY on pixels where
    combined_mask == 0 (usable, not cloud/shadow/vegetation).
    The linear correction is applied to ALL pixels — even masked
    ones — so their values remain plausible after masking is
    removed during augmentation.

    DO NOT zero out masked pixels here — that is Step 6's job.

INPUT:
    upsamp_dir : data/processed/alignment/upsampled/{city}/*.tif
    mask_dir   : data/processed/masking/vegetation/{city}/*_combined.tif

OUTPUT:
    data/processed/normalization/irmad/{city}/*.tif
    Same shape/dtype as input, radiometrically corrected.

NEXT STEP → 3.2 pif_regression.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Bands to normalise (R, G, B) — Alpha is a validity flag, not normalised
NORM_BANDS = [1, 2, 3]   # 1-indexed (rasterio convention)

# Minimum number of usable pixels required to compute reliable statistics.
# If fewer are available (very cloudy month), we skip normalisation for that image.
MIN_USABLE_PIXELS = 10_000   # ~6% of 4096×4096


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_combined_mask(mask_dir: Path, img_stem: str) -> Optional[np.ndarray]:
    """
    Loads the combined quality mask for one image.

    Combined mask: 0=usable, 1=cloud/shadow/vegetation.
    Returns None if mask not found (caller treats as all-zero / all-usable).
    """
    mask_path = mask_dir / f"{img_stem}_combined.tif"
    if not mask_path.exists():
        logger.warning(f"    Combined mask not found for {img_stem} — using all pixels")
        return None
    with rasterio.open(mask_path) as src:
        return src.read(1).astype(np.uint8)


def _compute_band_stats(
    data: np.ndarray,
    usable: np.ndarray,
) -> Tuple[float, float]:
    """
    Computes mean and std of `data` using only `usable` pixels.

    ARGS:
        data   : (H, W) float32 array of pixel values
        usable : (H, W) boolean array, True = usable pixel

    RETURNS:
        (mean, std) — or (0, 1) if too few usable pixels
    """
    vals = data[usable]
    if vals.size < 10:
        return 0.0, 1.0   # degenerate fallback
    return float(vals.mean()), float(vals.std()) + 1e-8


def _linear_correction(
    data: np.ndarray,
    src_mean: float,
    src_std: float,
    ref_mean: float,
    ref_std: float,
) -> np.ndarray:
    """
    Applies histogram-matching linear correction:
        corrected = (data - src_mean) / src_std * ref_std + ref_mean

    This is a z-score normalisation followed by rescaling to the
    reference distribution — equivalent to the linear term of IR-MAD.

    Output is clamped to [0, 255] and rounded to uint8.

    ARGS:
        data     : (H, W) float32 band data
        src_mean : mean of source (moving) image (usable pixels only)
        src_std  : std  of source (moving) image (usable pixels only)
        ref_mean : mean of reference image (usable pixels only)
        ref_std  : std  of reference image (usable pixels only)

    RETURNS:
        uint8 (H, W) corrected band
    """
    corrected = (data - src_mean) / src_std * ref_std + ref_mean
    corrected  = np.clip(corrected, 0, 255)
    return corrected.astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_irmad_normalisation(
    upsamp_dir: Path,
    mask_dir:   Path,
    out_dir:    Path,
) -> Dict:
    """
    Runs mask-aware linear radiometric normalisation for ONE city.

    Uses month 0 as the radiometric reference. All other months
    are shifted to match its per-band histogram on usable pixels.

    Called by run_pipeline.py → run_step3_normalisation()

    ARGS:
        upsamp_dir : upsampled images (step 1.3 output)
        mask_dir   : combined quality masks (step 2.3 output)
        out_dir    : destination for normalised images
    """
    city = upsamp_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 3.1 : IR-MAD RADIOMETRIC NORMALISATION")
    logger.info("=" * 60)
    logger.info(f"City      : {city}")
    logger.info(f"Method    : Mask-aware linear histogram matching")
    logger.info(f"Ref month : 0  (first temporally)")
    logger.info(f"Bands     : R, G, B  (Alpha not normalised)")

    tif_files: List[Path] = sorted(upsamp_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {upsamp_dir}")

    n_months = len(tif_files)
    logger.info(f"Time steps: {n_months} monthly images (dynamic)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Reference image: month 0 ──────────────────────────────────────────
    ref_path = tif_files[0]
    ref_mask_raw = _load_combined_mask(mask_dir, ref_path.stem)

    with rasterio.open(ref_path) as ref_src:
        ref_meta   = ref_src.meta.copy()
        ref_data   = {b: ref_src.read(b).astype(np.float32) for b in NORM_BANDS}
        ref_alpha  = ref_src.read(4)   # preserve Alpha band unchanged

    ref_usable = (ref_mask_raw == 0) if ref_mask_raw is not None \
                 else np.ones(ref_data[1].shape, dtype=bool)

    # Compute reference statistics
    ref_stats = {}
    for b in NORM_BANDS:
        mu, sigma = _compute_band_stats(ref_data[b], ref_usable)
        ref_stats[b] = (mu, sigma)
        logger.info(f"  Reference Band {b}: mean={mu:.1f}  std={sigma:.1f}  "
                    f"usable_px={ref_usable.sum():,}")

    # Copy reference month unchanged (it IS the reference)
    import shutil
    shutil.copy2(ref_path, out_dir / ref_path.name)
    logger.info(f"  [1/{n_months}] {ref_path.name[:45]}  [REFERENCE — copied]")

    # ── Normalise remaining months ────────────────────────────────────────
    n_skipped = 0

    for idx, tif_path in enumerate(tif_files[1:], start=2):
        mask_raw = _load_combined_mask(mask_dir, tif_path.stem)

        with rasterio.open(tif_path) as src:
            meta  = src.meta.copy()
            alpha = src.read(4)

            usable = (mask_raw == 0) if mask_raw is not None \
                     else np.ones((src.height, src.width), dtype=bool)
            n_usable = int(usable.sum())

            # ── Skip if too few usable pixels (very cloudy) ───────────────
            if n_usable < MIN_USABLE_PIXELS:
                logger.warning(
                    f"  [{idx}/{n_months}] {tif_path.name[:40]}  "
                    f"SKIPPED — only {n_usable:,} usable px "
                    f"(min={MIN_USABLE_PIXELS:,}) — copied as-is"
                )
                shutil.copy2(tif_path, out_dir / tif_path.name)
                n_skipped += 1
                continue

            # ── Normalise each visible band ───────────────────────────────
            corrected_bands = {}
            shift_log = []

            for b in NORM_BANDS:
                band_data = src.read(b).astype(np.float32)
                src_mu, src_sigma = _compute_band_stats(band_data, usable)
                ref_mu, ref_sigma = ref_stats[b]

                corrected = _linear_correction(
                    band_data, src_mu, src_sigma, ref_mu, ref_sigma
                )
                corrected_bands[b] = corrected
                shift_log.append(f"B{b}:({src_mu:.0f}→{ref_mu:.0f})")

        # ── Write normalised image ────────────────────────────────────────
        dst_path = out_dir / tif_path.name
        with rasterio.open(dst_path, "w", **meta) as dst:
            for b in NORM_BANDS:
                dst.write(corrected_bands[b], b)
            dst.write(alpha, 4)    # Alpha band unchanged

        logger.info(
            f"  [{idx}/{n_months}] {tif_path.name[:40]}  "
            f"usable={n_usable:,}  shifts={', '.join(shift_log)}"
        )

    logger.info("STATUS : IR-MAD normalisation complete ✓")
    logger.info("NEXT   : Sub-step 3.2 → PIF Regression")
    return {
        "city": city, "n_months": n_months, "n_skipped": n_skipped,
        "output_dir": str(out_dir), "status": "complete"
    }


if __name__ == "__main__":
    run_irmad_normalisation(
        upsamp_dir = Path("data/processed/alignment/upsampled/L15-0331E-1257N_1327_3160_13"),
        mask_dir   = Path("data/processed/masking/vegetation/L15-0331E-1257N_1327_3160_13"),
        out_dir    = Path("data/processed/normalization/irmad/L15-0331E-1257N_1327_3160_13"),
    )
