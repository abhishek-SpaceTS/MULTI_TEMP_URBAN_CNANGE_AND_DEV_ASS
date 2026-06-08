"""
preprocessing/alignment/coregistration.py
==========================================

SUB-STEP 1.2 — Temporal Coregistration
----------------------------------------

PURPOSE:
    Align all monthly images for one city so that the same
    physical pixel location maps to the same ground point
    across ALL time steps. This is essential for the UNet-GFAM
    model to learn temporal differences rather than camera-shift
    artefacts.

WHY THIS MATTERS:
    Planet Dove satellites produce slightly different footprints
    each month due to orbital variation. Two consecutive months
    of the same city can be misaligned by 1-3 pixels. Over 24
    months this accumulates and creates false "change" signals
    that confuse the model.

METHOD — NCC Phase Correlation:
    1. Pick a reference image (month 0 — first temporally)
    2. For every other month:
       a. Compute normalised cross-correlation (NCC) in frequency
          domain (fast Fourier approach via skimage)
       b. Estimate the (row_shift, col_shift) in pixels
       c. Apply the shift using an affine translation transform
          (rasterio warp) — nearest-neighbour for speed, since
          bicubic upsampling happens next in step 1.3 anyway

LIMITATION / FUTURE WORK:
    Phase correlation handles pure translation only (x/y shift).
    Rotation/scale/shear (rare in Planet mosaics) would need
    a full feature-point method (e.g. SIFT + RANSAC). For
    SpaceNet 7 the translation-only approach is sufficient.

VARIABLE TIME STEPS:
    SpaceNet 7 AOIs have 18-25 monthly images.
    We NEVER hardcode the count — we discover it dynamically
    from whatever .tif files exist in the input directory.

INPUT:
    data/processed/alignment/reprojected/{city}/*.tif
    (output from step 1.1 — guaranteed EPSG:3857)

OUTPUT:
    data/processed/alignment/coregistered/{city}/*.tif
    Same shape/dtype/CRS as input, shifted to align with month 0.
    Month 0 itself is copied unchanged (it IS the reference).

NEXT STEP → 1.3 upsampling.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from skimage.registration import phase_cross_correlation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Band used for NCC computation.
# Band 1 (Red) has good contrast for urban scenes.
# Alpha (Band 4) is excluded — it is a mask, not imagery.
COREG_BAND = 1          # 1-indexed (rasterio convention)

# Maximum allowed shift in pixels. Shifts larger than this are
# almost certainly wrong (image was re-tiled, not just shifted).
MAX_SHIFT_PIXELS = 30


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_band(img_path: Path, band: int = COREG_BAND) -> np.ndarray:
    """
    Loads a single band as float32 for NCC computation.

    We convert to float32 here so the cross-correlation math
    is stable regardless of the source dtype (uint8 → 0-255 range).

    ARGS:
        img_path : path to GeoTIFF
        band     : 1-indexed band number

    RETURNS:
        2-D numpy array (H, W) float32
    """
    with rasterio.open(img_path) as src:
        data = src.read(band).astype(np.float32)
    return data


def _estimate_shift(
    ref: np.ndarray,
    mov: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimates (row_shift, col_shift) needed to align `mov` → `ref`.

    Uses skimage.registration.phase_cross_correlation which
    computes the normalised cross-power spectrum (frequency-domain
    NCC). This is O(N log N) — much faster than spatial NCC.

    A positive row_shift means `mov` is shifted DOWN relative to
    `ref`; we need to shift it UP to align.

    ARGS:
        ref : reference image band (H, W) float32
        mov : moving image band (H, W) float32

    RETURNS:
        (row_shift, col_shift) in pixels — possibly sub-pixel floats
    """
    # upsample_factor=1 → integer pixel precision (sufficient for us)
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=1)
    row_shift, col_shift = float(shift[0]), float(shift[1])
    return row_shift, col_shift


def _apply_shift(
    src_path: Path,
    dst_path: Path,
    row_shift: float,
    col_shift: float,
) -> None:
    """
    Applies a pixel-level translation (row_shift, col_shift) to
    every band of a GeoTIFF by updating the affine transform.

    WHY UPDATE THE TRANSFORM rather than rolling the array:
        Rolling the pixel array loses the geospatial metadata.
        Instead we shift the affine transform — downstream steps
        (rasterio.warp.reproject) then handle the actual pixel
        re-gridding correctly.

    The new transform shifts the origin by:
        col_shift × pixel_width_in_metres  (x direction)
        row_shift × pixel_height_in_metres (y direction — negative in rasterio)

    ARGS:
        src_path  : source GeoTIFF
        dst_path  : output GeoTIFF (same shape/dtype/CRS)
        row_shift : rows to shift (+ = image shifts down)
        col_shift : cols to shift (+ = image shifts right)
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        old_tf = src.transform

        # Build translation matrix
        # Affine has (x_pixel_size, 0, x_origin,
        #             0, y_pixel_size, y_origin)
        # We shift origin by col_shift pixels in x and row_shift in y
        new_tf = Affine(
            old_tf.a,                                     # pixel width (m)
            old_tf.b,                                     # rotation (0)
            old_tf.c + col_shift * old_tf.a,              # new x origin
            old_tf.d,                                     # rotation (0)
            old_tf.e,                                     # pixel height (m, negative)
            old_tf.f + row_shift * old_tf.e,              # new y origin
        )
        meta["transform"] = new_tf

        with rasterio.open(dst_path, "w", **meta) as dst:
            # Copy all bands with the new transform metadata
            for band_idx in range(1, src.count + 1):
                dst.write(src.read(band_idx), band_idx)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_coregistration(
    reprojected_dir: Path,
    out_dir: Path,
) -> Dict:
    """
    Coregisters all monthly images for ONE city.

    Picks month 0 as the fixed reference, then aligns every
    subsequent month to it via NCC phase correlation.

    Called by run_pipeline.py → run_step1_alignment()

    ARGS:
        reprojected_dir : output of step 1.1
                          e.g. data/processed/alignment/reprojected/{city}
        out_dir         : destination for coregistered images
                          e.g. data/processed/alignment/coregistered/{city}

    RETURNS:
        summary dict with shifts for every month
    """
    city = reprojected_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 1.2 : COREGISTRATION  (NCC Phase Correlation)")
    logger.info("=" * 60)
    logger.info(f"City    : {city}")
    logger.info(f"Source  : {reprojected_dir}")
    logger.info(f"Output  : {out_dir}")

    # ── Discover monthly images ───────────────────────────────────────────
    # Variable time steps: 18-25 months depending on AOI
    # Sorted alphabetically = chronological (filenames contain date)
    tif_files: List[Path] = sorted(reprojected_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif files in {reprojected_dir}. "
            "Did step 1.1 (CRS reprojection) complete successfully?"
        )

    n_months = len(tif_files)
    logger.info(f"\nTime steps : {n_months} monthly images (dynamic — no hardcoding)")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load reference image (month 0) ────────────────────────────────────
    ref_path = tif_files[0]
    ref_band = _load_band(ref_path, COREG_BAND)
    logger.info(f"Reference  : {ref_path.name}  (month 0, fixed)")
    logger.info(f"Ref shape  : {ref_band.shape}")

    # Copy reference unchanged — it IS the reference, no shift needed
    dst_ref = out_dir / ref_path.name
    import shutil
    shutil.copy2(ref_path, dst_ref)
    logger.info(f"  [0/{n_months-1}] {ref_path.name}  → shift=(0.0, 0.0)  [REFERENCE]")

    # ── Align every subsequent month ─────────────────────────────────────
    shift_log: List[Dict] = [{"file": ref_path.name, "row_shift": 0.0, "col_shift": 0.0, "status": "reference"}]
    n_large_shift = 0

    for idx, tif_path in enumerate(tif_files[1:], start=1):
        mov_band = _load_band(tif_path, COREG_BAND)
        row_shift, col_shift = _estimate_shift(ref_band, mov_band)

        # ── Sanity check: reject implausibly large shifts ─────────────────
        # A shift > MAX_SHIFT_PIXELS almost certainly indicates that the
        # images are from different tiles, not just camera offset.
        shift_mag = float(np.hypot(row_shift, col_shift))
        if shift_mag > MAX_SHIFT_PIXELS:
            logger.warning(
                f"  [{idx}/{n_months-1}] {tif_path.name}  "
                f"LARGE SHIFT ({shift_mag:.1f}px) — clamping to zero "
                f"(copy without shift)"
            )
            row_shift, col_shift = 0.0, 0.0
            n_large_shift += 1

        dst_path = out_dir / tif_path.name
        _apply_shift(tif_path, dst_path, row_shift, col_shift)

        status = "shifted" if (row_shift != 0 or col_shift != 0) else "no_shift"
        shift_log.append({
            "file":       tif_path.name,
            "row_shift":  round(row_shift, 2),
            "col_shift":  round(col_shift, 2),
            "shift_mag":  round(shift_mag, 2),
            "status":     status,
        })
        logger.info(
            f"  [{idx}/{n_months-1}] {tif_path.name}  "
            f"→ shift=({row_shift:.1f}, {col_shift:.1f})px  [{status}]"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    n_shifted = sum(1 for s in shift_log if s.get("status") == "shifted")
    logger.info("\n" + "=" * 60)
    logger.info("COREGISTRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total time steps : {n_months}")
    logger.info(f"Shifted          : {n_shifted}")
    logger.info(f"No shift needed  : {n_months - n_shifted - 1}")
    logger.info(f"Large shift warn : {n_large_shift}  (copied without shift)")
    logger.info("STATUS : Coregistration complete ✓")
    logger.info("NEXT   : Sub-step 1.3 → Upsampling")
    logger.info("=" * 60)

    return {
        "city":          city,
        "n_months":      n_months,
        "n_shifted":     n_shifted,
        "n_large_shift": n_large_shift,
        "shifts":        shift_log,
        "output_dir":    str(out_dir),
        "status":        "complete",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    REPROJ_DIR = Path(
        "data/processed/alignment_1/reprojected/"
        "L15-0358E-1220N_1433_3310_13"
    )
    OUT_DIR = Path(
        "data/processed/alignment_1/coregistered/"
        "L15-0358E-1220N_1433_3310_13"
    )

    summary = run_coregistration(REPROJ_DIR, OUT_DIR)

    print("\n✓ Sub-step 1.2 Complete")
    print(f"  Time steps   : {summary['n_months']}")
    print(f"  Shifted      : {summary['n_shifted']}")
    print(f"  Large shifts : {summary['n_large_shift']}")
    print(f"  Output       : {summary['output_dir']}")