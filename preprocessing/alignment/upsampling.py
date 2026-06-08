"""
preprocessing/alignment/upsampling.py
=======================================

SUB-STEP 1.3 — 4× Bicubic Upsampling → Force 4096×4096
---------------------------------------------------------

PURPOSE:
    Upsample every coregistered monthly image from ~1024×1024
    to exactly 4096×4096 pixels so that:
      1. 256×256 tiling in Step 4 produces exactly 256 non-overlapping
         patches per image (4096 / 256 = 16 per axis = 256 patches)
      2. The 1023×1024 vs 1024×1024 dimension mismatch is eliminated
         permanently — every downstream file is exactly 4096×4096

THE DIMENSION MISMATCH PROBLEM:
    SpaceNet 7 images should be 1024×1024 but some AOIs deliver
    1023×1024 (one row short). This is a known dataset quirk.
    Our fix: pad-then-resize BOTH dimensions to exactly 4×1024 = 4096.
    Padding adds a 1-pixel-wide strip of zeros (mirrored edge),
    then bicubic resize brings everything to 4096×4096.
    The 1-pixel pad artefact is negligible after bicubic smoothing.

WHY BICUBIC:
    - Better than bilinear for preserving edges in urban imagery
    - Better than nearest-neighbour for smooth gradients
    - Standard for super-resolution upsampling in remote sensing

WHY 4096×4096 (MANDATORY):
    4096 = 256 × 16 — exactly 256 patches per axis, no leftover strips.
    Any other size would require overlap-based tiling (complex) or
    would discard partial patches (wastes data).

STORAGE WARNING:
    A uint8 4-band 4096×4096 image = 4096 × 4096 × 4 bytes = 67 MB.
    24 months × 67 MB = ~1.6 GB per city.
    These INTERMEDIATE files are automatically DELETED by run_pipeline.py
    cleanup_city() after Step 4 (tiling) completes.

INPUT:
    data/processed/alignment/coregistered/{city}/*.tif  (~1024×1024)

OUTPUT:
    data/processed/alignment/upsampled/{city}/*.tif  (4096×4096)

NEXT STEP → 2.1 cloud_mask.py
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SIZE  = 4096          # Mandatory output size (both H and W)
UPSAMPLE_FACTOR = 4          # 1024 × 4 = 4096 (nominal)


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pad_to_square(
    data: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Zero-pads `data` to exactly (target_h, target_w) if smaller.

    HANDLES 1023×1024 → 1024×1024 MISMATCH:
        If H=1023 and target_h=1024, we add 1 row of zeros at bottom.
        If W=1023 and target_w=1024, we add 1 col of zeros at right.
        This preserves all real pixels; the padding strip is negligible
        after bicubic upsampling.

    DOES NOT CROP:
        If data is already >= target size we leave it unchanged.
        A log warning is issued so you can investigate.

    ARGS:
        data     : numpy array (bands, H, W) or (H, W)
        target_h : required height
        target_w : required width

    RETURNS:
        numpy array padded to at least (target_h, target_w)
    """
    if data.ndim == 2:
        h, w = data.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        return np.pad(data, ((0, pad_h), (0, pad_w)), mode="edge")
    else:
        # (bands, H, W)
        _, h, w = data.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        return np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")


def _upsample_image(
    src_path: Path,
    dst_path: Path,
    target_size: int = TARGET_SIZE,
) -> Dict:
    """
    Upsamples one GeoTIFF to (target_size × target_size) pixels.

    PIPELINE:
        1. Open source image, read all bands as uint8
        2. Pad if H or W < 1024 (fixes 1023×1024 issue)
        3. Write to dst with updated width/height/transform
           using rasterio's built-in Resampling.cubic
           (rasterio handles the actual cubic resize internally
            when width/height differ from the source)

    NOTE ON TRANSFORM UPDATE:
        When we change width from 1024 to 4096, pixel size changes
        from ~4.78 m to ~4.78/4 = ~1.195 m (better resolution).
        We update the affine transform so geospatial metadata stays valid.

    ARGS:
        src_path    : input GeoTIFF (~1024×1024)
        dst_path    : output GeoTIFF (4096×4096)
        target_size : pixel size for both H and W (default 4096)

    RETURNS:
        dict with original shape, padded shape, output shape, resolution
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        orig_h, orig_w = src.height, src.width
        orig_res = src.res[0]          # metres per pixel (before upsampling)
        meta = src.meta.copy()

        # ── Step 1: Read all bands ────────────────────────────────────────
        data = src.read()              # shape: (bands, H, W)
        orig_transform = src.transform

        # ── Step 2: Pad to 1024×1024 if mismatched ───────────────────────
        # This fixes the 1023×1024 dataset quirk
        BASE = 1024                    # nominal SpaceNet 7 size
        pad_h = max(0, BASE - orig_h)
        pad_w = max(0, BASE - orig_w)
        if pad_h > 0 or pad_w > 0:
            data = _pad_to_square(data, BASE, BASE)
            logger.debug(
                f"  Padded {src_path.name}: "
                f"({orig_h},{orig_w}) → ({BASE},{BASE})"
            )
        padded_h, padded_w = data.shape[1], data.shape[2]

        # ── Step 3: Update metadata for 4096×4096 output ─────────────────
        # New pixel size = old_pixel_size × (padded_dim / target_dim)
        # = ~4.78 × (1024/4096) = ~1.195 m/pixel
        scale_x = padded_w / target_size
        scale_y = padded_h / target_size

        new_transform = Affine(
            orig_transform.a * scale_x,    # new pixel width (m)
            orig_transform.b,              # rotation (0)
            orig_transform.c,              # x origin unchanged
            orig_transform.d,              # rotation (0)
            orig_transform.e * scale_y,    # new pixel height (m, negative)
            orig_transform.f,              # y origin unchanged
        )

        meta.update({
            "width":     target_size,
            "height":    target_size,
            "transform": new_transform,
        })

        # ── Step 4: Write with bicubic resampling ─────────────────────────
        # We write a temporary in-memory array of the padded data,
        # then use rasterio's resampling when reading at target_size.
        # The cleanest way: write padded → reopen → read at target resolution.

        import tempfile, shutil, os
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Write padded (intermediate)
        padded_meta = src.meta.copy()
        padded_meta.update({
            "width":  padded_w,
            "height": padded_h,
        })
        with rasterio.open(tmp_path, "w", **padded_meta) as tmp_ds:
            tmp_ds.write(data)

        # Re-read at target_size using bicubic — rasterio handles the math
        upsample_meta = padded_meta.copy()
        upsample_meta.update({
            "width":     target_size,
            "height":    target_size,
            "transform": new_transform,
        })
        with rasterio.open(tmp_path) as tmp_ds:
            upsampled = tmp_ds.read(
                out_shape=(
                    tmp_ds.count,
                    target_size,
                    target_size,
                ),
                resampling=Resampling.cubic,   # Bicubic
            )

        os.unlink(tmp_path)   # delete temp file

        # Write final output
        with rasterio.open(dst_path, "w", **upsample_meta) as dst:
            dst.write(upsampled)

        new_res = round(orig_res * (padded_w / target_size), 4)
        return {
            "file":        src_path.name,
            "orig_shape":  (orig_h, orig_w),
            "padded_shape": (padded_h, padded_w),
            "out_shape":   (target_size, target_size),
            "orig_res_m":  round(orig_res, 4),
            "new_res_m":   new_res,
            "bands":       meta["count"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_upsampling(
    coreg_dir: Path,
    out_dir: Path,
    target_size: int = TARGET_SIZE,
) -> Dict:
    """
    Upsamples all monthly images for ONE city to target_size×target_size.

    Called by run_pipeline.py → run_step1_alignment()

    ARGS:
        coreg_dir   : output of step 1.2
                      e.g. data/processed/alignment/coregistered/{city}
        out_dir     : destination for upsampled images
                      e.g. data/processed/alignment/upsampled/{city}
        target_size : output pixel dimensions (default 4096)

    RETURNS:
        summary dict
    """
    city = coreg_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 1.3 : UPSAMPLING  (4× Bicubic → 4096×4096)")
    logger.info("=" * 60)
    logger.info(f"City        : {city}")
    logger.info(f"Source      : {coreg_dir}")
    logger.info(f"Target size : {target_size} × {target_size} (MANDATORY)")
    logger.info(f"Upsample    : ~4× (1024 → 4096)")
    logger.info(f"Output      : {out_dir}")
    logger.info(
        f"⚠  STORAGE  : ~{target_size*target_size*4/1e6:.0f} MB per image "
        f"— DELETED after Step 4 tiling"
    )

    # ── Discover monthly images ───────────────────────────────────────────
    tif_files: List[Path] = sorted(coreg_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif files in {coreg_dir}. "
            "Did step 1.2 (coregistration) complete successfully?"
        )

    n_months = len(tif_files)
    logger.info(f"\nTime steps : {n_months} monthly images")

    # ── Upsample each month ───────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []
    n_padded = 0

    for idx, tif_path in enumerate(tif_files):
        dst_path = out_dir / tif_path.name
        result = _upsample_image(tif_path, dst_path, target_size)
        results.append(result)

        if result["orig_shape"] != (1024, 1024):
            n_padded += 1
            logger.info(
                f"  [{idx+1}/{n_months}] {tif_path.name}  "
                f"orig={result['orig_shape']} → "
                f"pad={result['padded_shape']} → "
                f"out={result['out_shape']}  [PADDED]"
            )
        else:
            logger.info(
                f"  [{idx+1}/{n_months}] {tif_path.name}  "
                f"1024×1024 → 4096×4096"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    first = results[0]
    approx_mb_per_file = target_size * target_size * first["bands"] / 1e6
    total_gb = approx_mb_per_file * n_months / 1e3

    logger.info("\n" + "=" * 60)
    logger.info("UPSAMPLING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images      : {n_months}")
    logger.info(f"Padded (was <1024): {n_padded}")
    logger.info(f"Output shape      : {target_size} × {target_size}")
    logger.info(f"Original res      : ~{first['orig_res_m']} m/pixel")
    logger.info(f"New resolution    : ~{first['new_res_m']} m/pixel")
    logger.info(f"Bands             : {first['bands']} (R, G, B, Alpha)")
    logger.info(f"Disk usage (est.) : {total_gb:.2f} GB  [TEMPORARY]")
    logger.info("STATUS : Upsampling complete ✓")
    logger.info("NEXT   : Step 2.1 → Cloud Mask generation")
    logger.info("=" * 60)

    return {
        "city":          city,
        "n_months":      n_months,
        "n_padded":      n_padded,
        "target_size":   target_size,
        "bands":         first["bands"],
        "orig_res_m":    first["orig_res_m"],
        "new_res_m":     first["new_res_m"],
        "disk_gb_est":   round(total_gb, 2),
        "output_dir":    str(out_dir),
        "status":        "complete",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    COREG_DIR = Path(
        "data/processed/alignment_1/coregistered/"
        "L15-0358E-1220N_1433_3310_13"
    )
    OUT_DIR = Path(
        "data/processed/alignment_1/upsampled/"
        "L15-0358E-1220N_1433_3310_13"
    )

    summary = run_upsampling(COREG_DIR, OUT_DIR)

    print("\n✓ Sub-step 1.3 Complete")
    print(f"  Time steps    : {summary['n_months']}")
    print(f"  Padded images : {summary['n_padded']}")
    print(f"  Output shape  : {summary['target_size']} × {summary['target_size']}")
    print(f"  Resolution    : {summary['new_res_m']} m/pixel")
    print(f"  Disk (est.)   : {summary['disk_gb_est']} GB  (temporary)")
    print(f"  Output dir    : {summary['output_dir']}")
