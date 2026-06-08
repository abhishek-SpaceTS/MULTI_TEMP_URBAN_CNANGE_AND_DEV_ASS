"""
preprocessing/alignment/crs_reprojection.py
============================================

SUB-STEP 1.1 — CRS Reprojection
---------------------------------

PURPOSE:
    Verify that every image in images_masked/ is in EPSG:3857
    (Web Mercator). If not, reproject it. Copy compliant images
    as-is so the next step always receives EPSG:3857 files.

WHY images_masked/ AND NOT images/:
    Planet Labs applied their proprietary cloud-removal pass to
    produce images_masked/. Our own cloud/shadow/vegetation masks
    (Step 2) build on top of that cleaner base.
    We never go back to the rawer images/ folder.

WHY EPSG:3857:
    - Planet Dove imagery ships in EPSG:3857 (Web Mercator)
    - All downstream pixel-level ops (coreg, upsampling, tiling)
      must work in a consistent metric CRS (units = metres)
    - GeoJSON labels/ are in EPSG:4326 (GPS degrees) — they are
      NOT touched here; they are consumed only in Step 5

SPACENET 7 REALITY (from dataset inspection):
    images_masked/ → already EPSG:3857 for the training split
    Resolution     → ~4.78 m/pixel
    Shape          → 1023×1024 or 1024×1024  (handled in step 1.3)
    Bands          → 4  (R, G, B, Alpha)
    Dtype          → uint8

    For production Planet raw imagery: may be UTM or WGS84 —
    auto-reprojected here before anything else runs.

OUTPUT:
    data/processed/alignment/reprojected/{city}/*.tif
    Identical to input if already EPSG:3857 (fast copy).
    Reprojected if not (bilinear resampling).

NEXT STEP → 1.2 coregistration.py
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, List

import rasterio
from rasterio.crs import CRS
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_CRS = "EPSG:3857"          # Web Mercator — our pipeline standard
IMAGE_SUBDIR = "images_masked"    # Always use masked images, never raw images/


# ══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _read_image_meta(img_path: Path) -> Dict:
    """
    Opens one GeoTIFF and returns a dict of key metadata.

    No pixel data is loaded — rasterio reads only the header.
    This is fast even for large files.

    RETURNS:
        dict with crs, epsg, shape (H,W), bands, resolution_m,
        dtype, bounds, needs_reprojection flag
    """
    with rasterio.open(img_path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        return {
            "file":               img_path.name,
            "crs":                str(src.crs),
            "epsg":               epsg,
            "shape":              (src.height, src.width),
            "bands":              src.count,
            "resolution_m":      round(src.res[0], 4),
            "dtype":              src.dtypes[0],
            "bounds": {
                "left":   round(src.bounds.left,   2),
                "right":  round(src.bounds.right,  2),
                "top":    round(src.bounds.top,     2),
                "bottom": round(src.bounds.bottom,  2),
            },
            # True only if CRS is NOT already the target
            "needs_reprojection": src.crs != CRS.from_string(TARGET_CRS),
        }


def _reproject_image(
    src_path: Path,
    dst_path: Path,
    target_crs: str = TARGET_CRS,
) -> Dict:
    """
    Reprojects a single GeoTIFF to target_crs using bilinear resampling.

    If the source is already in target_crs, we copy instead — this
    preserves the exact pixel values and is much faster.

    Bilinear chosen over nearest-neighbour:
        - imagery (continuous values) → bilinear is smoother
        - masks would need nearest-neighbour but we handle those
          separately in Step 2

    ARGS:
        src_path   : source GeoTIFF (any CRS)
        dst_path   : destination GeoTIFF (will be target_crs)
        target_crs : EPSG string, default EPSG:3857

    RETURNS:
        dict with status ("copied" or "reprojected"), src/dst CRS
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_crs = src.crs

        # ── Fast path: already correct ────────────────────────────────────
        if src_crs == CRS.from_string(target_crs):
            shutil.copy2(src_path, dst_path)
            return {
                "status":       "copied",
                "src_crs":      str(src_crs),
                "dst_crs":      target_crs,
                "reprojected":  False,
            }

        # ── Reprojection path ─────────────────────────────────────────────
        # calculate_default_transform computes the best output transform
        # (pixel size, affine matrix) for the new CRS
        transform, width, height = calculate_default_transform(
            src_crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        meta = src.meta.copy()
        meta.update({
            "crs":       target_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
        })

        with rasterio.open(dst_path, "w", **meta) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,  # smooth for imagery
                )

        return {
            "status":      "reprojected",
            "src_crs":     str(src_crs),
            "dst_crs":     target_crs,
            "reprojected": True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_crs_reprojection(aoi_dir: Path, out_dir: Path) -> Dict:
    """
    Runs CRS reprojection for all monthly images of ONE city.

    Called by run_pipeline.py → run_step1_alignment()

    ARGS:
        aoi_dir : raw city folder
                  e.g. .../train/L15-0331E-1257N_1327_3160_13
        out_dir : destination for reprojected images
                  e.g. data/processed/alignment/reprojected/{city}

    RETURNS:
        summary dict (counts, status, resolution, shape)
    """
    city = aoi_dir.name
    img_dir = aoi_dir / IMAGE_SUBDIR

    logger.info("=" * 60)
    logger.info("SUB-STEP 1.1 : CRS REPROJECTION")
    logger.info("=" * 60)
    logger.info(f"City       : {city}")
    logger.info(f"Source     : {img_dir}")
    logger.info(f"Target CRS : {TARGET_CRS}")
    logger.info(f"Output     : {out_dir}")

    # ── Discover all monthly GeoTIFFs ─────────────────────────────────────
    # images_masked/ contains one .tif per month, sorted alphabetically
    # (filenames include the date, so sort = chronological order)
    tif_files: List[Path] = sorted(img_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif files found in {img_dir}. "
            "Check that the dataset path is correct."
        )

    logger.info(f"\nFound {len(tif_files)} monthly images")

    # ── Inspect first image for metadata summary ──────────────────────────
    first_meta = _read_image_meta(tif_files[0])
    logger.info(f"CRS        : {first_meta['crs']}")
    logger.info(f"EPSG       : {first_meta['epsg']}")
    logger.info(f"Shape      : {first_meta['shape']}  (H × W)")
    logger.info(f"Resolution : {first_meta['resolution_m']} m/pixel")
    logger.info(f"Bands      : {first_meta['bands']}  (R, G, B, Alpha)")
    logger.info(f"Dtype      : {first_meta['dtype']}")

    # ── Process every month ───────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    n_reprojected = 0
    n_copied = 0

    for tif_path in tif_files:
        dst_path = out_dir / tif_path.name
        result = _reproject_image(tif_path, dst_path, TARGET_CRS)
        results.append(result)

        if result["reprojected"]:
            n_reprojected += 1
            logger.info(f"  REPROJECTED : {tif_path.name}")
        else:
            n_copied += 1
            logger.debug(f"  Copied      : {tif_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CRS REPROJECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total images  : {len(tif_files)}")
    logger.info(f"Copied (ok)   : {n_copied}")
    logger.info(f"Reprojected   : {n_reprojected}")

    status = "all_correct" if n_reprojected == 0 else "reprojected"
    if n_reprojected == 0:
        logger.info("STATUS : All images already EPSG:3857 — no reprojection needed ✓")
    else:
        logger.info(f"STATUS : Reprojected {n_reprojected} images to EPSG:3857 ✓")

    logger.info("NEXT   : Sub-step 1.2 → Coregistration")
    logger.info("=" * 60)

    return {
        "city":            city,
        "total_images":    len(tif_files),
        "n_copied":        n_copied,
        "n_reprojected":   n_reprojected,
        "target_crs":      TARGET_CRS,
        "resolution_m":    first_meta["resolution_m"],
        "image_shape":     first_meta["shape"],
        "bands":           first_meta["bands"],
        "dtype":           first_meta["dtype"],
        "status":          status,
        "output_dir":      str(out_dir),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN (single city test)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick test on one city — change path to match your dataset location
    AOI_DIR = Path(
        "data/raw/dataset/SN7_buildings_train/train/"
        "L15-0358E-1220N_1433_3310_13"
    )
    OUT_DIR = Path(
        "data/processed/alignment_1/reprojected/"
        "L15-0358E-1220N_1433_3310_13"
    )

    summary = run_crs_reprojection(AOI_DIR, OUT_DIR)

    print("\n✓ Sub-step 1.1 Complete")
    print(f"  Images total  : {summary['total_images']}")
    print(f"  Copied        : {summary['n_copied']}")
    print(f"  Reprojected   : {summary['n_reprojected']}")
    print(f"  CRS           : {summary['target_crs']}")
    print(f"  Resolution    : {summary['resolution_m']} m/pixel")
    print(f"  Shape         : {summary['image_shape']}")
    print(f"  Bands         : {summary['bands']} (R,G,B,Alpha)")
    print(f"  Dtype         : {summary['dtype']}")
    print(f"  Status        : {summary['status']}")
    print(f"  Output        : {summary['output_dir']}")
