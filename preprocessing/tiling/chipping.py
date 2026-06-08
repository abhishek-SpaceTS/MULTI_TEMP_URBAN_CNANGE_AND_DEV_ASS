"""
preprocessing/tiling/chipping.py
==================================

SUB-STEP 4.1 — Tiling: 4096×4096 → 256×256 Patches
-----------------------------------------------------

PURPOSE:
    Slice each normalised 4096×4096 monthly image into
    non-overlapping 256×256 patches for model training.

    4096 / 256 = 16 patches per axis → 256 patches per image.
    n_months × 256 patches × 60 cities = ~370,000 candidate patches.

WHY 256×256:
    - Standard crop size for UNet variants
    - Fits in GPU memory: batch_size=8 × 256×256×4band × float32 = ~2GB
    - 256 pixels × 1.195 m/pixel = ~306 m ground footprint
      Large enough to capture building context, small enough to be
      computationally tractable.

PATCH NAMING CONVENTION:
    {city_id}_{month_idx:02d}_{row:02d}_{col:02d}.tif
    e.g. L15-0331E_m00_r00_c00.tif  (city, month 0, row 0, col 0)
    row/col are 0-indexed patch grid coordinates (0-15 each axis).

GEOSPATIAL METADATA:
    Each patch retains its correct affine transform — the top-left
    origin is shifted by (col × 256 × pixel_width, row × 256 × pixel_height).
    This keeps patches spatially registered, important for:
      - Overlaying with GeoJSON labels in Step 5
      - QA/debug visualisation in GIS software

SMART SAMPLING — GeoJSON-Based (Data Imbalance Guard):
    Two sequential filters are applied before saving any patch:

    Filter 1 — Alpha / No-data:
        If Band 4 (Alpha) shows >90% masked pixels (value == 0),
        the patch is entirely ocean/void. Discard unconditionally.

    Filter 2 — GeoJSON Building Presence + Background Subsampling:
        For each month, the corresponding labels_match_pix GeoJSON is
        loaded ONCE and cached. All building polygons are scaled by 4×
        (from 1024-px label space to 4096-px image space). For each
        256×256 patch, we check whether any scaled polygon intersects
        the patch bounding box using Shapely:

          has_buildings = any(poly.intersects(patch_box) for poly in polys)

        - Patches WITH buildings    → always saved (100%)
        - Patches WITHOUT buildings → saved with 10% probability

        This is the correct signal: we know exactly which patches
        have buildings because the labels tell us so. No spectral
        heuristics, no proxy metrics — ground truth labels drive
        the sampling decision.

        FALLBACK: If no GeoJSON is found for a month (labels missing),
        the filter degrades gracefully to a spectral mean heuristic
        (band mean < SPECTRAL_FALLBACK_THRESHOLD → background).

    Net effect: ~370k candidates → ~150k high-quality patches.
    Only kept patches are written to disk; nothing needs cleanup.

STORAGE CLEANUP:
    After chipping completes, the 4096×4096 intermediate files
    (~1.6 GB per city) are DELETED by run_pipeline.py cleanup_city().
    Only the 256×256 patches are kept.

INPUT:
    data/processed/normalization/cielab/{city}/*.tif  (4096×4096)

LABEL INPUT:
    data/raw/dataset/SN7_buildings_train/train/{city}/labels_match_pix/*.geojson

OUTPUT:
    data/processed/tiling/patches/{city}/*.tif  (256×256)

NEXT STEP → 4.2 json_sidecars.py
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
import shapely.geometry as sg
import shapely.affinity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PATCH_SIZE  = 256   # output patch size in pixels
GRID_SIZE   = 16    # 4096 / 256 = 16 patches per axis

# Filter 1 — Alpha no-data threshold
NODATA_DISCARD_THRESHOLD = 0.90   # >90% masked pixels → discard

# Filter 2 — Background subsampling
BACKGROUND_KEEP_RATE = 0.10       # keep 10% of no-building patches

# Fallback spectral threshold (used only when GeoJSON is unavailable)
SPECTRAL_FALLBACK_THRESHOLD = 15.0

# Label coordinate space → image coordinate space
LABEL_UPSAMPLE_FACTOR = 4         # labels in 1024-px space; image in 4096-px space


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _short_city_id(city: str) -> str:
    """
    Shortens the city folder name for use in patch filenames.

    L15-0331E-1257N_1327_3160_13 → L15-0331E
    Keeps the unique identifier prefix; drops the redundant suffix.
    """
    return city.split("_")[0] if "_" in city else city[:12]


def _extract_month_name(tif_name: str) -> Optional[str]:
    """
    Extracts YYYY_MM from a SpaceNet 7 image filename.

    e.g. global_monthly_2018_01_mosaic_L15-0331E...tif → "2018_01"
    Returns None if the pattern is not found.
    """
    m = re.search(r"(\d{4}_\d{2})", tif_name)
    return m.group(1) if m else None


def _find_geojson(labels_dir: Optional[Path], month_name: Optional[str]) -> Optional[Path]:
    """
    Finds the GeoJSON for a given month in labels_match_pix/.

    SpaceNet 7 label filenames contain the YYYY_MM string:
        global_monthly_2018_01_mosaic_L15-0331E..._Buildings.geojson

    Tries an exact substring match first, then a de-underscored match.
    Returns None if labels_dir is missing or no match is found.
    """
    if labels_dir is None or not labels_dir.exists() or month_name is None:
        return None

    for gj in labels_dir.glob("*.geojson"):
        if month_name in gj.name:
            return gj
        if month_name.replace("_", "") in gj.name.replace("_", ""):
            return gj
    return None


def _load_scaled_polygons(
    geojson_path: Path,
    scale:        int = LABEL_UPSAMPLE_FACTOR,
) -> List[sg.Polygon]:
    """
    Loads building polygons from a labels_match_pix GeoJSON and scales
    them from 1024-px label space to 4096-px image space.

    Called ONCE per month — the result is cached in poly_cache so all
    256 patch windows for that month reuse the same polygon list.

    ARGS:
        geojson_path : path to the _Buildings.geojson file
        scale        : upsampling factor (default 4×)

    RETURNS:
        list of shapely Polygons in 4096-px image coordinates
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    polys: List[sg.Polygon] = []
    for feature in gj.get("features", []):
        geom = feature.get("geometry")
        if geom is None:
            continue
        try:
            poly = sg.shape(geom)
            # Scale from 1024-px label space → 4096-px image space
            poly = shapely.affinity.scale(
                poly,
                xfact=scale,
                yfact=scale,
                origin=(0, 0),
            )
            polys.append(poly)
        except Exception:
            continue

    return polys


def _has_building_intersection(
    polys:      List[sg.Polygon],
    col_offset: int,
    row_offset: int,
    patch_size: int = PATCH_SIZE,
) -> bool:
    """
    Returns True if ANY scaled polygon intersects the patch bounding box.

    The patch box is defined in 4096-px image space:
        left   = col_offset
        right  = col_offset + patch_size
        top    = row_offset
        bottom = row_offset + patch_size

    A linear scan is acceptable here: per-month polygon counts for
    SpaceNet 7 are typically < 2,000, and this runs per-patch (not
    per-pixel). For cities with very high building density (>5,000
    polygons/month) an STRtree spatial index would be faster.

    ARGS:
        polys      : scaled polygons in 4096-px image space
        col_offset : left pixel of the patch in the full 4096-px image
        row_offset : top pixel of the patch in the full 4096-px image
        patch_size : patch dimension (default 256)

    RETURNS:
        True if at least one polygon overlaps this patch window
    """
    if not polys:
        return False

    patch_box = sg.box(
        col_offset,
        row_offset,
        col_offset + patch_size,
        row_offset + patch_size,
    )

    return any(poly.intersects(patch_box) for poly in polys)


def _spectral_has_buildings(data: np.ndarray) -> bool:
    """
    Fallback building-presence heuristic when no GeoJSON is available.

    Uses the mean pixel value of the first three bands. A very dark
    mean (< SPECTRAL_FALLBACK_THRESHOLD) indicates empty ground/water.
    Called only when labels_match_pix/ GeoJSON is missing for a month.
    """
    if data.shape[0] < 3:
        return True   # unknown band count — conservatively keep the patch
    return float(data[:3].mean()) >= SPECTRAL_FALLBACK_THRESHOLD


def _chip_one_image(
    src_path:   Path,
    out_dir:    Path,
    city_id:    str,
    month_idx:  int,
    polys:      Optional[List[sg.Polygon]],
    patch_size: int = PATCH_SIZE,
    grid_size:  int = GRID_SIZE,
) -> Tuple[int, int, int, int]:
    """
    Slices ONE 4096×4096 image into patch_size × patch_size patches,
    applying two-stage smart sampling (alpha filter + GeoJSON gate)
    before saving each patch.

    ARGS:
        src_path   : normalised 4096×4096 GeoTIFF
        out_dir    : destination directory for saved patches
        city_id    : short city identifier for filename
        month_idx  : 0-indexed month number
        polys      : scaled building polygons in 4096-px space,
                     or None → triggers spectral fallback per patch
        patch_size : output patch dimension (default 256)
        grid_size  : patches per axis (default 16)

    RETURNS:
        (n_written, n_nodata_skipped, n_bg_skipped, n_fallback_used)
          n_nodata_skipped : patches dropped by Filter 1 (alpha)
          n_bg_skipped     : background patches dropped by Filter 2
          n_fallback_used  : patches that used spectral heuristic
                             (incremented only when polys is None)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written        = 0
    n_nodata_skipped = 0
    n_bg_skipped     = 0
    n_fallback_used  = 0

    with rasterio.open(src_path) as src:
        meta      = src.meta.copy()
        transform = src.transform
        h, w      = src.height, src.width

        if h != grid_size * patch_size or w != grid_size * patch_size:
            logger.warning(
                f"    Unexpected size {h}×{w} for {src_path.name}. "
                f"Expected {grid_size*patch_size}×{grid_size*patch_size}. "
                f"Tiling anyway — edge patches may be incomplete."
            )

        meta.update({"width": patch_size, "height": patch_size})

        for row in range(grid_size):
            for col in range(grid_size):

                row_off = row * patch_size
                col_off = col * patch_size

                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=patch_size,
                    height=patch_size,
                )

                # ── Read patch data ───────────────────────────────────────
                data = src.read(window=window)   # (bands, H, W)

                # ── FILTER 1: Alpha / No-data ─────────────────────────────
                # Discard patches that are >90% masked (ocean/void).
                if data.shape[0] >= 4:
                    alpha       = data[3]
                    masked_frac = np.count_nonzero(alpha == 0) / alpha.size
                    if masked_frac > NODATA_DISCARD_THRESHOLD:
                        n_nodata_skipped += 1
                        continue

                # ── FILTER 2: Building presence check ────────────────────
                # PRIMARY: GeoJSON intersection — exact ground-truth signal.
                # FALLBACK: spectral mean heuristic when labels are missing.
                if polys is not None:
                    has_buildings = _has_building_intersection(
                        polys, col_off, row_off, patch_size
                    )
                else:
                    has_buildings = _spectral_has_buildings(data)
                    n_fallback_used += 1

                # Background subsampling: keep only BACKGROUND_KEEP_RATE
                # of patches with no building intersection. This controls
                # the class imbalance without discarding all background
                # context (some background is useful for the model).
                if not has_buildings:
                    if random.random() >= BACKGROUND_KEEP_RATE:
                        n_bg_skipped += 1
                        continue   # patch not saved — never written to disk

                # ── Compute patch-specific affine transform ────────────────
                # Shift the geographic origin to the top-left pixel of
                # this patch window so the output TIF is spatially correct.
                patch_transform = Affine(
                    transform.a,
                    transform.b,
                    transform.c + col_off * transform.a,   # shifted x origin
                    transform.d,
                    transform.e,
                    transform.f + row_off * transform.e,   # shifted y origin
                )
                meta["transform"] = patch_transform

                # ── Write patch ───────────────────────────────────────────
                fname    = f"{city_id}_m{month_idx:02d}_r{row:02d}_c{col:02d}.tif"
                dst_path = out_dir / fname
                with rasterio.open(dst_path, "w", **meta) as dst:
                    dst.write(data)

                n_written += 1

    return n_written, n_nodata_skipped, n_bg_skipped, n_fallback_used


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_chipping(
    cielab_dir:  Path,
    out_dir:     Path,
    labels_root: Optional[Path] = None,
    patch_size:  int = PATCH_SIZE,
    seed:        int = 42,
) -> Dict:
    """
    Tiles ALL monthly images for ONE city into 256×256 patches,
    using GeoJSON building labels to drive the smart-sampling filter.

    Called by run_pipeline.py → run_step4_tiling()

    ARGS:
        cielab_dir  : final normalised 4096×4096 images (step 3.3)
        out_dir     : patch output directory
        labels_root : path to the city's raw AOI folder containing
                      labels_match_pix/  e.g.:
                      data/raw/.../SN7_buildings_train/train/{city}
                      Pass None to use spectral fallback for all months.
        patch_size  : output patch size (default 256)
        seed        : random seed for background subsampling
    """
    random.seed(seed)

    city       = cielab_dir.name
    city_id    = _short_city_id(city)
    grid_sz    = 4096 // patch_size
    labels_dir = (labels_root / "labels_match_pix") if labels_root else None

    logger.info("=" * 60)
    logger.info("SUB-STEP 4.1 : CHIPPING  (4096×4096 → 256×256 PATCHES)")
    logger.info("=" * 60)
    logger.info(f"City             : {city}  (id: {city_id})")
    logger.info(f"Patch size       : {patch_size} × {patch_size}")
    logger.info(f"Grid             : {grid_sz} × {grid_sz} = {grid_sz**2} patches/image")
    logger.info(f"Labels dir       : {labels_dir}  "
                f"{'✓' if (labels_dir and labels_dir.exists()) else '(missing — spectral fallback)'}")
    logger.info(f"Nodata threshold : >{NODATA_DISCARD_THRESHOLD*100:.0f}% masked → discard")
    logger.info(f"BG keep rate     : {BACKGROUND_KEEP_RATE*100:.0f}% of no-building patches kept")
    logger.info(f"Random seed      : {seed}")
    logger.info(f"Output           : {out_dir}")

    tif_files: List[Path] = sorted(cielab_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {cielab_dir}")

    n_months        = len(tif_files)
    candidate_count = n_months * grid_sz ** 2
    logger.info(f"Time steps       : {n_months}")
    logger.info(f"Candidate patches: ~{candidate_count:,}  (before filters)")

    # ── Per-month polygon cache ───────────────────────────────────────────
    # Each GeoJSON is parsed exactly once per month and its scaled polygons
    # are reused across all 256 patch windows for that month.
    # None in the cache means "GeoJSON not found → use spectral fallback."
    poly_cache: Dict[Optional[str], Optional[List[sg.Polygon]]] = {}

    total_written        = 0
    total_nodata_skipped = 0
    total_bg_skipped     = 0
    total_fallback       = 0

    for month_idx, tif_path in enumerate(tif_files):

        # ── Resolve and cache polygons for this month ─────────────────────
        month_name = _extract_month_name(tif_path.name)

        if month_name not in poly_cache:
            gj_path = _find_geojson(labels_dir, month_name)

            if gj_path is not None:
                polys = _load_scaled_polygons(gj_path, scale=LABEL_UPSAMPLE_FACTOR)
                poly_cache[month_name] = polys
                logger.info(
                    f"  [m{month_idx:02d}] GeoJSON: {gj_path.name}  "
                    f"({len(polys):,} polygons, scaled {LABEL_UPSAMPLE_FACTOR}×)"
                )
            else:
                poly_cache[month_name] = None
                logger.warning(
                    f"  [m{month_idx:02d}] No GeoJSON for '{month_name}' "
                    f"— spectral fallback active for this month"
                )

        polys = poly_cache[month_name]

        # ── Chip the image with smart sampling ───────────────────────────
        n_w, n_nd, n_bg, n_fb = _chip_one_image(
            src_path   = tif_path,
            out_dir    = out_dir,
            city_id    = city_id,
            month_idx  = month_idx,
            polys      = polys,
            patch_size = patch_size,
            grid_size  = grid_sz,
        )

        total_written        += n_w
        total_nodata_skipped += n_nd
        total_bg_skipped     += n_bg
        total_fallback       += n_fb

        logger.info(
            f"  [m{month_idx:02d}] {tif_path.name[:38]}"
            f"  saved={n_w}"
            f"  nodata_skip={n_nd}"
            f"  bg_skip={n_bg}"
            + (f"  fallback={n_fb}" if n_fb else "")
        )

    # ── Summary ───────────────────────────────────────────────────────────
    total_skipped = total_nodata_skipped + total_bg_skipped
    reduction_pct = (total_skipped / candidate_count * 100) if candidate_count else 0.0

    logger.info("\n" + "=" * 60)
    logger.info("CHIPPING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Months              : {n_months}")
    logger.info(f"Candidate patches   : {candidate_count:,}")
    logger.info(f"Nodata discarded    : {total_nodata_skipped:,}  (Filter 1 — >90% alpha-masked)")
    logger.info(f"Background skipped  : {total_bg_skipped:,}  "
                f"(Filter 2 — {100 - BACKGROUND_KEEP_RATE*100:.0f}% of no-building patches dropped)")
    logger.info(f"Total skipped       : {total_skipped:,}  ({reduction_pct:.1f}% reduction)")
    logger.info(f"Patches saved       : {total_written:,}")
    if total_fallback:
        logger.warning(
            f"Spectral fallback   : {total_fallback:,} patches used heuristic "
            f"(GeoJSON missing for ≥1 month)"
        )
    logger.info("STATUS : Chipping complete ✓")
    logger.info("NOTE   : 4096×4096 files will be DELETED by cleanup_city()")
    logger.info("NEXT   : Sub-step 4.2 → JSON Sidecars")
    logger.info("=" * 60)

    return {
        "city":               city,
        "n_months":           n_months,
        "candidate_patches":  candidate_count,
        "total_patches":      total_written,
        "nodata_skipped":     total_nodata_skipped,
        "background_skipped": total_bg_skipped,
        "total_skipped":      total_skipped,
        "fallback_patches":   total_fallback,
        "patch_size":         patch_size,
        "output_dir":         str(out_dir),
        "status":             "complete",
    }


if __name__ == "__main__":
    C = "L15-0331E-1257N_1327_3160_13"
    summary = run_chipping(
        cielab_dir  = Path(f"data/processed/normalization/cielab/{C}"),
        out_dir     = Path(f"data/processed/tiling/patches/{C}"),
        labels_root = Path(f"data/raw/dataset/SN7_buildings_train/train/{C}"),
    )
    print(f"\n✓ Sub-step 4.1 Complete")
    print(f"  Months      : {summary['n_months']}")
    print(f"  Candidates  : {summary['candidate_patches']:,}")
    print(f"  Saved       : {summary['total_patches']:,}")
    print(f"  Nodata skip : {summary['nodata_skipped']:,}")
    print(f"  BG skip     : {summary['background_skipped']:,}")
    print(f"  Total skip  : {summary['total_skipped']:,}")
    if summary["fallback_patches"]:
        print(f"  ⚠ Fallback  : {summary['fallback_patches']:,} patches used spectral heuristic")
    print(f"  Output      : {summary['output_dir']}")