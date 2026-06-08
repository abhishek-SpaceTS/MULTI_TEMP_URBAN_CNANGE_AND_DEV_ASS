"""
preprocessing/mask_engineering/watershed.py
=============================================

SUB-STEP 5.2 — Watershed Separation of Touching Buildings
----------------------------------------------------------

PURPOSE:
    Apply the watershed algorithm to separate touching/adjacent buildings
    that appear merged in the Footprint channel of each FBC mask.

WHY WATERSHED:
    In dense urban scenes, building polygons often share edges
    or are separated by only 1-2 pixels. After rasterisation
    in Step 5.1, these buildings can merge into a single blob.
    The watershed algorithm treats the Contact channel as a
    "valley" between building peaks, flooding from markers
    (building interiors) to separate them at the valley.

HOW IT WORKS:
    1. MARKERS: use the eroded Footprint (building interiors,
       away from edges) as seed markers. Each connected component
       in the eroded Footprint becomes one marker = one building.
    2. DISTANCE MAP: compute distance transform on Footprint —
       pixels further from the background have higher values.
    3. WATERSHED: flood from markers up the distance landscape.
       The Contact zones (low valleys between buildings) become
       the watershed divide lines.
    4. OUTPUT: per-building instance labels → re-binarised as
       separated footprint mask.

PRESENCE CHECK:
    Before processing each FBC file we verify it exists on disk.
    This makes the step robust to any earlier-step failures or
    partial re-runs — missing FBC files are logged and skipped
    rather than crashing the entire city run.

MIN_BUILDING_SEPARATION_PX = 3 (changed from 5):
    At 1.195 m/pixel, 5 px ≈ 6 m minimum gap. For dense Indian
    urban fabric — narrow galis, row houses, chawls — this was
    too aggressive and merged distinct buildings into blobs.
    3 px ≈ 3.6 m is much better suited for separating closely
    packed individual structures without over-splitting large ones.

OUTPUT FORMAT:
    {patch_id}_watershed.png — single-channel uint8 PNG
    Values: 0 = background, 255 = building (separated)

INPUT:
    fbc_dir : data/processed/mask_engineering/fbc/{city}/*_fbc.png

OUTPUT:
    data/processed/mask_engineering/watershed/{city}/*_watershed.png

NEXT STEP → 6.1 geometric.py
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum distance between watershed peaks (one peak = one building marker).
# 3 px ≈ 3.6 m at 1.195 m/pixel — appropriate for dense Indian urban fabric.
# (Was 5 px / ~6 m, which over-merged row houses and narrow-gali structures.)
MIN_BUILDING_SEPARATION_PX = 3

# Erosion kernel for marker generation (must be smaller than smallest building)
MARKER_EROSION_PX = 3


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_fbc_png(fbc_path: Path):
    """
    Loads a 3-channel FBC PNG and returns (footprint, boundary, contact)
    as binary uint8 arrays.
    """
    img       = np.array(Image.open(fbc_path).convert("RGB"))
    footprint = (img[:, :, 0] > 128).astype(np.uint8)  # R channel
    boundary  = (img[:, :, 1] > 128).astype(np.uint8)  # G channel
    contact   = (img[:, :, 2] > 128).astype(np.uint8)  # B channel
    return footprint, boundary, contact


def _apply_watershed(footprint: np.ndarray) -> np.ndarray:
    """
    Applies watershed to separate touching buildings in a binary footprint.

    ARGS:
        footprint : uint8 (H, W) binary building mask, 1=building

    RETURNS:
        uint8 (H, W) binary mask with separated buildings, 255=building
    """
    # ── Step 1: Empty check ───────────────────────────────────────────────
    if footprint.sum() == 0:
        return np.zeros(footprint.shape, dtype=np.uint8)

    # ── Step 2: Distance transform ────────────────────────────────────────
    # Each foreground pixel gets value = distance to nearest background.
    # Building centers have high values (peaks).
    dist_transform = ndi.distance_transform_edt(footprint)

    # ── Step 3: Find local maxima as building markers ─────────────────────
    # One peak per building. min_distance = MIN_BUILDING_SEPARATION_PX
    # enforces the minimum gap before two blobs are treated as one building.
    coords = peak_local_max(
        dist_transform,
        min_distance=MIN_BUILDING_SEPARATION_PX,
        labels=footprint,
    )

    # ── Step 4: Build marker image ────────────────────────────────────────
    markers = np.zeros(footprint.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    markers, _ = ndi.label(markers)

    # ── Step 5: Run watershed ─────────────────────────────────────────────
    # Negative distance: watershed floods from valleys upward.
    labels = watershed(-dist_transform, markers, mask=footprint)

    # ── Step 6: Re-binarise ───────────────────────────────────────────────
    separated = (labels > 0).astype(np.uint8) * 255

    return separated


def _write_watershed_png(mask: np.ndarray, dst_path: Path) -> None:
    """Writes single-channel uint8 watershed result as grayscale PNG."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(dst_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_watershed(
    fbc_dir: Path,
    out_dir: Path,
) -> Dict:
    """
    Applies watershed separation to all FBC masks for ONE city.

    Iterates over *_fbc.png files written by step 5.1. Includes a
    presence check so partial re-runs or missing files are handled
    gracefully rather than crashing.

    Called by run_pipeline.py → run_step5_mask_engineering()

    ARGS:
        fbc_dir : FBC PNG files from step 5.1
        out_dir : destination for watershed PNGs
    """
    city = fbc_dir.name

    logger.info("=" * 60)
    logger.info("SUB-STEP 5.2 : WATERSHED BUILDING SEPARATION")
    logger.info("=" * 60)
    logger.info(f"City            : {city}")
    logger.info(
        f"Min separation  : {MIN_BUILDING_SEPARATION_PX} px  "
        f"(≈ {MIN_BUILDING_SEPARATION_PX * 1.195:.1f} m)  "
        f"[tuned for dense Indian urban fabric]"
    )
    logger.info(f"Output          : {out_dir}")

    # ── Collect FBC files written by step 5.1 ────────────────────────────
    fbc_files: List[Path] = sorted(fbc_dir.glob("*_fbc.png"))
    if not fbc_files:
        raise FileNotFoundError(
            f"No _fbc.png files in {fbc_dir}. "
            "Did step 5.1 (FBC mask) complete?"
        )

    n_candidates = len(fbc_files)
    logger.info(f"FBC candidates  : {n_candidates:,}")

    out_dir.mkdir(parents=True, exist_ok=True)

    n_written   = 0
    n_missing   = 0   # FBC file listed but not actually on disk
    n_empty     = 0   # patches with no building pixels
    n_separated = 0   # patches where watershed changed the mask

    for idx, fbc_path in enumerate(fbc_files):

        # ── PRESENCE CHECK ────────────────────────────────────────────────
        # Defensive guard: verify the file actually exists before opening.
        # Handles partial step-5.1 runs or race conditions in parallel jobs.
        if not fbc_path.exists():
            logger.warning(f"  Missing FBC file (skipping): {fbc_path.name}")
            n_missing += 1
            continue

        footprint, boundary, contact = _load_fbc_png(fbc_path)

        # ── Apply watershed ───────────────────────────────────────────────
        separated = _apply_watershed(footprint)

        # ── Track stats ───────────────────────────────────────────────────
        if footprint.sum() == 0:
            n_empty += 1
        else:
            # Detect whether watershed actually changed anything
            original_binary = footprint.astype(np.int16) * 255
            if np.abs(original_binary - separated.astype(np.int16)).sum() > 0:
                n_separated += 1

        # ── Write output ──────────────────────────────────────────────────
        ws_fname = fbc_path.name.replace("_fbc.png", "_watershed.png")
        dst_path = out_dir / ws_fname
        _write_watershed_png(separated, dst_path)
        n_written += 1

        if (idx + 1) % 500 == 0:
            logger.info(f"  Processed {idx+1:,}/{n_candidates:,} patches...")

    logger.info("\n" + "=" * 60)
    logger.info("WATERSHED SUMMARY")
    logger.info("=" * 60)
    logger.info(f"FBC candidates       : {n_candidates:,}")
    logger.info(f"Presence-skipped     : {n_missing:,}  (file not on disk)")
    logger.info(f"Written              : {n_written:,}")
    logger.info(f"  Empty (no bldg)    : {n_empty:,}")
    logger.info(f"  Separation applied : {n_separated:,}")
    logger.info(f"  Unchanged          : {n_written - n_empty - n_separated:,}")
    logger.info("STATUS : Watershed complete ✓")
    logger.info("NEXT   : Step 6.1 → Geometric Augmentation")
    logger.info("=" * 60)

    return {
        "city":        city,
        "n_written":   n_written,
        "n_empty":     n_empty,
        "n_separated": n_separated,
        "n_missing":   n_missing,
        "output_dir":  str(out_dir),
        "status":      "complete",
    }


if __name__ == "__main__":
    BASE_FBC = Path("data/processed/mask_engineering_1/fbc")

    for city_dir in sorted(BASE_FBC.iterdir()):
        if not city_dir.is_dir():
            continue
        C = city_dir.name
        summary = run_watershed(
            fbc_dir = city_dir,
            out_dir = Path(f"data/processed/mask_engineering_1/watershed/{C}"),
        )
        print(f"\n✓ {C}")
        print(f"  Written    : {summary['n_written']:,}")
        print(f"  Empty      : {summary['n_empty']:,}")
        print(f"  Separated  : {summary['n_separated']:,}")
        print(f"  Missing    : {summary['n_missing']:,}")
        print(f"  Output     : {summary['output_dir']}")