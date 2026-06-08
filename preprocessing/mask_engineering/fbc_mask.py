
"""
preprocessing/mask_engineering/fbc_mask.py
============================================

SUB-STEP 5.1 — Footprint-Boundary-Contact (FBC) Mask
------------------------------------------------------

PURPOSE:
    Convert the GeoJSON building polygon labels into a 3-channel
    semantic target mask for each patch:

      Channel 0 : FOOTPRINT  — interior of each building (1 = building)
      Channel 1 : BOUNDARY   — eroded boundary ring (~2px wide)
      Channel 2 : CONTACT    — zone between adjacent buildings

    The UNet-GFAM predicts all 3 channels simultaneously.
    This 3-class encoding solves the "touching buildings" problem:
    without a Contact channel, adjacent buildings merge into
    one blob and the model cannot separate them.

DATA SOURCE: labels_match_pix/ GeoJSON
    These files contain building polygons in PIXEL coordinates
    aligned to the original 1024×1024 image grid.
    We scale them by UPSAMPLE_FACTOR=4 to reach 4096-px space,
    then extract the 256×256 sub-window for each patch.

    Confirmed coordinate space (from diagnose_fbc.py output):
        X: -2.2 → 1024.9   Y: -0.4 → 1022.5
        → 1024-px pixel space → scale by 4 → correct ✓

PATH RESOLUTION BUG FIX:
    The sidecar labels_geojson field stores a relative path like:
        "L15-0331E.../labels_match_pix/global_monthly_2018_01...geojson"

    This path already includes the city folder name, so it must be
    prepended with raw_city_dir.parent (= .../train/) to resolve:
        .../train/ + L15-0331E.../labels_match_pix/...geojson  ✓

    The original code used raw_city_dir.parent.parent which skipped
    the train/ directory, producing a path that never existed on disk,
    so poly_cache[month_name] was always [] → all-black FBC masks.

PRESENCE CHECK (1:1 Alignment Guard):
    Before processing any sidecar, we verify that the corresponding
    .tif image patch actually exists on disk. This guarantees the
    FBC output folder is a perfect 1:1 match with the images folder.

SHORT-CIRCUIT for Empty Patches:
    If a patch has zero building polygons, we write an all-black PNG
    immediately without running the expensive morphological operations.

OUTPUT:
    data/processed/mask_engineering/fbc/{city}/{patch_id}_fbc.png
    3-channel uint8 PNG (0/255), channels = Footprint, Boundary, Contact

NEXT STEP → 5.2 watershed.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
from PIL import Image
from rasterio.features import rasterize
import shapely.geometry as sg
import shapely.affinity          # separate submodule — sg.affinity does NOT exist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PATCH_SIZE          = 256
UPSAMPLE_FACTOR     = 4.0   # labels in 1024-px space → scale to 4096-px space
BOUNDARY_EROSION_PX = 2  # ring width around building interior
CONTACT_DILATE_PX   = 3     # approach-zone dilation


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_sidecar(sidecar_path: Path) -> Dict:
    with open(sidecar_path) as f:
        return json.load(f)


def _resolve_geojson_path(
    sc:           Dict,
    raw_city_dir: Path,
    labels_dir:   Path,
    month_name:   str,
) -> Optional[Path]:
    """
    Resolves the GeoJSON label file for one month using a 3-level fallback.

    WHY THIS FUNCTION EXISTS:
        The sidecar stores labels_geojson as a relative path that
        already starts with the city folder name, e.g.:
            "L15-0331E.../labels_match_pix/global_monthly_2018_01...geojson"

        To get the full path we prepend raw_city_dir.parent (.../train/).
        The original code used .parent.parent which skipped train/ and
        produced a path that never existed → all-black FBC masks.

    FALLBACK ORDER:
        1. raw_city_dir.parent / gj_rel       (.../train/{city}/labels/...)
        2. raw_city_dir.parent.parent / gj_rel (one level higher, just in case)
        3. Glob labels_match_pix/ for *{month_name}*.geojson

    ARGS:
        sc           : sidecar dict with labels_geojson key
        raw_city_dir : .../train/{city}
        labels_dir   : .../train/{city}/labels_match_pix
        month_name   : e.g. "2018_01"

    RETURNS:
        Resolved Path that exists on disk, or None
    """
    gj_rel = sc.get("labels_geojson")

    if gj_rel:
        # ── Level 1: parent = .../train/ ──────────────────────────────────
        # This is the correct fix. The relative path stored in the sidecar
        # starts with the city folder name, so we only need to prepend
        # the train/ directory (= raw_city_dir.parent).
        p1 = raw_city_dir.parent / gj_rel
        if p1.exists():
            return p1

        # ── Level 2: parent.parent as fallback ────────────────────────────
        # Handles unusual dataset layouts where train/ is not the immediate
        # parent of the city folder.
        p2 = raw_city_dir.parent.parent / gj_rel
        if p2.exists():
            return p2

    # ── Level 3: glob by month_name ───────────────────────────────────────
    # Completely independent of the sidecar path — searches labels_match_pix/
    # directly for any file containing the month string.
    if labels_dir.exists() and month_name and month_name != "unknown":
        matches = sorted(labels_dir.glob(f"*{month_name}*.geojson"))
        if matches:
            return matches[0]

    return None


def _load_geojson_polygons(geojson_path: Path) -> List[sg.Polygon]:
    """
    Loads all building polygons from a labels_match_pix GeoJSON.

    Input coordinates are in 1024-px image space.
    Output polygons are scaled to 4096-px space (UPSAMPLE_FACTOR=4).

    No debug print statements — check diagnose_fbc.py if you need
    to inspect raw coordinate values.
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    polys = []
    for feature in gj.get("features", []):
        geom = feature.get("geometry")
        if geom is None:
            continue
        try:
            poly = sg.shape(geom)
            # Scale 1024-px → 4096-px coordinate space
            poly = shapely.affinity.scale(
                poly,
                xfact=UPSAMPLE_FACTOR,
                yfact=UPSAMPLE_FACTOR,
                origin=(0, 0),
            )
            polys.append(poly)
        except Exception:
            continue
    return polys


def _rasterise_polygons(
    polys:      List[sg.Polygon],
    col_offset: int,
    row_offset: int,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """
    Rasterises building polygons into a single 256×256 patch binary mask.

    Steps:
      1. Define patch bounding box in 4096-px coordinates
      2. Clip each polygon to the patch bounding box
      3. Translate clipped polygons to local 0-indexed coordinates
      4. Rasterise into (patch_size × patch_size) uint8 array

    ARGS:
        polys      : polygons in 4096-px coordinate space
        col_offset : left pixel of this patch in the 4096-px image
        row_offset : top pixel of this patch in the 4096-px image
        patch_size : output size in pixels (default 256)

    RETURNS:
        uint8 (H, W) binary mask, 1 = building footprint, 0 = background
    """
    if not polys:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    patch_box = sg.box(
        col_offset,
        row_offset,
        col_offset + patch_size,
        row_offset + patch_size,
    )

    shapes = []
    for poly in polys:
        clipped = poly.intersection(patch_box)
        if clipped.is_empty:
            continue
        # Shift to local patch coordinates: (col_offset, row_offset) → (0, 0)
        local = shapely.affinity.translate(
            clipped,
            xoff=-col_offset,
            yoff=-row_offset,
        )
        shapes.append(local)

    if not shapes:
        return np.zeros((patch_size, patch_size), dtype=np.uint8)

    return rasterize(
        shapes,
        out_shape=(patch_size, patch_size),
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )


def _make_boundary(
    footprint:  np.ndarray,
    erosion_px: int = BOUNDARY_EROSION_PX,
) -> np.ndarray:
    """
    Boundary = footprint − erode(footprint).
    Produces a ring of width erosion_px around each building interior.
    Teaches the model where building edges are.
    """
    kernel   = np.ones((erosion_px * 2 + 1, erosion_px * 2 + 1), np.uint8)
    eroded   = cv2.erode(footprint, kernel, iterations=1)
    boundary = footprint - eroded
    return np.clip(boundary, 0, 1).astype(np.uint8)


def _make_contact(
    footprint: np.ndarray,
    dilate_px: int = CONTACT_DILATE_PX,
) -> np.ndarray:
    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    num_labels, labeled = cv2.connectedComponents(footprint, connectivity=8)

    if num_labels <= 2:
        return np.zeros_like(footprint, dtype=np.uint8)

    H, W = footprint.shape
    contact = np.zeros((H, W), dtype=np.uint8)

    # Precompute each building's dilation and bounding box
    dilated_masks = {}
    bboxes = {}
    for lid in range(1, num_labels):
        single = (labeled == lid).astype(np.uint8)
        dilated_masks[lid] = cv2.dilate(single, kernel, iterations=1)
        ys, xs = np.where(single)
        bboxes[lid] = (ys.min(), ys.max(), xs.min(), xs.max())

    # Check every pair of buildings
    label_ids = list(range(1, num_labels))
    for i in range(len(label_ids)):
        for j in range(i + 1, len(label_ids)):
            a_id = label_ids[i]
            b_id = label_ids[j]

            ay0, ay1, ax0, ax1 = bboxes[a_id]
            by0, by1, bx0, bx1 = bboxes[b_id]

            # Contact = pixels both dilations reach
            pair_contact = dilated_masks[a_id] & dilated_masks[b_id]

            # Remove building interiors
            pair_contact[footprint > 0] = 0

            # Clip to shared extent so contact never bleeds outside
            clip = np.zeros((H, W), dtype=np.uint8)
            y_min = max(ay0, by0)
            y_max = min(ay1, by1)

            if y_min <= y_max:
                # Horizontal neighbours — clip to shared row range
                x_min = min(ax0, bx0)
                x_max = max(ax1, bx1)
                clip[y_min:y_max + 1, x_min:x_max + 1] = 1
            else:
                # Vertical neighbours — clip to shared col range
                x_min = max(ax0, bx0)
                x_max = min(ax1, bx1)
                y_min2 = min(ay0, by0)
                y_max2 = max(ay1, by1)
                if x_min <= x_max:
                    clip[y_min2:y_max2 + 1, x_min:x_max + 1] = 1

            contact |= (pair_contact & clip)

    return contact


def _write_fbc_png(
    footprint: np.ndarray,
    boundary:  np.ndarray,
    contact:   np.ndarray,
    dst_path:  Path,
) -> None:
    """
    Writes 3-channel FBC mask as uint8 PNG.
    R = Footprint, G = Boundary, B = Contact.
    Values are 0 (background) or 255 (foreground).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fbc = np.stack([
        footprint * 255,
        boundary  * 255,
        contact   * 255,
    ], axis=-1).astype(np.uint8)
    Image.fromarray(fbc, mode="RGB").save(dst_path)


def _write_empty_fbc_png(dst_path: Path, patch_size: int = PATCH_SIZE) -> None:
    """
    Writes all-black 3-channel PNG.
    Used for patches with zero buildings — skips morphological ops.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.zeros((patch_size, patch_size, 3), dtype=np.uint8), mode="RGB"
    ).save(dst_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_fbc_mask(
    sidecars_dir: Path,
    patches_dir:  Path,
    raw_city_dir: Path,
    out_dir:      Path,
) -> Dict:
    """
    Generates FBC masks for ALL patches of ONE city.

    Called by run_pipeline.py → run_step5_mask_engineering()

    ARGS:
        sidecars_dir : JSON sidecar files from step 4.2
        patches_dir  : 256×256 .tif patches from step 4.1 (for presence check)
        raw_city_dir : .../train/{city}  (parent of labels_match_pix/)
        out_dir      : destination for {patch_id}_fbc.png files
    """
    city       = sidecars_dir.name
    labels_dir = raw_city_dir / "labels_match_pix"

    logger.info("=" * 60)
    logger.info("SUB-STEP 5.1 : FBC MASK  (Footprint / Boundary / Contact)")
    logger.info("=" * 60)
    logger.info(f"City          : {city}")
    logger.info(f"Labels dir    : {labels_dir}  "
                f"{'✓' if labels_dir.exists() else '*** MISSING ***'}")
    logger.info(f"Patches dir   : {patches_dir}")
    logger.info(f"Output        : {out_dir}")
    logger.info(f"Scale factor  : {UPSAMPLE_FACTOR}×  (1024-px labels → 4096-px space)")

    sidecar_files: List[Path] = sorted(sidecars_dir.glob("*.json"))
    if not sidecar_files:
        raise FileNotFoundError(f"No .json sidecars in {sidecars_dir}")

    logger.info(f"Sidecars      : {len(sidecar_files):,}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-month polygon cache so each GeoJSON is opened only once
    poly_cache:     Dict[str, List] = {}
    n_written       = 0
    n_missing_patch = 0
    n_empty         = 0
    n_geojson_miss  = 0

    for sc_idx, sc_path in enumerate(sidecar_files):
        sc         = _load_sidecar(sc_path)
        patch_id   = sc["patch_id"]
        month_name = sc.get("month_name", "unknown")
        col_offset = sc["col_offset_px"]
        row_offset = sc["row_offset_px"]

        # ── Guard: image patch must exist ────────────────────────────────
        if not (patches_dir / f"{patch_id}.tif").exists():
            n_missing_patch += 1
            continue

        # ── Load polygons for this month (cached) ─────────────────────────
        if month_name not in poly_cache:
            gj_path = _resolve_geojson_path(sc, raw_city_dir, labels_dir, month_name)

            if gj_path is not None:
                poly_cache[month_name] = _load_geojson_polygons(gj_path)
                logger.info(
                    f"  Loaded month {month_name}: "
                    f"{len(poly_cache[month_name])} buildings  ← {gj_path.name}"
                )
            else:
                logger.warning(f"  Month {month_name}: GeoJSON not found → empty FBC")
                poly_cache[month_name] = []
                n_geojson_miss += 1

        polys    = poly_cache[month_name]
        dst_path = out_dir / f"{patch_id}_fbc.png"

        # ── Short-circuit: no polygons for entire month ───────────────────
        if not polys:
            _write_empty_fbc_png(dst_path)
            n_empty  += 1
            n_written += 1
            continue

        # ── Rasterise footprint ───────────────────────────────────────────
        footprint = _rasterise_polygons(polys, col_offset, row_offset, PATCH_SIZE)

        # ── Short-circuit: no buildings in this patch window ──────────────
        if footprint.sum() == 0:
            _write_empty_fbc_png(dst_path)
            n_empty  += 1
            n_written += 1
            continue

        # ── Build all three FBC channels and write PNG ────────────────────
        boundary = _make_boundary(footprint)
        contact  = _make_contact(footprint)
        _write_fbc_png(footprint, boundary, contact, dst_path)
        n_written += 1

        if (sc_idx + 1) % 1000 == 0:
            logger.info(f"  Progress: {sc_idx+1:,}/{len(sidecar_files):,}")

    n_with_buildings = n_written - n_empty
    logger.info("\n" + "=" * 60)
    logger.info("FBC MASK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Sidecars        : {len(sidecar_files):,}")
    logger.info(f"Patch missing   : {n_missing_patch:,}  (skipped)")
    logger.info(f"Written total   : {n_written:,}")
    logger.info(f"  With buildings: {n_with_buildings:,}")
    logger.info(f"  Empty         : {n_empty:,}")
    logger.info(f"GeoJSON missing : {n_geojson_miss} months")

    if n_with_buildings == 0:
        logger.error(
            "*** ALL FBC MASKS ARE EMPTY — check GeoJSON path resolution ***\n"
            "    Run: python diagnose_fbc.py"
        )

    logger.info("STATUS : FBC mask generation complete ✓")
    logger.info("NEXT   : Sub-step 5.2 → Watershed")
    logger.info("=" * 60)

    return {
        "city":             city,
        "n_written":        n_written,
        "n_with_buildings": n_with_buildings,
        "n_empty":          n_empty,
        "n_missing_patch":  n_missing_patch,
        "n_geojson_miss":   n_geojson_miss,
        "output_dir":       str(out_dir),
        "status":           "complete",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    BASE_TRAIN = Path("data/raw/dataset/SN7_buildings_train/train")

    for city_dir in sorted(BASE_TRAIN.iterdir()):
        if not city_dir.is_dir():
            continue
        C = city_dir.name
        run_fbc_mask(
            sidecars_dir = Path(f"data/processed/tiling/sidecars/{C}"),
            patches_dir  = Path(f"data/processed/tiling/patches/{C}"),
            raw_city_dir = city_dir,
            out_dir      = Path(f"data/processed/mask_engineering_1/fbc/{C}"),
        )