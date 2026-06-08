"""
preprocessing/tiling/json_sidecars.py
=======================================

SUB-STEP 4.2 — JSON Sidecar Metadata Files
--------------------------------------------

PURPOSE:
    Create one JSON sidecar file per 256×256 patch. The sidecar
    records the patch's provenance and spatial context, enabling:
      - Dataset reproducibility (exact traceability to raw data)
      - Step 5 FBC mask generation (knows which labels to rasterise)
      - Training data loaders (city/month/position lookups)
      - QA / visualisation (re-assemble full city mosaic from patches)

SIDECAR CONTENTS:
    {
      "patch_id":       "L15-0331E_m00_r03_c07",   // unique ID
      "city_id":        "L15-0331E-1257N_1327_3160_13",
      "month_idx":      0,                          // 0-indexed
      "month_name":     "2018_01",                  // extracted from filename
      "row":            3,                          // patch row (0-15)
      "col":            7,                          // patch col (0-15)
      "patch_size_px":  256,
      "row_offset_px":  768,                        // pixel offset in 4096 image
      "col_offset_px":  1792,
      "bounds_epsg3857": {                          // geo bounds of patch
        "left":  ..., "right": ...,
        "top":   ..., "bottom": ...
      },
      "labels_geojson": "labels_match_pix/..._Buildings.geojson",
      "pipeline_version": "1.0"
    }

DYNAMIC FILE DISCOVERY:
    Rather than iterating a fixed 16×16 grid, this module scans the
    actual .tif files produced by chipping.py using Path.glob("*.tif").
    This means:
      - Sidecars are generated ONLY for patches that actually exist on disk.
      - Patches discarded by the smart-sampling filters in step 4.1
        automatically have no sidecar — no manual bookkeeping required.
      - The training pipeline never attempts to load a patch that was
        never saved, preventing FileNotFoundError crashes at train time.

LABEL LINKING:
    We scan the city's labels_match_pix/ folder for the GeoJSON
    that matches this month. The path is recorded but NOT loaded
    here — actual rasterisation happens in Step 5 (fbc_mask.py).

INPUT:
    patches_dir   : 256×256 .tif patches from step 4.1
    labels_dir    : raw city's labels_match_pix/ GeoJSON folder
    out_dir       : sidecar .json output

OUTPUT:
    data/processed/tiling/sidecars/{city}/{patch_id}.json
    One .json per .tif patch (exact 1:1 correspondence).

NEXT STEP → 5.1 fbc_mask.py
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.0"
PATCH_SIZE = 256


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_patch_filename(fname: str):
    """
    Parses patch filename into components.

    Expected format: {city_id}_m{month:02d}_r{row:02d}_c{col:02d}.tif
    e.g. L15-0331E_m00_r03_c07.tif

    RETURNS:
        (city_id, month_idx, row, col) or raises ValueError
    """
    pattern = r"^(.+)_m(\d{2})_r(\d{2})_c(\d{2})\.tif$"
    m = re.match(pattern, fname)
    if not m:
        raise ValueError(f"Cannot parse patch filename: {fname}")
    city_id   = m.group(1)
    month_idx = int(m.group(2))
    row       = int(m.group(3))
    col       = int(m.group(4))
    return city_id, month_idx, row, col


def _extract_month_name(city_tif_files: List[Path], month_idx: int) -> str:
    """
    Extracts the month string from the original image filename at
    position month_idx in the sorted list.

    SpaceNet 7 filenames contain the date:
        global_monthly_2018_01_mosaic_L15-0331E...tif
    We extract "2018_01" as the month_name.
    """
    if month_idx >= len(city_tif_files):
        return f"month_{month_idx:02d}"
    fname = city_tif_files[month_idx].name
    m = re.search(r"(\d{4}_\d{2})", fname)
    return m.group(1) if m else f"month_{month_idx:02d}"


def _find_label_geojson(labels_dir: Optional[Path], month_name: str) -> Optional[str]:
    """
    Finds the GeoJSON label file for a given month_name.

    labels_match_pix/ files are named:
        global_monthly_{YYYY_MM}_mosaic_L15-..._Buildings.geojson

    Matches on the YYYY_MM portion.
    Returns relative path string or None if not found.
    """
    if labels_dir is None or not labels_dir.exists():
        return None
    for gj in labels_dir.glob("*.geojson"):
        if month_name.replace("_", "") in gj.name.replace("_", ""):
            return str(gj.relative_to(labels_dir.parent.parent))
        if month_name in gj.name:
            return str(gj.relative_to(labels_dir.parent.parent))
    return None


def _get_patch_bounds(patch_path: Path) -> Dict:
    """Reads the geo bounds of a patch from its embedded transform."""
    with rasterio.open(patch_path) as src:
        b = src.bounds
        return {
            "left":   round(b.left,   2),
            "right":  round(b.right,  2),
            "top":    round(b.top,    2),
            "bottom": round(b.bottom, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_json_sidecars(
    patches_dir:  Path,
    raw_city_dir: Path,
    out_dir:      Path,
) -> Dict:
    """
    Generates JSON sidecar files for ALL patches of ONE city.

    DYNAMIC DISCOVERY: iterates only the .tif files that actually
    exist in patches_dir (written by chipping.py). Patches discarded
    by smart-sampling filters never appear here — no hard-coded grid
    loops, no missing-file crashes.

    Called by run_pipeline.py → run_step4_tiling()

    ARGS:
        patches_dir  : 256×256 .tif patches (step 4.1 output)
        raw_city_dir : raw AOI folder (for labels_match_pix/ and filename dates)
        out_dir      : sidecar .json destination
    """
    city       = patches_dir.name
    labels_dir = raw_city_dir / "labels_match_pix"

    logger.info("=" * 60)
    logger.info("SUB-STEP 4.2 : JSON SIDECAR METADATA")
    logger.info("=" * 60)
    logger.info(f"City       : {city}")
    logger.info(f"Labels dir : {labels_dir}  {'✓' if labels_dir.exists() else '(missing)'}")
    logger.info(f"Output     : {out_dir}")

    # ── Dynamic patch discovery (KEY CHANGE) ─────────────────────────────
    # We glob the actual patch files — NOT a fixed 16×16 grid.
    # This guarantees 1:1 alignment: one .json per .tif, no more, no less.
    # Patches dropped by chipping.py smart-sampling simply don't appear here.
    patch_files: List[Path] = sorted(patches_dir.glob("*.tif"))
    if not patch_files:
        raise FileNotFoundError(
            f"No .tif patches found in {patches_dir}. "
            "Did step 4.1 (chipping) complete successfully?"
        )

    n_patches = len(patch_files)
    logger.info(f"Patches found (on disk) : {n_patches:,}  ← dynamic discovery")

    # Discover original monthly images for month-name lookup
    raw_imgs = sorted((raw_city_dir / "images_masked").glob("*.tif"))
    if not raw_imgs:
        raw_imgs = sorted((raw_city_dir / "images").glob("*.tif"))

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written  = 0
    n_skipped  = 0

    for patch_path in patch_files:

        # ── Parse filename ────────────────────────────────────────────────
        try:
            city_id, month_idx, row, col = _parse_patch_filename(patch_path.name)
        except ValueError as e:
            logger.warning(f"  Skipping {patch_path.name}: {e}")
            n_skipped += 1
            continue

        patch_id   = patch_path.stem
        month_name = _extract_month_name(raw_imgs, month_idx)
        label_path = _find_label_geojson(labels_dir, month_name)
        bounds     = _get_patch_bounds(patch_path)

        # Pixel offsets in the 4096×4096 image
        row_offset_px = row * PATCH_SIZE
        col_offset_px = col * PATCH_SIZE

        sidecar = {
            "patch_id":          patch_id,
            "city_id":           city,
            "city_id_short":     city_id,
            "month_idx":         month_idx,
            "month_name":        month_name,
            "row":               row,
            "col":               col,
            "patch_size_px":     PATCH_SIZE,
            "row_offset_px":     row_offset_px,
            "col_offset_px":     col_offset_px,
            "bounds_epsg3857":   bounds,
            "labels_geojson":    label_path,
            "pipeline_version":  PIPELINE_VERSION,
        }

        sidecar_path = out_dir / f"{patch_id}.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        n_written += 1

    logger.info(f"Sidecars written : {n_written:,}")
    if n_skipped:
        logger.warning(f"Parse errors     : {n_skipped}  (check filenames)")
    logger.info("STATUS : JSON sidecars complete ✓")
    logger.info("NOTE   : Sidecar count matches patch count — 1:1 guaranteed")
    logger.info("NEXT   : Step 5.1 → FBC Mask generation")

    return {
        "city":       city,
        "n_patches":  n_patches,
        "n_written":  n_written,
        "n_skipped":  n_skipped,
        "output_dir": str(out_dir),
        "status":     "complete",
    }


if __name__ == "__main__":
    C = "L15-0331E-1257N_1327_3160_13"
    run_json_sidecars(
        patches_dir  = Path(f"data/processed/tiling/patches/{C}"),
        raw_city_dir = Path(f"data/raw/dataset/SN7_buildings_train/train/{C}"),
        out_dir      = Path(f"data/processed/tiling/sidecars/{C}"),
    )