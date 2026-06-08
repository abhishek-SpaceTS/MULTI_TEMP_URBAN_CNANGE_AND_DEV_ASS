"""
run_pipeline.py
================

MASTER PIPELINE SCRIPT — SpaceNet 7 Preprocessing
====================================================

Automatically processes ALL cities in the SpaceNet 7 dataset through
a complete preprocessing pipeline. Designed for large-scale, unattended
execution with robust error handling and progress tracking.

FEATURES:
    ✓ Automatic city discovery (no manual configuration)
    ✓ Smart folder synchronization (auto-creates output directories)
    ✓ Graceful error handling (one failure doesn't stop the whole pipeline)
    ✓ Detailed progress tracking with time estimates
    ✓ Comprehensive logging (both console + error_log.txt)
    ✓ Resume capability (skip already-processed cities)
    ✓ Selective cleanup (keeps only final outputs except for gold city)
    ✓ Multi-processing (4 cities in parallel)

USAGE:
    # Process all cities with 4 parallel workers (recommended):
    python run_pipeline.py

    # Test on a single city first:
    python run_pipeline.py --city L15-0331E-1257N_1327_3160_13

    # Resume from a specific step:
    python run_pipeline.py --from-step 3

    # Process specific steps only:
    python run_pipeline.py --from-step 2 --to-step 4
    
    # Adjust number of parallel workers:
    python run_pipeline.py --workers 2

PIPELINE STEPS:
    Step 1 — Alignment & Upsampling
        ├─ CRS Reprojection (ensure EPSG:3857)
        ├─ Coregistration (align time series using NCC)
        └─ Upsampling (bicubic 4× → 4096×4096)

    Step 2 — Quality Masking
        ├─ Cloud Mask (UDM if available, else spectral)
        ├─ Shadow Mask (dark pixel detection)
        └─ Vegetation Mask (proxy-NDVI + combined mask)

    Step 3 — Radiometric Normalization (mask-aware)
        ├─ IR-MAD (histogram matching)
        ├─ PIF Regression (pseudo-invariant features)
        └─ CIELab Matching (perceptual color consistency)

    Step 4 — Tiling
        ├─ Chipping (4096×4096 → 256×256 patches)
        └─ JSON Sidecars (provenance metadata)

    Step 5 — Label Engineering
        ├─ FBC Mask (Footprint/Boundary/Contact 3-channel)
        └─ Watershed (separate touching buildings)

TIME ESTIMATE:
    ~15-25 min per city × 60 cities ÷ 4 workers = ~4-6 hours total
    With cleanup: reduces storage from ~720GB to ~120GB

DATA PATHS:
    Raw data    : data/raw/dataset/SN7_buildings_train/train/{city}/
    Processed   : data/processed/{step}/{city}/
    Error log   : error_log.txt

DISK USAGE:
    With selective cleanup: ~2 GB per city (final outputs only)
    Gold city (full): ~12 GB (all intermediate steps preserved)
    Total for 60 cities: ~120-140 GB
"""

import os
import sys
import time
import argparse
import logging
import traceback
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Gold city: keep ALL intermediate outputs for analysis/debugging
GOLD_CITY = "L15-0331E-1257N_1327_3160_13"

# Number of parallel workers (cities processed simultaneously)
DEFAULT_WORKERS = 8

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Thread-safe logging
log_lock = threading.Lock()

class CityPrefixLogger:
    """
    Custom logger that adds city ID prefix to all messages.
    
    This helps distinguish between logs from different cities
    when running in multi-processing mode.
    
    Example output:
        [L15-0331E-1257N_1327_3160_13] Starting Step 1...
        [L15-0424E-1289N_1416_3208_13] Starting Step 1...
    """
    
    def __init__(self, city_id: str):
        self.city_id = city_id
        self.prefix = f"[{city_id}]"
    
    def _log(self, level: str, message: str):
        """Thread-safe logging with city prefix."""
        with log_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} {level:8s} {self.prefix} {message}")
    
    def info(self, message: str):
        self._log("[INFO]", message)
    
    def warning(self, message: str):
        self._log("[WARNING]", message)
    
    def error(self, message: str):
        self._log("[ERROR]", message)
    
    def debug(self, message: str):
        self._log("[DEBUG]", message)


# Global logger (used for non-city-specific messages)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Error log file for tracking failures
ERROR_LOG_FILE = Path("error_log.txt")


def log_error(city: str, step: int, error: Exception):
    """
    Logs detailed error information to error_log.txt.
   
    This allows you to review failures after the pipeline completes
    without stopping execution for every single error.
   
    Args:
        city: Name of the city where error occurred
        step: Pipeline step number that failed
        error: The exception that was raised
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_lock:
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"CITY:      {city}\n")
            f.write(f"STEP:      {step}\n")
            f.write(f"ERROR:     {str(error)}\n")
            f.write(f"TRACEBACK:\n{traceback.format_exc()}\n")
            f.write(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PATHS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Where SpaceNet 7 raw data lives
RAW_DATA_DIR = Path("data/raw/dataset/SN7_buildings_train/train")

# Where all processed outputs go
PROC_BASE_DIR = Path("data/processed")

# Output subdirectories for each pipeline step
OUTPUT_DIRS = {
    "alignment": {
        "reprojected": PROC_BASE_DIR / "alignment" / "reprojected",
        "coregistered": PROC_BASE_DIR / "alignment" / "coregistered",
        "upsampled": PROC_BASE_DIR / "alignment" / "upsampled",
    },
    "masking": {
        "cloud": PROC_BASE_DIR / "masking" / "cloud",
        "shadow": PROC_BASE_DIR / "masking" / "shadow",
        "vegetation": PROC_BASE_DIR / "masking" / "vegetation",
    },
    "normalization": {
        "irmad": PROC_BASE_DIR / "normalization" / "irmad",
        "pif": PROC_BASE_DIR / "normalization" / "pif",
        "cielab": PROC_BASE_DIR / "normalization" / "cielab",
    },
    "tiling": {
        "patches": PROC_BASE_DIR / "tiling" / "patches",
        "sidecars": PROC_BASE_DIR / "tiling" / "sidecars",
    },
    "mask_engineering": {
        "fbc": PROC_BASE_DIR / "mask_engineering" / "fbc",
        "watershed": PROC_BASE_DIR / "mask_engineering" / "watershed",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CITY DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_all_cities() -> List[str]:
    """
    Automatically discovers all city folders in the raw dataset.
   
    SpaceNet 7 cities follow the naming pattern:
        L15-{longitude}E-{latitude}N_{x}_{y}_{z}
   
    Example: L15-0331E-1257N_1327_3160_13
   
    Returns:
        List of city folder names sorted alphabetically
       
    Raises:
        FileNotFoundError: If RAW_DATA_DIR doesn't exist
    """
    if not RAW_DATA_DIR.exists():
        error_msg = (
            f"❌ Raw data directory not found: {RAW_DATA_DIR}\n"
            f"   Expected path: data/raw/dataset/SN7_buildings_train/train/\n"
            f"   Please check that your SpaceNet 7 dataset is extracted correctly."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
   
    # Find all directories starting with "L15" (SpaceNet 7 naming convention)
    cities = sorted([
        folder.name
        for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir() and folder.name.startswith("L15")
    ])
   
    if not cities:
        logger.warning(f"⚠️  No cities found in {RAW_DATA_DIR}")
        logger.warning("    Expected folders matching pattern: L15-*")
    else:
        logger.info(f"✓ Discovered {len(cities)} cities in dataset")
        # Show a sample for verification
        sample = cities[:3] + (["..."] if len(cities) > 3 else [])
        logger.info(f"  Sample: {', '.join(sample)}")
   
    return cities


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT FOLDER SYNCHRONIZATION
# ══════════════════════════════════════════════════════════════════════════════

def ensure_output_directories(city: str):
    """
    Creates all necessary output directories for a city.
   
    This ensures that when preprocessing steps try to save outputs,
    the destination folders already exist. Prevents "FileNotFoundError"
    during long unattended runs.
   
    Directory structure created:
        data/processed/
        ├── alignment/
        │   ├── reprojected/{city}/
        │   ├── coregistered/{city}/
        │   └── upsampled/{city}/
        ├── masking/
        │   ├── cloud/{city}/
        │   ├── shadow/{city}/
        │   └── vegetation/{city}/
        ├── normalization/
        │   ├── irmad/{city}/
        │   ├── pif/{city}/
        │   └── cielab/{city}/
        ├── tiling/
        │   ├── patches/{city}/
        │   └── sidecars/{city}/
        └── mask_engineering/
            ├── fbc/{city}/
            └── watershed/{city}/
   
    Args:
        city: Name of the city folder
    """
    created_dirs = []
   
    for category, subdirs in OUTPUT_DIRS.items():
        for subdir_name, base_path in subdirs.items():
            city_dir = base_path / city
            if not city_dir.exists():
                city_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.append(city_dir.relative_to(PROC_BASE_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# SELECTIVE CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_intermediate_files(city: str, city_logger: CityPrefixLogger):
    """
    Removes intermediate processing files to save disk space.
    
    CLEANUP POLICY:
        Gold City (GOLD_CITY):
            - Keep ALL folders (alignment, masking, normalization, tiling, mask_engineering)
            - Useful for debugging, analysis, and visualization
        
        Other Cities:
            - DELETE: alignment, masking, normalization folders
            - KEEP: tiling and mask_engineering (needed for training)
    
    This reduces storage from ~12 GB/city to ~2 GB/city while preserving
    the final outputs needed for model training.
    
    Args:
        city: Name of the city to clean up
        city_logger: Logger instance with city prefix
    """
    # Never clean up the gold city
    if city == GOLD_CITY:
        city_logger.info(f"✨ Gold city detected — preserving all intermediate files")
        return
    
    city_logger.info("🧹 Starting selective cleanup...")
    
    # Folders to remove (intermediate processing steps)
    cleanup_categories = ["alignment", "masking", "normalization"]
    
    total_freed = 0  # Track disk space freed (in bytes)
    
    for category in cleanup_categories:
        for subdir_name, base_path in OUTPUT_DIRS[category].items():
            city_dir = base_path / city
            
            if city_dir.exists():
                try:
                    # Calculate size before deletion
                    dir_size = sum(
                        f.stat().st_size 
                        for f in city_dir.rglob('*') 
                        if f.is_file()
                    )
                    
                    # Remove directory and all contents
                    shutil.rmtree(city_dir)
                    total_freed += dir_size
                    
                    city_logger.debug(f"  Removed {category}/{subdir_name}/ "
                                     f"({dir_size / 1e6:.1f} MB)")
                
                except Exception as e:
                    city_logger.warning(f"  Failed to remove {category}/{subdir_name}/: {e}")
    
    # Log total space freed
    if total_freed > 0:
        city_logger.info(f"✓ Cleanup complete — freed {total_freed / 1e9:.2f} GB")
    else:
        city_logger.info("✓ Cleanup complete — no files to remove")


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETION CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def count_files(directory: Path, pattern: str = "*") -> int:
    """
    Counts files matching a pattern in a directory.
   
    Args:
        directory: Path to search
        pattern: Glob pattern (default: all files)
       
    Returns:
        Number of matching files (0 if directory doesn't exist)
    """
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def is_step_complete(city: str, step: int) -> bool:
    """
    Checks if a pipeline step is already complete for a city.
   
    This allows the pipeline to resume gracefully after interruptions
    and skip cities that have already been processed.
   
    Completion criteria per step:
        Step 1: At least 18 .tif files in upsampled/ folder
        Step 2: At least 18 combined masks in vegetation/ folder
        Step 3: At least 18 .tif files in cielab/ folder
        Step 4: At least 100 .tif patches in patches/ folder
        Step 5: At least 100 _fbc.png files in fbc/ folder
   
    Args:
        city: Name of the city
        step: Pipeline step number (1-5)
       
    Returns:
        True if step is complete, False otherwise

    """
    fbc_dir = OUTPUT_DIRS["mask_engineering"]["fbc"] / city
    if count_files(fbc_dir, "*_fbc.png") >= 100:
        return True
        
    if step == 1:
        # Check upsampled images
        upsampled_dir = OUTPUT_DIRS["alignment"]["upsampled"] / city
        return count_files(upsampled_dir, "*.tif") >= 18
   
    elif step == 2:
        # Check combined quality masks
        veg_dir = OUTPUT_DIRS["masking"]["vegetation"] / city
        return count_files(veg_dir, "*_combined.tif") >= 18
   
    elif step == 3:
        # Check normalized images
        cielab_dir = OUTPUT_DIRS["normalization"]["cielab"] / city
        return count_files(cielab_dir, "*.tif") >= 18
   
    elif step == 4:
        # Check tiled patches
        patches_dir = OUTPUT_DIRS["tiling"]["patches"] / city
        return count_files(patches_dir, "*.tif") >= 100
   
    elif step == 5:
        # Check FBC masks
        fbc_dir = OUTPUT_DIRS["mask_engineering"]["fbc"] / city
        return count_files(fbc_dir, "*_fbc.png") >= 100
   
    else:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEP RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_step1_alignment(city: str, city_logger: CityPrefixLogger):
    """
    Step 1: Alignment & Upsampling
   
    Sub-steps:
        1.1 CRS Reprojection → Ensure all images are in EPSG:3857
        1.2 Coregistration → Align time series using phase correlation
        1.3 Upsampling → Bicubic 4× interpolation to 4096×4096
   
    Args:
        city: Name of the city to process
        city_logger: Logger instance with city prefix
    """
    from preprocessing.alignment.crs_reprojection import run_crs_reprojection
    from preprocessing.alignment.coregistration import run_coregistration
    from preprocessing.alignment.upsampling import run_upsampling
   
    # Define input/output paths
    raw_dir = RAW_DATA_DIR / city
    reproj_dir = OUTPUT_DIRS["alignment"]["reprojected"] / city
    coreg_dir = OUTPUT_DIRS["alignment"]["coregistered"] / city
    upsamp_dir = OUTPUT_DIRS["alignment"]["upsampled"] / city
   
    city_logger.info("    [1.1] CRS Reprojection → EPSG:3857")
    run_crs_reprojection(raw_dir, reproj_dir)
   
    city_logger.info("    [1.2] Coregistration (NCC phase correlation)")
    run_coregistration(reproj_dir, coreg_dir)
   
    city_logger.info("    [1.3] Upsampling → 4096×4096 (bicubic 4×)")
    run_upsampling(coreg_dir, upsamp_dir)


def run_step2_masking(city: str, city_logger: CityPrefixLogger):
    """
    Step 2: Quality Masking
   
    Sub-steps:
        2.1 Cloud Mask → UDM if available, else spectral fallback
        2.2 Shadow Mask → Dark pixel detection with blue-shift
        2.3 Vegetation Mask → Proxy-NDVI + combined quality mask
   
    Handles both masked_images and UDM files from raw data.
   
    Args:
        city: Name of the city to process
        city_logger: Logger instance with city prefix
    """
    from preprocessing.masking.cloud_mask import run_cloud_masking
    from preprocessing.masking.shadow_mask import run_shadow_masking
    from preprocessing.masking.vegetation_mask import run_vegetation_masking
   
    # Define input/output paths
    raw_dir = RAW_DATA_DIR / city  # For UDM files
    upsamp_dir = OUTPUT_DIRS["alignment"]["upsampled"] / city
    cloud_dir = OUTPUT_DIRS["masking"]["cloud"] / city
    shadow_dir = OUTPUT_DIRS["masking"]["shadow"] / city
    veg_dir = OUTPUT_DIRS["masking"]["vegetation"] / city
   
    city_logger.info("    [2.1] Cloud Mask (UDM or spectral)")
    run_cloud_masking(raw_dir, upsamp_dir, cloud_dir)
   
    city_logger.info("    [2.2] Shadow Mask (dark pixel detection)")
    run_shadow_masking(upsamp_dir, cloud_dir, shadow_dir)
   
    city_logger.info("    [2.3] Vegetation Mask + Combined quality mask")
    run_vegetation_masking(upsamp_dir, cloud_dir, shadow_dir, veg_dir)


def run_step3_normalization(city: str, city_logger: CityPrefixLogger):
    """
    Step 3: Radiometric Normalization (mask-aware)
   
    Sub-steps:
        3.1 IR-MAD → Histogram matching on clear pixels
        3.2 PIF Regression → OLS on pseudo-invariant features
        3.3 CIELab Matching → Perceptual color consistency
   
    All normalizations use the combined quality mask from Step 2
    to exclude clouds, shadows, and vegetation from statistics.
   
    Args:
        city: Name of the city to process
        city_logger: Logger instance with city prefix
    """
    from preprocessing.normalization.irmad import run_irmad_normalisation
    from preprocessing.normalization.pif_regression import run_pif_regression
    from preprocessing.normalization.cielab_matching import run_cielab_matching
   
    # Define input/output paths
    upsamp_dir = OUTPUT_DIRS["alignment"]["upsampled"] / city
    mask_dir = OUTPUT_DIRS["masking"]["vegetation"] / city  # Combined masks
    irmad_dir = OUTPUT_DIRS["normalization"]["irmad"] / city
    pif_dir = OUTPUT_DIRS["normalization"]["pif"] / city
    cielab_dir = OUTPUT_DIRS["normalization"]["cielab"] / city
   
    city_logger.info("    [3.1] IR-MAD (mask-aware histogram matching)")
    run_irmad_normalisation(upsamp_dir, mask_dir, irmad_dir)
   
    city_logger.info("    [3.2] PIF Regression (pseudo-invariant features)")
    run_pif_regression(upsamp_dir, irmad_dir, mask_dir, pif_dir)
   
    city_logger.info("    [3.3] CIELab Color Matching")
    run_cielab_matching(pif_dir, mask_dir, cielab_dir)


def run_step4_tiling(city: str, city_logger: CityPrefixLogger):
    """
    Step 4: Tiling
   
    Sub-steps:
        4.1 Chipping → 4096×4096 images split into 256×256 patches
        4.2 JSON Sidecars → Provenance metadata for each patch
   
    Creates ~256 patches per 4096×4096 image (16×16 grid).
    With ~18 images per city → ~4,600 patches per city.
   
    Args:
        city: Name of the city to process
        city_logger: Logger instance with city prefix
    """
    from preprocessing.tiling.chipping import run_chipping
    from preprocessing.tiling.json_sidecars import run_json_sidecars
   
    # Define input/output paths
    cielab_dir = OUTPUT_DIRS["normalization"]["cielab"] / city
    patches_dir = OUTPUT_DIRS["tiling"]["patches"] / city
    sidecars_dir = OUTPUT_DIRS["tiling"]["sidecars"] / city
   
    city_logger.info("    [4.1] Chipping (4096×4096 → 256×256)")
    run_chipping(cielab_dir, patches_dir)
   
    city_logger.info("    [4.2] JSON Sidecars (metadata generation)")
    run_json_sidecars(patches_dir, RAW_DATA_DIR / city, sidecars_dir)


def run_step5_mask_engineering(city: str, city_logger: CityPrefixLogger):
    """
    Step 5: Label Engineering
   
    Sub-steps:
        5.1 FBC Mask → Footprint/Boundary/Contact 3-channel labels
        5.2 Watershed → Separate touching buildings for instance segmentation
   
    Creates training-ready semantic segmentation targets.
   
    Args:
        city: Name of the city to process
        city_logger: Logger instance with city prefix
    """
    from preprocessing.mask_engineering.fbc_mask_1 import run_fbc_mask
    from preprocessing.mask_engineering.watershed import run_watershed
   
    # Define input/output paths
    patches_dir = OUTPUT_DIRS["tiling"]["patches"] / city
    sidecars_dir = OUTPUT_DIRS["tiling"]["sidecars"] / city
    fbc_dir = OUTPUT_DIRS["mask_engineering"]["fbc"] / city
    ws_dir = OUTPUT_DIRS["mask_engineering"]["watershed"] / city
   
    city_logger.info("    [5.1] FBC Mask (Footprint/Boundary/Contact)")
    run_fbc_mask(sidecars_dir, patches_dir, RAW_DATA_DIR / city, fbc_dir)
   
    city_logger.info("    [5.2] Watershed (building separation)")
    run_watershed(fbc_dir, ws_dir)


# ══════════════════════════════════════════════════════════════════════════════
# STEP REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

PIPELINE_STEPS = {
    1: ("Alignment & Upsampling", run_step1_alignment),
    2: ("Quality Masking", run_step2_masking),
    3: ("Radiometric Normalization", run_step3_normalization),
    4: ("Tiling & Metadata", run_step4_tiling),
    5: ("Label Engineering", run_step5_mask_engineering),
}


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CITY PROCESSOR (runs in worker process)
# ══════════════════════════════════════════════════════════════════════════════

def process_single_city(
    city: str,
    city_idx: int,
    total_cities: int,
    from_step: int,
    to_step: int,
) -> Dict[str, any]:
    """
    Processes a single city through the pipeline.
    
    This function is designed to run in a separate process via
    ProcessPoolExecutor. Each city gets its own process, allowing
    multiple cities to be processed in parallel.
    
    Args:
        city: Name of the city to process
        city_idx: Index of this city (for progress tracking)
        total_cities: Total number of cities being processed
        from_step: Starting pipeline step
        to_step: Ending pipeline step
    
    Returns:
        Dictionary containing:
            - city: City name
            - success: True if all steps completed
            - failed_step: Step number that failed (if any)
            - error: Error message (if failed)
            - processing_time: Total time in seconds
    """
    city_start = time.time()
    city_logger = CityPrefixLogger(city)
    
    # City header
    city_logger.info(f"{'='*60}")
    city_logger.info(f"🌆 PROCESSING [{city_idx}/{total_cities}]")
    city_logger.info(f"{'='*60}")
    
    try:
        # Ensure all output directories exist
        ensure_output_directories(city)
        
        # ── Process Each Step ──────────────────────────────────────────
        for step_num in range(from_step, to_step + 1):
            step_name, step_func = PIPELINE_STEPS[step_num]
            
            # Check if step already complete (resume capability)
            if is_step_complete(city, step_num):
                city_logger.info(f"  ✓ Step {step_num} [{step_name}] — SKIPPED (already complete)")
                continue
            
            # Run the step
            step_start = time.time()
            city_logger.info(f"  ⏳ Step {step_num} [{step_name}] — RUNNING...")
            
            try:
                step_func(city, city_logger)
                step_elapsed = time.time() - step_start
                city_logger.info(f"  ✓ Step {step_num} COMPLETE in {step_elapsed:.1f}s")
            
            except Exception as e:
                step_elapsed = time.time() - step_start
                city_logger.error(f"  ❌ Step {step_num} FAILED after {step_elapsed:.1f}s")
                city_logger.error(f"     Error: {str(e)}")
                
                # Log to error file
                log_error(city, step_num, e)
                
                # Return failure info
                return {
                    "city": city,
                    "success": False,
                    "failed_step": step_num,
                    "step_name": step_name,
                    "error": str(e),
                    "processing_time": time.time() - city_start,
                }
        
        # ── All Steps Complete — Run Cleanup ──────────────────────────
        if to_step == 5:  # Only cleanup if full pipeline completed
            cleanup_intermediate_files(city, city_logger)
        
        # ── City Complete ──────────────────────────────────────────────
        city_elapsed = time.time() - city_start
        city_logger.info(f"✅ COMPLETE in {city_elapsed/60:.1f} minutes")
        
        return {
            "city": city,
            "success": True,
            "processing_time": city_elapsed,
        }
    
    except Exception as e:
        # Catch-all for unexpected errors
        city_elapsed = time.time() - city_start
        city_logger.error(f"❌ FAILED with unexpected error:")
        city_logger.error(f"   {str(e)}")
        log_error(city, -1, e)  # -1 indicates error outside step execution
        
        return {
            "city": city,
            "success": False,
            "failed_step": -1,
            "step_name": "Unexpected Error",
            "error": str(e),
            "processing_time": city_elapsed,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    Thread-safe progress tracker for multi-processing pipeline.
   
    Features:
        - City completion percentage
        - Time elapsed and estimated time remaining
        - Average processing time per city
        - Real-time updates during execution
        - Thread-safe updates
    """
   
    def __init__(self, total_cities: int):
        self.total_cities = total_cities
        self.completed_cities = 0
        self.failed_cities = 0
        self.start_time = time.time()
        self.city_times = []  # Track individual city processing times
        self.lock = threading.Lock()
   
    def update(self, city_time_seconds: float, success: bool = True):
        """
        Updates progress after completing a city.
       
        Thread-safe: can be called from multiple worker processes.
       
        Args:
            city_time_seconds: Time taken to process the city
            success: Whether the city processed successfully
        """
        with self.lock:
            if success:
                self.completed_cities += 1
                self.city_times.append(city_time_seconds)
            else:
                self.failed_cities += 1
   
    def get_stats(self) -> Dict[str, any]:
        """
        Calculates current progress statistics.
       
        Thread-safe: acquires lock before reading shared state.
       
        Returns:
            Dictionary with progress metrics
        """
        with self.lock:
            total_processed = self.completed_cities + self.failed_cities
            elapsed_seconds = time.time() - self.start_time
           
            # Calculate average time per city (only from successful runs)
            avg_time_per_city = (
                sum(self.city_times) / len(self.city_times)
                if self.city_times else 0
            )
           
            # Estimate remaining time
            cities_remaining = self.total_cities - total_processed
            estimated_remaining_seconds = avg_time_per_city * cities_remaining
           
            return {
                "completed": self.completed_cities,
                "failed": self.failed_cities,
                "total_processed": total_processed,
                "total_cities": self.total_cities,
                "percent_complete": (total_processed / self.total_cities * 100)
                                   if self.total_cities > 0 else 0,
                "elapsed_seconds": elapsed_seconds,
                "avg_time_per_city": avg_time_per_city,
                "estimated_remaining_seconds": estimated_remaining_seconds,
            }
   
    def print_summary(self):
        """Prints a formatted progress summary to console."""
        stats = self.get_stats()
       
        elapsed_str = str(timedelta(seconds=int(stats["elapsed_seconds"])))
        remaining_str = str(timedelta(seconds=int(stats["estimated_remaining_seconds"])))
        avg_str = str(timedelta(seconds=int(stats["avg_time_per_city"])))
       
        with log_lock:
            print(f"\n{'─'*70}")
            print(f"📊 PROGRESS SUMMARY")
            print(f"{'─'*70}")
            print(f"  Cities Completed : {stats['completed']}/{stats['total_cities']} "
                  f"({stats['percent_complete']:.1f}%)")
            print(f"  Cities Failed    : {stats['failed']}")
            print(f"  Time Elapsed     : {elapsed_str}")
            print(f"  Avg Time/City    : {avg_str}")
            print(f"  Est. Remaining   : {remaining_str}")
            print(f"{'─'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR (Multi-Processing)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    cities: List[str],
    from_step: int = 1,
    to_step: int = 5,
    max_workers: int = DEFAULT_WORKERS,
) -> Dict[str, any]:
    """
    Main pipeline orchestrator with multi-processing support.
   
    Uses ProcessPoolExecutor to process multiple cities in parallel,
    dramatically reducing total processing time for large datasets.
   
    Features:
        - Parallel processing of cities (default: 4 workers)
        - Thread-safe logging and progress tracking
        - Graceful error handling per city
        - Automatic cleanup of intermediate files
        - Resume capability
   
    Args:
        cities: List of city names to process
        from_step: Starting step number (inclusive)
        to_step: Ending step number (inclusive)
        max_workers: Number of parallel worker processes
       
    Returns:
        Dictionary with execution summary:
            - completed: List of successfully processed cities
            - failed: List of dicts with error details
            - total_time_seconds: Total execution time
    """
    # ── Pipeline Header ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("🛰️  SPACENET 7 — MULTI-TEMPORAL PREPROCESSING PIPELINE")
    print("="*70)
    print(f"  Cities to Process : {len(cities)}")
    print(f"  Parallel Workers  : {max_workers}")
    print(f"  Pipeline Steps    : {from_step} → {to_step}")
    print(f"  Gold City         : {GOLD_CITY}")
    print(f"  Raw Data Path     : {RAW_DATA_DIR}")
    print(f"  Output Path       : {PROC_BASE_DIR}")
    print(f"  Started At        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
   
    # Initialize tracking
    pipeline_start = time.time()
    tracker = ProgressTracker(total_cities=len(cities))
    completed_cities = []
    failed_cities = []
   
    # Clear error log at start
    if ERROR_LOG_FILE.exists():
        ERROR_LOG_FILE.unlink()
    ERROR_LOG_FILE.touch()
   
    # ── Process Cities in Parallel ─────────────────────────────────────────
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all cities to the executor
        future_to_city = {
            executor.submit(
                process_single_city,
                city,
                idx,
                len(cities),
                from_step,
                to_step
            ): city
            for idx, city in enumerate(cities, start=1)
        }
        
        # Process results as they complete
        for future in as_completed(future_to_city):
            city = future_to_city[future]
            
            try:
                result = future.result()
                
                # Update tracking
                if result["success"]:
                    completed_cities.append(result["city"])
                    tracker.update(result["processing_time"], success=True)
                else:
                    failed_cities.append(result)
                    tracker.update(result["processing_time"], success=False)
                
                # Print progress update every 5 cities or at end
                total_processed = len(completed_cities) + len(failed_cities)
                if (total_processed % 5 == 0) or (total_processed == len(cities)):
                    tracker.print_summary()
            
            except Exception as e:
                logger.error(f"[{city}] ❌ Worker process failed: {e}")
                failed_cities.append({
                    "city": city,
                    "step": -1,
                    "step_name": "Worker Process Error",
                    "error": str(e),
                })
                tracker.update(0, success=False)
   
    # ── Pipeline Complete ──────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
   
    print("\n" + "="*70)
    print("🏁 PIPELINE COMPLETE")
    print("="*70)
    print(f"  Total Time        : {timedelta(seconds=int(total_elapsed))}")
    print(f"  Cities Completed  : {len(completed_cities)}/{len(cities)}")
    print(f"  Cities Failed     : {len(failed_cities)}")
    print(f"  Success Rate      : {len(completed_cities)/len(cities)*100:.1f}%")
   
    if failed_cities:
        print(f"\n  ⚠️  {len(failed_cities)} cities failed — details in {ERROR_LOG_FILE}")
        print("\n  Failed cities:")
        for failure in failed_cities[:10]:  # Show first 10
            print(f"    • {failure['city']} (Step {failure.get('failed_step', '?')}: "
                  f"{failure.get('step_name', 'Unknown')})")
        if len(failed_cities) > 10:
            print(f"    ... and {len(failed_cities) - 10} more")
   
    print("\n" + "="*70)
    print("📁 OUTPUT LOCATIONS:")
    print("="*70)
    print(f"  TRAINING DATA (all cities):")
    print(f"    └─ Patches       → {OUTPUT_DIRS['tiling']['patches']}")
    print(f"    └─ FBC Masks     → {OUTPUT_DIRS['mask_engineering']['fbc']}")
    print(f"    └─ Watershed     → {OUTPUT_DIRS['mask_engineering']['watershed']}")
    print(f"\n  GOLD CITY ({GOLD_CITY}):")
    print(f"    └─ All Steps     → data/processed/*/{GOLD_CITY}/")
   
    print("\n" + "="*70)
    print("💾 DISK SPACE SAVED:")
    print("="*70)
    non_gold_cities = len(cities) - (1 if GOLD_CITY in cities else 0)
    estimated_savings = non_gold_cities * 10  # ~10 GB saved per non-gold city
    print(f"  Intermediate files cleaned for {non_gold_cities} cities")
    print(f"  Estimated space saved: ~{estimated_savings} GB")
   
    print("\n" + "="*70)
    print("🚀 NEXT STEPS:")
    print("="*70)
    print("  1. Review error log (if any failures): cat error_log.txt")
    print("  2. Verify outputs: ls -lh data/processed/tiling/patches/")
    print("  3. Start model training: python segmentation/train.py")
    print("  4. Analyze gold city: ls -lh data/processed/*/L15-0331E-1257N_1327_3160_13/")
    print("="*70 + "\n")
   
    return {
        "completed": completed_cities,
        "failed": failed_cities,
        "total_time_seconds": total_elapsed,
        "success_rate": len(completed_cities) / len(cities) if cities else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="SpaceNet 7 Multi-Temporal Preprocessing Pipeline (Multi-Processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all cities with 4 parallel workers (recommended):
  python run_pipeline.py

  # Test on a single city first:
  python run_pipeline.py --city L15-0331E-1257N_1327_3160_13

  # Resume from step 3 with 2 workers:
  python run_pipeline.py --from-step 3 --workers 2

  # Run only tiling and label engineering:
  python run_pipeline.py --from-step 4 --to-step 5

Features:
  - Multi-processing: 4 cities processed in parallel (configurable)
  - Selective cleanup: intermediate files removed (except gold city)
  - Disk savings: ~500 GB saved vs. keeping all intermediate outputs
  - Resume capability: skip already-processed cities
  - Thread-safe logging: [CityID] prefix for clarity
        """,
    )
   
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Process a single city (for testing). Example: L15-0331E-1257N_1327_3160_13",
    )
   
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Start from this step (default: 1)",
    )
   
    parser.add_argument(
        "--to-step",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Stop after this step (default: 5)",
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
   
    args = parser.parse_args()
   
    # ── Validate Arguments ─────────────────────────────────────────────────
    if args.from_step > args.to_step:
        logger.error("❌ --from-step cannot be greater than --to-step")
        sys.exit(1)
    
    if args.workers < 1:
        logger.error("❌ --workers must be at least 1")
        sys.exit(1)
    
    if args.workers > 8:
        logger.warning(f"⚠️  --workers={args.workers} is very high. "
                      f"Recommended: 2-6 depending on CPU/RAM")
   
    # ── Build City List ────────────────────────────────────────────────────
    try:
        if args.city:
            # Single city mode
            city_path = RAW_DATA_DIR / args.city
            if not city_path.exists():
                logger.error(f"❌ City not found: {args.city}")
                logger.error(f"   Expected path: {city_path}")
                sys.exit(1)
           
            cities = [args.city]
            logger.info(f"🎯 Single city mode: {args.city}")
            
            # Single city: use 1 worker
            if args.workers > 1:
                logger.info(f"   Note: Using 1 worker for single city (--workers ignored)")
                args.workers = 1
       
        else:
            # Full dataset mode
            cities = discover_all_cities()
            if not cities:
                logger.error("❌ No cities found in dataset")
                sys.exit(1)
   
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
   
    # ── Run Pipeline ───────────────────────────────────────────────────────
    try:
        results = run_pipeline(
            cities=cities,
            from_step=args.from_step,
            to_step=args.to_step,
            max_workers=args.workers,
        )
       
        # Exit with appropriate code
        if results["failed"]:
            sys.exit(1)  # Some cities failed
        else:
            sys.exit(0)  # All successful
   
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        logger.info("   You can resume later with --from-step option")
        sys.exit(130)
   
    except Exception as e:
        logger.error(f"\n❌ Fatal error in pipeline:")
        logger.error(traceback.format_exc())
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()