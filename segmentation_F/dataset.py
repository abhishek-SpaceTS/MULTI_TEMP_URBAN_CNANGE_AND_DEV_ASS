"""
segmentation/dataset.py
========================

Dataset for TE-UNet v2 with fbc_ch=3 boundary supervision.

CHANGES FROM v1:
  1. fbc_ch=3 support — loads from FBC_DIR (colored PNG) not WATERSHED_DIR
     Returns all 3 channels: footprint (R), boundary (G), contact (B)
     Dataset auto-detects which dir to use based on fbc_ch setting.

  2. Change label still uses footprint channel only (ch 0 of FBC masks)
     Change = did a building footprint appear/disappear? Not boundary change.

  3. VISUALIZATION: footprint channel (ch 0) only → white=building, black=bg
     Boundary/contact are internal training signals only.

  4. fbc_ch=1 fallback: loads from WATERSHED_DIR as before (backwards compat).

KEY DESIGN:
  FBC_DIR   : colored PNG  R=footprint G=boundary B=contact  (fbc_ch=3)
  WATERSHED_DIR: grayscale PNG  white=building               (fbc_ch=1)
"""

import os
import re
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import rasterio
from PIL import Image
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

RAW_TRAIN_DIR = Path("data/raw/dataset/SN7_buildings_train/train")
PATCHES_DIR   = Path("data/processed/tiling/patches")
FBC_DIR       = Path("data/processed/mask_engineering/fbc")
WATERSHED_DIR = Path("data/processed/mask_engineering/watershed")

N_MONTHS          = 5
MIN_MONTH_GAP     = 12
SAMPLES_PER_EPOCH = 25_000
CHANGE_OVERSAMPLE = 0.7

# Normalization — computed from SpaceNet-7 training patches
NORM_MEAN = np.array([0.4736, 0.4196, 0.3154], dtype=np.float32)
NORM_STD  = np.array([0.2381, 0.1894, 0.1756], dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# CITY DISCOVERY AND SPLIT
# ══════════════════════════════════════════════════════════════

def _discover_cities() -> List[str]:
    if PATCHES_DIR.exists():
        cities = sorted([d.name for d in PATCHES_DIR.iterdir()
                         if d.is_dir() and d.name.startswith("L15")])
        if cities: return cities
    if RAW_TRAIN_DIR.exists():
        return sorted([d.name for d in RAW_TRAIN_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("L15")])
    return []


def create_city_splits(cities, train_ratio=0.7, val_ratio=0.15,
                       test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    cities_shuffled = cities.copy()
    random.Random(seed).shuffle(cities_shuffled)
    n_total = len(cities_shuffled)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    train_cities = cities_shuffled[:n_train]
    val_cities   = cities_shuffled[n_train: n_train + n_val]
    test_cities  = cities_shuffled[n_train + n_val:]
    logger.info(f"City split -> Train={len(train_cities)} | "
                f"Val={len(val_cities)} | Test={len(test_cities)}")
    return train_cities, val_cities, test_cities


ALL_CITIES = _discover_cities()
TRAIN_CITIES, VAL_CITIES, TEST_CITIES = create_city_splits(ALL_CITIES)
logger.info(f"Total cities discovered: {len(ALL_CITIES)}")


# ══════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════

def _build_geometric_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.Transpose(p=0.3),
    ], additional_targets={
        "image_m1": "image", "image_m2": "image",
        "image_m3": "image", "image_m4": "image",
        "mask_m0":  "mask",  "mask_m1":  "mask",
        "mask_m2":  "mask",  "mask_m3":  "mask",
        "mask_m4":  "mask",
    })


def _build_photometric_transform() -> A.Compose:
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.GaussNoise(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.CoarseDropout(p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
    ])


def _build_val_transform() -> A.Compose:
    return A.Compose([], additional_targets={
        "image_m1": "image", "image_m2": "image",
        "image_m3": "image", "image_m4": "image",
        "mask_m0":  "mask",  "mask_m1":  "mask",
        "mask_m2":  "mask",  "mask_m3":  "mask",
        "mask_m4":  "mask",
    })


# ══════════════════════════════════════════════════════════════
# FILENAME PARSING
# ══════════════════════════════════════════════════════════════

def _parse_patch_stem(stem: str) -> Optional[Dict]:
    m = re.match(r"^(.+?)_m(\d{2})_r(\d{2})_c(\d{2})$", stem)
    if not m: return None
    return {"location": m.group(1), "month_idx": int(m.group(2)),
            "row": int(m.group(3)), "col": int(m.group(4))}


# ══════════════════════════════════════════════════════════════
# I/O HELPERS
# ══════════════════════════════════════════════════════════════

def _load_tif_rgb(tif_path: Path) -> np.ndarray:
    """Load GeoTIFF as uint8 [H,W,3]."""
    try:
        with rasterio.open(tif_path) as src:
            r, g, b = src.read(1), src.read(2), src.read(3)
        return np.stack([r, g, b], axis=-1)
    except Exception:
        return np.zeros((256, 256, 3), dtype=np.uint8)


def _load_fbc_png_color(fbc_path: Path) -> np.ndarray:
    """
    Load colored FBC PNG as uint8 [H,W,3] — R=footprint G=boundary B=contact.
    Used when fbc_ch=3.
    """
    try:
        if not fbc_path.exists():
            return np.zeros((256, 256, 3), dtype=np.uint8)
        return np.array(Image.open(fbc_path).convert("RGB"))
    except Exception:
        return np.zeros((256, 256, 3), dtype=np.uint8)


def _load_fbc_png_gray(fbc_path: Path) -> np.ndarray:
    """
    Load watershed PNG (grayscale) replicated to [H,W,3].
    Used when fbc_ch=1.
    """
    try:
        if not fbc_path.exists():
            return np.zeros((256, 256, 3), dtype=np.uint8)
        arr = np.array(Image.open(fbc_path).convert("L"))
        return np.stack([arr, arr, arr], axis=-1)
    except Exception:
        return np.zeros((256, 256, 3), dtype=np.uint8)


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """uint8 [H,W,3] → float32 [H,W,3] normalized."""
    img = img.astype(np.float32) / 255.0
    return (img - NORM_MEAN) / NORM_STD


def _fbc_to_binary_channels(fbc: np.ndarray) -> np.ndarray:
    """
    Decode FBC PNG into binary channels independently.
    Returns float32 [H,W,3]: footprint, boundary, contact.
    Each channel thresholded at >127 independently.
    """
    footprint = (fbc[:, :, 0] > 127).astype(np.float32)
    boundary  = (fbc[:, :, 1] > 127).astype(np.float32)
    contact   = (fbc[:, :, 2] > 127).astype(np.float32)
    return np.stack([footprint, boundary, contact], axis=-1)


# ══════════════════════════════════════════════════════════════
# REGISTRY BUILDER
# ══════════════════════════════════════════════════════════════

def _registry_cache_dir() -> Path:
    for candidate in [
        Path("/kaggle/working"),
        PATCHES_DIR.parent if PATCHES_DIR != Path(".") else None,
        Path("/tmp"),
    ]:
        if candidate is not None and candidate.exists():
            return candidate
    return Path(".")


def _registry_cache_path(cities, n_months, min_month_gap,
                         sliding_windows, fbc_ch) -> Path:
    import hashlib
    key = f"{sorted(cities)}-{n_months}-{min_month_gap}-{sliding_windows}-fbc{fbc_ch}"
    h   = hashlib.md5(key.encode()).hexdigest()[:10]
    return _registry_cache_dir() / f"registry_cache_{h}.pkl"


def build_temporal_registry(
    cities,
    n_months:        int  = N_MONTHS,
    min_month_gap:   int  = MIN_MONTH_GAP,
    sliding_windows: bool = True,
    fbc_ch:          int  = 3,
) -> List[Dict]:
    """
    Build registry with sliding windows.
    fbc_ch=3: validates FBC_DIR colored PNGs exist.
    fbc_ch=1: validates WATERSHED_DIR grayscale PNGs exist (backwards compat).
    Cache key includes fbc_ch so the two don't collide.
    """
    import pickle

    cache_path = _registry_cache_path(cities, n_months, min_month_gap,
                                      sliding_windows, fbc_ch)
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                registry = pickle.load(f)
            n_chg = sum(1 for e in registry if e["has_change"])
            n_noc = len(registry) - n_chg
            logger.info(f"Registry loaded from cache: {cache_path}")
            logger.info(f"Registry: {len(registry):,} windows "
                        f"(changed={n_chg:,}, unchanged={n_noc:,})")
            return registry
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), rebuilding...")

    # Choose mask directory based on fbc_ch
    mask_dir  = FBC_DIR if fbc_ch == 3 else WATERSHED_DIR
    mask_suffix = "_fbc.png" if fbc_ch == 3 else "_watershed.png"
    logger.info(f"Building registry (fbc_ch={fbc_ch}, mask_dir={mask_dir}) ...")

    registry = []
    n_with_change = n_without_change = 0

    for city in cities:
        city_patch_dir = PATCHES_DIR / city
        city_mask_dir  = mask_dir    / city

        if not city_patch_dir.exists() or not city_mask_dir.exists():
            continue

        position_months: Dict[Tuple, Dict] = defaultdict(dict)
        for tif_path in city_patch_dir.glob("*.tif"):
            parsed = _parse_patch_stem(tif_path.stem)
            if parsed is None: continue
            pos = (parsed["row"], parsed["col"])
            position_months[pos][parsed["month_idx"]] = tif_path

        for (row, col), month_dict in position_months.items():
            available = sorted(month_dict.keys())
            if len(available) < n_months: continue
            if available[-1] - available[0] < min_month_gap: continue

            # Verify masks exist
            all_masks = {}
            valid = True
            for m_idx in available:
                patch_path = month_dict[m_idx]
                mask_path  = city_mask_dir / f"{patch_path.stem}{mask_suffix}"
                if not mask_path.exists():
                    valid = False; break
                all_masks[m_idx] = mask_path
            if not valid: continue

            # Create windows
            if sliding_windows:
                windows_to_add = [available[i:i+n_months]
                                  for i in range(len(available) - n_months + 1)]
            else:
                step   = (len(available) - 1) / (n_months - 1)
                idxs   = sorted(set(int(round(i * step)) for i in range(n_months)))
                while len(idxs) < n_months: idxs.append(idxs[-1])
                windows_to_add = [[available[i] for i in idxs[:n_months]]]

            for window_months in windows_to_add:
                # Change label uses FOOTPRINT only (ch 0)
                w_first = window_months[0]
                w_last  = window_months[-1]

                if fbc_ch == 3:
                    first_fp = _fbc_to_binary_channels(
                        _load_fbc_png_color(all_masks[w_first]))[:, :, 0]
                    last_fp  = _fbc_to_binary_channels(
                        _load_fbc_png_color(all_masks[w_last]))[:, :, 0]
                else:
                    first_fp = (_load_fbc_png_gray(all_masks[w_first])[:,:,0] > 127
                                ).astype(np.float32)
                    last_fp  = (_load_fbc_png_gray(all_masks[w_last])[:,:,0] > 127
                                ).astype(np.float32)

                w_change_pixels = int((first_fp != last_fp).sum())
                w_has_change    = w_change_pixels > 0

                if w_has_change: n_with_change += 1
                else:            n_without_change += 1

                registry.append({
                    "city":          city,
                    "row":           row,
                    "col":           col,
                    "window_months": window_months,
                    "all_patches":   month_dict,
                    "all_masks":     all_masks,
                    "first_month":   available[0],
                    "last_month":    available[-1],
                    "month_gap":     available[-1] - available[0],
                    "has_change":    w_has_change,
                    "change_pixels": w_change_pixels,
                    "fbc_ch":        fbc_ch,
                })

    logger.info(f"Registry built: {len(registry):,} windows "
                f"(n_months={n_months}, min_gap={min_month_gap}, fbc_ch={fbc_ch})")
    logger.info(f"  With change:    {n_with_change:,}")
    logger.info(f"  Without change: {n_without_change:,}")
    print(f"USING FBC_CH={fbc_ch}")
    print(f"CACHE FILE={cache_path}")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(registry, f)
        logger.info(f"Registry cached to {cache_path}")
    except Exception as e:
        logger.warning(f"Could not save registry cache: {e}")

    return registry


# ══════════════════════════════════════════════════════════════
# DATASET CLASS
# ══════════════════════════════════════════════════════════════

class SpaceNet7TemporalDataset(Dataset):
    """
    Multi-temporal dataset for TE-UNet v2.

    fbc_ch=3 returns:
        images       : float32 [N_MONTHS*3, H, W]    normalized RGB stack
        masks        : float32 [N_MONTHS*3, H, W]    FBC channels per month
                       layout: [fp_m0, bd_m0, ct_m0, fp_m1, bd_m1, ct_m1, ...]
        change_label : float32 [1, H, W]             footprint change only
        meta         : dict

    fbc_ch=1 returns:
        masks: float32 [N_MONTHS, H, W]  footprint only (backwards compat)

    VISUALIZATION:
        Footprint channel per month = masks[:, ::fbc_ch]  (ch 0,3,6,9,12)
        white=building, black=background — identical to before.
    """

    def __init__(
        self,
        cities,
        split:             str           = "train",
        augment:           bool          = True,
        n_months:          int           = N_MONTHS,
        min_month_gap:     int           = MIN_MONTH_GAP,
        samples_per_epoch: Optional[int] = SAMPLES_PER_EPOCH,
        change_oversample: float         = CHANGE_OVERSAMPLE,
        fbc_ch:            int           = 3,
    ):
        self.split             = split
        self.augment           = augment and (split == "train")
        self.n_months          = n_months
        self.samples_per_epoch = samples_per_epoch if self.augment else None
        self.change_oversample = change_oversample
        self.fbc_ch            = fbc_ch

        use_sliding = (split == "train")
        logger.info(f"Building {split} registry for {len(cities)} cities ...")

        self.full_registry = build_temporal_registry(
            cities, n_months=n_months, min_month_gap=min_month_gap,
            sliding_windows=use_sliding, fbc_ch=fbc_ch,
        )

        self.changed_registry   = [e for e in self.full_registry if e["has_change"]]
        self.unchanged_registry = [e for e in self.full_registry if not e["has_change"]]
        self.registry           = self.full_registry.copy()

        if not self.full_registry:
            logger.warning(f"Empty registry for split='{split}'!")

        self.geo_transform   = (_build_geometric_transform() if self.augment
                                else _build_val_transform())
        self.photo_transform = (_build_photometric_transform() if self.augment
                                else None)

        status = (f"subsampled to {self.samples_per_epoch:,}/epoch"
                  if self.samples_per_epoch and len(self.full_registry) > self.samples_per_epoch
                  else f"full {len(self.full_registry):,} windows")
        logger.info(f"  {split:5s}: {len(self.full_registry):,} total | {status} | "
                    f"changed={len(self.changed_registry):,} | "
                    f"unchanged={len(self.unchanged_registry):,}")

    def __len__(self): return len(self.registry)

    def shuffle(self, seed=None):
        rng = random.Random(seed)
        if not self.samples_per_epoch:
            self.registry = self.full_registry.copy()
            rng.shuffle(self.registry)
            return

        n_total     = self.samples_per_epoch
        n_changed   = int(n_total * self.change_oversample)
        n_unchanged = n_total - n_changed

        if len(self.changed_registry) >= n_changed:
            changed_sample = rng.sample(self.changed_registry, n_changed)
        else:
            changed_sample = self.changed_registry.copy()
            while len(changed_sample) < n_changed:
                changed_sample.extend(rng.sample(
                    self.changed_registry,
                    min(len(self.changed_registry), n_changed - len(changed_sample))
                ))
            changed_sample = changed_sample[:n_changed]

        unchanged_sample = (rng.sample(self.unchanged_registry, n_unchanged)
                            if len(self.unchanged_registry) >= n_unchanged
                            else self.unchanged_registry.copy())

        self.registry = changed_sample + unchanged_sample
        rng.shuffle(self.registry)

    def _apply_photometric(self, img):
        if self.photo_transform is None: return img
        return self.photo_transform(image=img)["image"]

    def __getitem__(self, idx):
        entry = self.registry[idx]

        # Load images and masks
        imgs = []
        fbcs = []
        for m_idx in entry["window_months"]:
            imgs.append(_load_tif_rgb(entry["all_patches"][m_idx]))
            if self.fbc_ch == 3:
                fbcs.append(_load_fbc_png_color(entry["all_masks"][m_idx]))
            else:
                fbcs.append(_load_fbc_png_gray(entry["all_masks"][m_idx]))

        # Geometric augmentation — same transform for all months + masks
        geo_out = self.geo_transform(
            image    = imgs[0], image_m1=imgs[1], image_m2=imgs[2],
            image_m3 = imgs[3], image_m4=imgs[4],
            mask_m0  = fbcs[0], mask_m1 =fbcs[1], mask_m2 =fbcs[2],
            mask_m3  = fbcs[3], mask_m4 =fbcs[4],
        )
        aug_imgs = [geo_out["image"],    geo_out["image_m1"], geo_out["image_m2"],
                    geo_out["image_m3"], geo_out["image_m4"]]
        aug_fbcs = [geo_out["mask_m0"],  geo_out["mask_m1"],  geo_out["mask_m2"],
                    geo_out["mask_m3"],  geo_out["mask_m4"]]

        # Photometric augmentation — independent per month (images only)
        if self.augment:
            aug_imgs = [self._apply_photometric(im) for im in aug_imgs]

        # Normalize images
        norm_imgs = [_normalize_image(im) for im in aug_imgs]

        # Decode FBC masks
        fbc_binaries = [_fbc_to_binary_channels(f) for f in aug_fbcs]

        # Stack images: [3,H,W] × 5 → [15,H,W]
        chw_imgs    = [im.transpose(2, 0, 1) for im in norm_imgs]
        stacked_imgs = np.concatenate(chw_imgs, axis=0)  # [15,H,W]

        # Stack FBC masks
        if self.fbc_ch == 3:
            # All 3 channels per month → [3,H,W] × 5 → [15,H,W]
            # Layout: [fp_m0, bd_m0, ct_m0, fp_m1, bd_m1, ct_m1, ...]
            chw_fbcs     = [f.transpose(2, 0, 1) for f in fbc_binaries]
            stacked_fbcs = np.concatenate(chw_fbcs, axis=0)  # [15,H,W]
        else:
            # Footprint only → [1,H,W] × 5 → [5,H,W]
            chw_fbcs     = [f[np.newaxis, :, :, 0] for f in fbc_binaries]
            stacked_fbcs = np.concatenate(chw_fbcs, axis=0)  # [5,H,W]

        # Change label — footprint only regardless of fbc_ch
        first_fp     = fbc_binaries[0][:, :, 0]   # [H,W]
        last_fp      = fbc_binaries[-1][:, :, 0]  # [H,W]
        change_label = (first_fp != last_fp).astype(np.float32)

        meta = {
            "city":          entry["city"],
            "window_months": entry["window_months"],
            "first_month":   entry["first_month"],
            "last_month":    entry["last_month"],
            "month_gap":     entry["month_gap"],
            "has_change":    entry["has_change"],
            "change_pixels": entry["change_pixels"],
            "row":           entry["row"],
            "col":           entry["col"],
        }

        return (
            torch.from_numpy(stacked_imgs).float(),               # [15,H,W]
            torch.from_numpy(stacked_fbcs).float(),               # [15 or 5,H,W]
            torch.from_numpy(change_label).float().unsqueeze(0),  # [1,H,W]
            meta,
        )


# ══════════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════

def get_dataloaders(
    batch_size:        int           = 8,
    num_workers:       int           = 4,
    pin_memory:        bool          = True,
    n_months:          int           = N_MONTHS,
    samples_per_epoch: Optional[int] = SAMPLES_PER_EPOCH,
    fbc_ch:            int           = 3,
) -> Dict:
    worker_kw = dict(num_workers=num_workers, pin_memory=pin_memory,
                     persistent_workers=False,
                     prefetch_factor=2 if num_workers > 0 else None)

    train_ds = SpaceNet7TemporalDataset(
        TRAIN_CITIES, split="train", augment=True, n_months=n_months,
        samples_per_epoch=samples_per_epoch, fbc_ch=fbc_ch,
    )
    val_ds = SpaceNet7TemporalDataset(
        VAL_CITIES, split="val", augment=False, n_months=n_months,
        samples_per_epoch=None, fbc_ch=fbc_ch,
    )
    test_ds = SpaceNet7TemporalDataset(
        TEST_CITIES, split="test", augment=False, n_months=n_months,
        samples_per_epoch=None, fbc_ch=fbc_ch,
    )

    train_ds.shuffle(seed=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              drop_last=True, **worker_kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, **worker_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              drop_last=False, **worker_kw)

    logger.info("DataLoaders ready:")
    logger.info(f"  Train: {len(train_ds):,} windows [{len(train_loader):,} batches]")
    logger.info(f"  Val:   {len(val_ds):,} windows   [{len(val_loader):,} batches]")
    logger.info(f"  Test:  {len(test_ds):,} windows  [{len(test_loader):,} batches]")

    return {"train_ds": train_ds, "train": train_loader,
            "val": val_loader, "test": test_loader}


# ══════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config import cfg
    print("=" * 60)
    print(f"DATASET SELF-TEST  fbc_ch={cfg.fbc_ch}")
    print("=" * 60)

    ds = SpaceNet7TemporalDataset(
        TRAIN_CITIES[:2], split="train", augment=False,
        samples_per_epoch=None, fbc_ch=cfg.fbc_ch,
    )
    print(f"Registry: {len(ds.full_registry):,} windows")

    images, masks, change_label, meta = ds[0]
    print(f"images:       {tuple(images.shape)}")   # [15,256,256]
    print(f"masks:        {tuple(masks.shape)}")    # [15,256,256] if fbc_ch=3
    print(f"change_label: {tuple(change_label.shape)}")  # [1,256,256]

    # Footprint channel for visualization: every fbc_ch-th starting at 0
    FOOTPRINT_IDX = list(range(0, cfg.fbc_ch * N_MONTHS, cfg.fbc_ch))
    footprint = masks[FOOTPRINT_IDX]
    print(f"footprint (ch0 per month): {tuple(footprint.shape)}")  # [5,256,256]

    if cfg.fbc_ch == 3:
        assert masks.shape == (15, 256, 256), f"Expected [15,256,256] got {masks.shape}"
        bd = masks[1::3]  # boundary channels
        ct = masks[2::3]  # contact channels
        print(f"boundary pixels month0: {(bd[0]>0.5).sum().item()}")
        print(f"contact  pixels month0: {(ct[0]>0.5).sum().item()}")
    else:
        assert masks.shape == (5, 256, 256), f"Expected [5,256,256] got {masks.shape}"

    assert images.shape       == (15, 256, 256)
    assert change_label.shape == (1,  256, 256)
    print("\n✅ ALL TESTS PASSED")
    print("=" * 60)