"""
data_loader_extra.py — Dataset and DataLoader for the flood surrogate with
two additional static features: flow accumulation and soil type.

Identical to data_loader.py in every respect (patch tiling, simulation-level
splitting, dynamic channels, n-step rollout targets) except the static
feature stack grows from 4 to 6 channels:

    [DEM, Manning, Pervious, Slope, FlowAccLog, SoilType]

FloodDataset.__getitem__ (imported unchanged from data_loader.py) builds
x by concatenating whatever static tensor it's given with the 3 dynamic
channels, so no Dataset subclass is needed — only a static-loading function
that reads the two extra rasters alongside the original four.

Each __getitem__ returns a 4-tuple for N-step rollout training:
    x            (9, P, P)       — [DEM, Manning, Pervious, Slope, FlowAccLog,
                                     SoilType, Rain_{t-1}, Rain_t, Depth_t]
    rain_future  (N-1, 1, P, P)  — Rain_{t+1} … Rain_{t+N-1}
    depth_future (N,   1, P, P)  — Depth_{t+1} … Depth_{t+N}
    land         (1, P, P)       — land mask (1=land, 0=ocean)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import (
    NORM_DEM, NORM_MAN, NORM_SLOPE,
    _DEFAULT_STATIC_DIR, _DEFAULT_SAMPLES_DIR,
    _read, _discover_samples, FloodDataset,
)

# flow_accum.tif is a D8 upstream-cell count, heavily right-skewed (median 3,
# max ~98k on this domain) — log1p-transformed before normalising, same
# margin-above-observed-max convention as NORM_DEM/NORM_SLOPE.
NORM_FLOWACC_LOG = 12.0

# soil_type.tif holds 6 categorical codes (3, 4, 13, 14, 101 = land classes,
# 102 = ocean/water). Treated here as a single ordinal-normalised channel for
# simplicity (2 extra channels total, as requested) rather than one-hot
# encoded — revisit as one-hot per class if this proves too coarse a signal.
NORM_SOIL = 128.0


# --------------------------------------------------------------------------- #
# Static loading
# --------------------------------------------------------------------------- #

def _load_static_extra(static_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load shared static features → (6, H, W) and land_mask → (1, H, W)."""
    dem       = _read(static_dir / 'dem.tif')           / NORM_DEM
    manning   = _read(static_dir / 'manning_coef.tif')  / NORM_MAN
    pervious  = _read(static_dir / 'pervious_cover.tif')
    slope     = _read(static_dir / 'slope.tif')         / NORM_SLOPE
    flow_acc  = np.log1p(_read(static_dir / 'flow_accum.tif')) / NORM_FLOWACC_LOG
    soil_type = _read(static_dir / 'soil_type.tif')     / NORM_SOIL

    static = torch.from_numpy(np.stack(
        [dem, manning, pervious, slope, flow_acc, soil_type], axis=0))
    mask   = torch.from_numpy(_read(static_dir / 'land_mask.tif')[None])
    return static, mask


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #

def build_loaders(
    samples_dir:  str | Path = _DEFAULT_SAMPLES_DIR,
    static_dir:   str | Path = _DEFAULT_STATIC_DIR,
    patch_size:   int   = 512,
    stride:       int   = 568,
    val_split:    float = 0.15,
    test_split:   float = 0.15,
    batch_size:   int   = 4,
    num_workers:  int   = 4,
    seed:         int   = 42,
    n_steps:      int   = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return ``(train_loader, val_loader, test_loader)``.

    Splitting is identical to data_loader.build_loaders: samples are shuffled
    then split **by simulation** to prevent data leakage.
    """
    samples_dir = Path(samples_dir)
    static_dir  = Path(static_dir)

    static, land_mask = _load_static_extra(static_dir)

    all_dirs = _discover_samples(samples_dir)
    rng      = np.random.default_rng(seed)
    rng.shuffle(all_dirs)

    n       = len(all_dirs)
    n_test  = max(1, int(n * test_split))
    n_val   = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    train_dirs = all_dirs[:n_train]
    val_dirs   = all_dirs[n_train : n_train + n_val]
    test_dirs  = all_dirs[n_train + n_val:]

    print(f'Samples — train: {len(train_dirs)}  val: {len(val_dirs)}  '
          f'test: {len(test_dirs)}  (total: {n})')

    def _make_loader(dirs, shuffle):
        ds = FloodDataset(dirs, static, land_mask,
                          patch_size=patch_size, stride=stride, n_steps=n_steps)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        _make_loader(train_dirs, shuffle=True),
        _make_loader(val_dirs,   shuffle=False),
        _make_loader(test_dirs,  shuffle=False),
    )


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    train_loader, val_loader, test_loader = build_loaders(
        batch_size  = 2,
        num_workers = 0,
        patch_size  = 512,
        stride      = 568,
    )

    x, rain_future, depth_future, land = next(iter(train_loader))
    print(f'x            : {tuple(x.shape)}')
    print(f'rain_future  : {tuple(rain_future.shape)}')
    print(f'depth_future : {tuple(depth_future.shape)}')
    print(f'land         : {tuple(land.shape)}  land_frac={land.float().mean():.3f}')
    print(f'FlowAccLog   : {x[:, 4].min():.4f} … {x[:, 4].max():.4f}')
    print(f'SoilType     : {x[:, 5].min():.4f} … {x[:, 5].max():.4f}')
    print(f'Train batches : {len(train_loader)}  ({len(train_loader.dataset)} samples)')
    print(f'Val   batches : {len(val_loader)}  ({len(val_loader.dataset)} samples)')
    print(f'Test  batches : {len(test_loader)}  ({len(test_loader.dataset)} samples)')
