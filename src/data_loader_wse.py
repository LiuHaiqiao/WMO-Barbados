"""
data_loader_wse.py — Dataset and DataLoader for WSE-based flood surrogates.

Identical layout and splitting to data_loader.py, except the dynamic water
state is the water-surface elevation (WSE = DEM + water depth) instead of
depth alone. WSE is built directly at load time and normalised by NORM_DEM —
the same scale as the DEM input channel:

    wse = (dem_m + depth_m) / NORM_DEM

Each __getitem__ returns a 4-tuple for N-step rollout training:
    x           (7, P, P)       — [DEM, Manning, Pervious, Slope, Rain_{t-1}, Rain_t, WSE_t]
    rain_future (N-1, 1, P, P)  — Rain_{t+1} … Rain_{t+N-1}  (inputs for steps 2…N)
    wse_future  (N,   1, P, P)  — WSE_{t+1} … WSE_{t+N}      (step targets)
    land        (1, P, P)       — land mask (1=land, 0=ocean)

Water depth in metres is recovered by de-normalising the WSE and reducing
the DEM from it:
    depth_m = wse * NORM_DEM - dem_m = (wse - x[:, 0:1]) * NORM_DEM
"""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from data_loader import (
    NORM_DEM, NORM_DEPTH,
    _DEFAULT_STATIC_DIR, _DEFAULT_SAMPLES_DIR,
    _load_static, _discover_samples, FloodDataset,
)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class FloodDatasetWSE(FloodDataset):
    """FloodDataset variant whose dynamic state channel and step targets are
    WSE = DEM + depth (normalised by NORM_DEM) instead of depth.

    The DEM needed to build WSE is already channel 0 of x, so construction
    parameters are identical to FloodDataset's.
    """

    def __getitem__(self, idx: int):
        x, rain_future, depth_future, land = super().__getitem__(idx)
        dem = x[0:1]                                   # normalised DEM channel

        # normalised depth (mm / NORM_DEPTH) → metres, then WSE = (DEM_m + depth_m) / NORM_DEM
        to_m = NORM_DEPTH / 1000.0
        x[6:7]     = dem + x[6:7]        * to_m / NORM_DEM   # Depth_t → WSE_t
        wse_future = dem + depth_future  * to_m / NORM_DEM   # (N, 1, P, P)
        return x, rain_future, wse_future, land


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
    Return ``(train_loader, val_loader, test_loader)`` of WSE samples.

    Splitting is identical to data_loader.build_loaders: samples are shuffled
    then split **by simulation** to prevent data leakage.
    """
    samples_dir = Path(samples_dir)
    static_dir  = Path(static_dir)

    static, land_mask = _load_static(static_dir)

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
        ds = FloodDatasetWSE(dirs, static, land_mask,
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

    x, rain_future, wse_future, land = next(iter(train_loader))
    print(f'x           : {tuple(x.shape)}')
    print(f'rain_future : {tuple(rain_future.shape)}')
    print(f'wse_future  : {tuple(wse_future.shape)}')
    print(f'land        : {tuple(land.shape)}  land_frac={land.float().mean():.3f}')
    print(f'WSE range   : {x[:, 6].min():.4f} … {x[:, 6].max():.4f} '
          f'(DEM channel: {x[:, 0].min():.4f} … {x[:, 0].max():.4f})')

    # depth recovered from WSE must be non-negative
    depth_m = (wse_future - x[:, 0:1].unsqueeze(1)) * NORM_DEM
    print(f'recovered depth range: {depth_m.min():.4f} … {depth_m.max():.4f} m (expect >= 0)')
    print(f'Train batches : {len(train_loader)}  ({len(train_loader.dataset)} samples)')
    print(f'Val   batches : {len(val_loader)}  ({len(val_loader.dataset)} samples)')
    print(f'Test  batches : {len(test_loader)}  ({len(test_loader.dataset)} samples)')
