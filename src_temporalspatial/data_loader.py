"""
data_loader.py — Dataset and DataLoader for FloodDCRNN.

Each raster patch is flattened into a graph of N = P×P nodes connected by a
4-connectivity undirected grid (shared across all patches).  Edge weights are
slope-based by default: w = exp(-|Δelev|/scale), so downhill neighbours get
higher weight.

Per item  (P = patch_size, N = P×P, T = n_steps):
    init_feat  : (N, D1=5)   depth_t + [DEM, Manning, Pervious, Slope]
    dyn_feat   : (T, N, D2=5) rain_{t+k} + [DEM, Manning, Pervious, Slope],  k=0…T-1
    target     : (T, N, 1)   depth_{t+k+1},  k=0…T-1
    land       : (N,)        land mask  (1 = land, 0 = ocean)

Shared graph (returned by build_loaders, NOT inside the batch):
    edge_index : (2, E)      4-connected grid topology  (same for every patch)
    edge_weight: (E,)        uniform or slope-based weights

D1 = D2 = 5.  Adjust in_channels of FloodDCRNN accordingly.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Normalisation constants  (same as src/data_loader.py)
# --------------------------------------------------------------------------- #
NORM_DEM   = 400.0
NORM_MAN   = 0.16
NORM_SLOPE = 40.0
NORM_RAIN  = 100.0
NORM_DEPTH = 30000.0

_DEFAULT_STATIC_DIR  = Path('/home/hl1138/TFNO/data/parms_bands')
_DEFAULT_SAMPLES_DIR = Path('/home/hl1138/TFNO/data/samples')

# Feature dimensions exposed for use in model construction
D_INIT = 5   # [depth_t, DEM, Manning, Pervious, Slope]
D_DYN  = 5   # [rain_t,  DEM, Manning, Pervious, Slope]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = 0.0
        return data


def _patch_offsets(H: int, W: int, patch: int, stride: int) -> list[tuple[int, int]]:
    def positions(dim: int) -> list[int]:
        pos = list(range(0, dim - patch, stride))
        pos.append(dim - patch)
        return sorted(set(pos))
    return [(r, c) for r in positions(H) for c in positions(W)]


def _load_static(static_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return static (4, H, W) float32 and land_mask (H, W) float32."""
    dem      = _read(static_dir / 'dem.tif')           / NORM_DEM
    manning  = _read(static_dir / 'manning_coef.tif')  / NORM_MAN
    pervious = _read(static_dir / 'pervious_cover.tif')
    slope    = _read(static_dir / 'slope.tif')         / NORM_SLOPE
    static   = np.stack([dem, manning, pervious, slope], axis=0)  # (4, H, W)
    land     = _read(static_dir / 'land_mask.tif')                # (H, W)
    return static, land


def _discover_samples(samples_dir: Path) -> list[Path]:
    dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir()],
        key=lambda d: (len(d.name), d.name),
    )
    ready = [
        d for d in dirs
        if (d / 'depth_timesteps').exists()
        and len(list((d / 'depth_timesteps').glob('depth_hr????.00.tif'))) > 0
        and (d / 'pcpout_timesteps').exists()
        and len(list((d / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))) > 0
    ]
    skipped = len(dirs) - len(ready)
    if skipped:
        print(f'[data_loader] {len(ready)}/{len(dirs)} samples ready '
              f'({skipped} skipped — not yet split)')
    return ready


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_grid_graph(
    patch_size:  int,
    dem_patch:   np.ndarray | None = None,
    slope_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a 4-connectivity undirected graph for a patch_size×patch_size grid.

    Parameters
    ----------
    patch_size  : grid side length P;  N = P×P nodes, indexed row-major.
    dem_patch   : (P, P) float32 normalised DEM values.  When provided,
                  edge weights are slope-based:
                      w = exp(-|dem_src - dem_dst| / slope_scale)
                  so neighbours at similar elevation get weight ≈ 1.
                  When None, uniform weights (all 1.0) are used.
    slope_scale : denominator for the exponent  (default 1.0).

    Returns
    -------
    edge_index  : (2, E)  long tensor
    edge_weight : (E,)    float32 tensor
    """
    P   = patch_size
    idx = torch.arange(P * P, dtype=torch.long).reshape(P, P)

    # Horizontal edges: (i, j) ↔ (i, j+1)
    h_src = idx[:, :-1].reshape(-1)
    h_dst = idx[:, 1: ].reshape(-1)

    # Vertical edges: (i, j) ↔ (i+1, j)
    v_src = idx[:-1, :].reshape(-1)
    v_dst = idx[1:,  :].reshape(-1)

    # Bidirectional: both directions for each undirected edge
    src = torch.cat([h_src, h_dst, v_src, v_dst])
    dst = torch.cat([h_dst, h_src, v_dst, v_src])
    edge_index = torch.stack([src, dst])             # (2, E)

    if dem_patch is not None:
        dem_flat    = torch.from_numpy(dem_patch.ravel())
        edge_weight = torch.exp(
            -torch.abs(dem_flat[src] - dem_flat[dst]) / slope_scale
        )
    else:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    return edge_index, edge_weight


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class FloodGraphDataset(Dataset):
    """
    Parameters
    ----------
    sample_dirs : list of simulation directories.
    static      : (4, H, W) normalised static features (pre-loaded).
    land_mask   : (H, W) land mask.
    patch_size  : spatial size of each square node grid (P×P nodes).
    stride      : step between patch origins for tiling the domain.
    n_steps     : number of prediction steps T.
    """

    def __init__(
        self,
        sample_dirs: list[Path],
        static:      np.ndarray,     # (4, H, W)
        land_mask:   np.ndarray,     # (H, W)
        patch_size:  int   = 64,
        stride:      int   = 32,
        n_steps:     int   = 6,
    ):
        self.patch_size = patch_size
        self.n_steps    = n_steps
        self.static     = static      # (4, H, W) — kept as numpy for cheap slicing
        self.land_mask  = land_mask   # (H, W)
        self.samples: list[tuple] = []

        H, W    = static.shape[1], static.shape[2]
        offsets = _patch_offsets(H, W, patch_size, stride)

        for d in sample_dirs:
            depth_files = sorted((d / 'depth_timesteps').glob('depth_hr????.00.tif'))
            rain_files  = sorted((d / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))

            if len(depth_files) != len(rain_files):
                print(f'[data_loader] WARNING: skipping {d.name} — '
                      f'count mismatch ({len(depth_files)} depth vs '
                      f'{len(rain_files)} rain)')
                continue

            # need depth[t] … depth[t+T]  and  rain[t] … rain[t+T-1]
            t_max = min(len(depth_files) - n_steps - 1,
                        len(rain_files)  - n_steps)
            for t in range(0, t_max + 1):
                for (r, c) in offsets:
                    self.samples.append((rain_files, depth_files, t, r, c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rain_files, depth_files, t, r, c = self.samples[idx]
        P = self.patch_size
        T = self.n_steps
        N = P * P

        def crop_file(path: Path) -> np.ndarray:
            return _read(path)[r:r+P, c:c+P]

        # Static patch  (4, P, P)
        static_p = self.static[:, r:r+P, c:c+P]   # (4, P, P)
        land_p   = self.land_mask[r:r+P, c:c+P]   # (P, P)

        # ── Initial features ──────────────────────────────────────────────────
        depth_t   = crop_file(depth_files[t]) / NORM_DEPTH   # (P, P)
        # [depth_t, DEM, Manning, Pervious, Slope]  →  (P, P, 5)  →  (N, 5)
        init_np   = np.stack([depth_t, *static_p], axis=-1).reshape(N, D_INIT)

        # ── Dynamic features  (T, P, P, D2)  →  (T, N, D2) ──────────────────
        dyn_list = []
        for k in range(T):
            rain_k = crop_file(rain_files[t + k]) / NORM_RAIN   # (P, P)
            # [rain_k, DEM, Manning, Pervious, Slope]  →  (P, P, 5)
            dyn_k  = np.stack([rain_k, *static_p], axis=-1)
            dyn_list.append(dyn_k)
        dyn_np = np.stack(dyn_list, axis=0).reshape(T, N, D_DYN)  # (T, N, 5)

        # ── Targets  (T, N, 1) ────────────────────────────────────────────────
        tgt_list = []
        for k in range(T):
            tgt_list.append(crop_file(depth_files[t + k + 1]) / NORM_DEPTH)
        tgt_np = np.stack(tgt_list, axis=0).reshape(T, N, 1)   # (T, N, 1)

        return (
            torch.from_numpy(init_np),   # (N, D1)
            torch.from_numpy(dyn_np),    # (T, N, D2)
            torch.from_numpy(tgt_np),    # (T, N, 1)
            torch.from_numpy(land_p.reshape(N).astype(np.float32)),  # (N,)
        )


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #

def build_loaders(
    samples_dir:  str | Path = _DEFAULT_SAMPLES_DIR,
    static_dir:   str | Path = _DEFAULT_STATIC_DIR,
    patch_size:   int   = 64,
    stride:       int   = 32,
    n_steps:      int   = 6,
    val_split:    float = 0.15,
    test_split:   float = 0.15,
    batch_size:   int   = 8,
    num_workers:  int   = 4,
    seed:         int   = 42,
    slope_weights: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Return ``(train_loader, val_loader, test_loader, edge_index, edge_weight)``.

    Samples are split **by simulation** to prevent data leakage.
    Pass ``edge_index`` and ``edge_weight`` directly to ``FloodDCRNN.forward()``.

    Parameters
    ----------
    slope_weights : if True, edge weights are DEM-slope-based (averaged over
                    the full domain DEM patch centred at the grid origin);
                    if False, uniform weights of 1.0 are used.
    """
    samples_dir = Path(samples_dir)
    static_dir  = Path(static_dir)

    static, land_mask = _load_static(static_dir)

    # Build graph (topology is the same for every patch)
    if slope_weights:
        # Use the top-left patch of the full DEM as a representative sample
        P   = patch_size
        dem = static[0, :P, :P]   # (P, P) normalised DEM
        edge_index, edge_weight = build_grid_graph(P, dem_patch=dem)
    else:
        edge_index, edge_weight = build_grid_graph(patch_size)

    print(f'Graph: {patch_size}×{patch_size} grid  '
          f'| nodes={patch_size**2}  edges={edge_index.shape[1]}  '
          f'weights={"slope-based" if slope_weights else "uniform"}')

    # Discover and shuffle simulations
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

    print(f'Simulations — train: {len(train_dirs)}  '
          f'val: {len(val_dirs)}  test: {len(test_dirs)}  (total: {n})')

    def _make_loader(dirs: list[Path], shuffle: bool) -> DataLoader:
        ds = FloodGraphDataset(
            dirs, static, land_mask,
            patch_size=patch_size, stride=stride, n_steps=n_steps,
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
        )

    train_loader = _make_loader(train_dirs, shuffle=True)
    val_loader   = _make_loader(val_dirs,   shuffle=False)
    test_loader  = _make_loader(test_dirs,  shuffle=False)

    return train_loader, val_loader, test_loader, edge_index, edge_weight


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    from model import FloodDCRNN

    train_loader, val_loader, test_loader, edge_index, edge_weight = build_loaders(
        patch_size  = 64,
        stride      = 32,
        n_steps     = 6,
        batch_size  = 4,
        num_workers = 0,
    )

    init_feat, dyn_feat, target, land = next(iter(train_loader))

    print(f'\n── Batch shapes ──────────────────────────────')
    print(f'init_feat  : {tuple(init_feat.shape)}   (B, N, D1={D_INIT})')
    print(f'dyn_feat   : {tuple(dyn_feat.shape)}  (B, T, N, D2={D_DYN})')
    print(f'target     : {tuple(target.shape)}  (B, T, N, 1)')
    print(f'land       : {tuple(land.shape)}   (B, N)')
    print(f'edge_index : {tuple(edge_index.shape)}')
    print(f'edge_weight: {tuple(edge_weight.shape)}')

    print(f'\n── Loader sizes ─────────────────────────────')
    print(f'train batches : {len(train_loader)}  ({len(train_loader.dataset)} samples)')
    print(f'val   batches : {len(val_loader)}  ({len(val_loader.dataset)} samples)')
    print(f'test  batches : {len(test_loader)}  ({len(test_loader.dataset)} samples)')

    # Forward pass through model
    model = FloodDCRNN(d_init=D_INIT, d_dyn=D_DYN, hidden_dim=32, n_layers=2, K=3)
    with torch.no_grad():
        pred = model(init_feat, dyn_feat, edge_index, edge_weight)
    print(f'\n── Model output ─────────────────────────────')
    print(f'pred depth : {tuple(pred.shape)}   (B, T, N, 1)')
    assert pred.shape == target.shape
    print('Shape assertion passed.')
