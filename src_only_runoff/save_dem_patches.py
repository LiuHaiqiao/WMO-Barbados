"""
save_dem_patches.py — Extract and save DEM patches using the same tiling
strategy as data_loader.py (patch_size=512, stride=568 → 4 patches).

Each patch is saved as a GeoTIFF with the correct affine transform so spatial
coordinates are preserved.

Usage
-----
python save_dem_patches.py
python save_dem_patches.py --dem_path /home/hl1138/TFNO/data/parms_bands/dem.tif \
                           --out_dir  /home/hl1138/TFNO/data/parms_bands/dem_patches \
                           --patch_size 512 --stride 568
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import _patch_offsets


def save_dem_patches(
    dem_path:   Path,
    out_dir:    Path,
    patch_size: int = 512,
    stride:     int = 568,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem      = src.read(1).astype(np.float32)
        profile  = src.profile.copy()
        transform = src.transform          # original affine
        H, W     = src.height, src.width

    offsets = _patch_offsets(H, W, patch_size, stride)
    print(f'DEM size    : {H} × {W}')
    print(f'Patch size  : {patch_size} × {patch_size}  stride={stride}')
    print(f'Patches     : {len(offsets)}')

    profile.update(
        height = patch_size,
        width  = patch_size,
        count  = 1,
        dtype  = 'float32',
    )

    for i, (r, c) in enumerate(offsets):
        patch = dem[r : r + patch_size, c : c + patch_size]

        # Shift origin: col_offset → x, row_offset → y
        # Affine(scale_x, shear_x, origin_x, shear_y, scale_y, origin_y)
        x0 = transform.c + c * transform.a   # new west edge
        y0 = transform.f + r * transform.e   # new north edge
        patch_transform = Affine(transform.a, transform.b, x0,
                                 transform.d, transform.e, y0)

        profile.update(transform=patch_transform)
        out_path = out_dir / f'dem_patch_{i:02d}_r{r:04d}_c{c:04d}.tif'

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(patch, 1)

        print(f'  [{i:02d}] row={r:4d} col={c:4d}  '
              f'range=[{patch.min():.2f}, {patch.max():.2f}] m  → {out_path.name}')

    print(f'\nSaved {len(offsets)} patches → {out_dir}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dem_path',   default='/home/hl1138/TFNO/data/parms_bands/dem.tif')
    p.add_argument('--out_dir',    default='/home/hl1138/TFNO/data/parms_bands/dem_patches')
    p.add_argument('--patch_size', type=int, default=512)
    p.add_argument('--stride',     type=int, default=568)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_dem_patches(
        dem_path   = Path(args.dem_path),
        out_dir    = Path(args.out_dir),
        patch_size = args.patch_size,
        stride     = args.stride,
    )
