"""
generate_flow_accumulation.py — D8 flow-accumulation raster from the DEM.

Standard hydrology DEM-conditioning pipeline (pysheds): fill single-cell
pits, fill depressions (breach-free priority-flood), resolve flats, derive
D8 flow direction, then flow accumulation (number of upstream cells draining
through each cell). Ocean cells (land_mask == 0) are included in the DEM
conditioning as a flat 0 m base level — this is the correct outlet/sink for
on-land drainage — but are masked to nodata in the saved raster since
accumulation values over open water aren't meaningful.

Output matches the other parms_bands rasters: same shape/CRS/transform,
float32 GeoTIFF, nodata = -9999.0 (same convention as slope.tif).

Usage
-----
python generate_flow_accumulation.py
python generate_flow_accumulation.py \\
    --dem_path /home/hl1138/surrogate/data/parms_bands/dem.tif \\
    --land_mask_path /home/hl1138/surrogate/data/parms_bands/land_mask.tif \\
    --out_path /home/hl1138/surrogate/data/parms_bands/flow_accum.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from pysheds.grid import Grid

# Standard ESRI D8 direction encoding, as used throughout pysheds examples.
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
NODATA_OUT = -9999.0


def generate_flow_accumulation(
    dem_path:        Path,
    land_mask_path:  Path,
    out_path:        Path,
) -> np.ndarray:
    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))

    print(f'DEM size : {dem.shape}')
    print('Conditioning DEM (fill pits → fill depressions → resolve flats)...')
    pit_filled   = grid.fill_pits(dem)
    flooded      = grid.fill_depressions(pit_filled)
    inflated     = grid.resolve_flats(flooded)

    print('Computing D8 flow direction...')
    fdir = grid.flowdir(inflated, dirmap=DIRMAP)

    print('Computing flow accumulation...')
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    acc = np.asarray(acc, dtype=np.float32)

    with rasterio.open(land_mask_path) as src:
        land = src.read(1)
    acc_masked = np.where(land == 0, NODATA_OUT, acc).astype(np.float32)

    land_acc = acc[land == 1]
    print(f'Land-cell accumulation: min={land_acc.min():.0f}  '
          f'max={land_acc.max():.0f}  mean={land_acc.mean():.1f}  '
          f'median={np.median(land_acc):.0f}')

    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
    profile.update(dtype='float32', nodata=NODATA_OUT, count=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(acc_masked, 1)
        dst.update_tags(1, description='D8 flow accumulation (upstream cell count)')

    print(f'Saved → {out_path}')
    return acc_masked


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dem_path',       default='/home/hl1138/surrogate/data/parms_bands/dem.tif')
    p.add_argument('--land_mask_path', default='/home/hl1138/surrogate/data/parms_bands/land_mask.tif')
    p.add_argument('--out_path',       default='/home/hl1138/surrogate/data/parms_bands/flow_accum.tif')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_flow_accumulation(
        dem_path       = Path(args.dem_path),
        land_mask_path = Path(args.land_mask_path),
        out_path       = Path(args.out_path),
    )
