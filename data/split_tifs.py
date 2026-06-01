"""
split_tifs.py — Split multi-band TIFs into per-timestep single-band files.

Outputs
-------
depth_timesteps/   depth_hr0000.00.tif  …  depth_hr0072.00.tif
pcpout_timesteps/  pcpout_hr0000.00.tif …  pcpout_hr0072.00.tif
u_timesteps/       u_band0001.tif       …  u_band0289.tif
v_timesteps/       v_band0001.tif       …  v_band0289.tif
parms_bands/       soil_type.tif, pervious_cover.tif, dem.tif, manning_coef.tif
"""

import rasterio
from rasterio.crs import CRS
from pathlib import Path

BASE    = Path(__file__).parent
CRS_GEO = CRS.from_epsg(4326)


def split(src_path, out_dir, name_fn):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, crs=CRS_GEO)
        n = src.count
        for i in range(1, n + 1):
            band = src.read(i)
            desc = src.descriptions[i - 1] or f'band {i}'
            fname = out_dir / name_fn(i, desc)
            with rasterio.open(fname, 'w', **profile) as dst:
                dst.write(band, 1)
                dst.update_tags(1, description=desc)
            if i % 50 == 0 or i == n:
                print(f'  [{Path(src_path).name}] {i}/{n}  ->  {fname.name}')


def hr_name(prefix):
    def fn(i, desc):
        hr_str = desc.split('HR')[-1].strip().split()[0]
        return f'{prefix}_hr{float(hr_str):07.2f}.tif'
    return fn


if __name__ == '__main__':
    split(BASE / 'depth.tif',         BASE / 'depth_timesteps',   hr_name('depth'))
    split(BASE / 'pcpout.tif',        BASE / 'pcpout_timesteps',  hr_name('pcpout'))
    split(BASE / 'u.tif',             BASE / 'u_timesteps',       lambda i, d: f'u_band{i:04d}.tif')
    split(BASE / 'v.tif',             BASE / 'v_timesteps',       lambda i, d: f'v_band{i:04d}.tif')
    split(BASE / 'parms.sample1.tif', BASE / 'parms_bands',       lambda i, d: f'{d.lower().replace(" ", "_")}.tif')
    print('Done.')
