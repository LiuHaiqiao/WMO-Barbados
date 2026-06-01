"""
split_all_samples.py — Split depth and pcpout TIFs for every sample in
/home/hl1138/TFNO/data/samples/.

Outputs per sample
------------------
  samples/sampleN/depth_timesteps/   depth_hr0000.00.tif … depth_hr0072.00.tif
  samples/sampleN/pcpout_timesteps/  pcpout_hr0000.00.tif … pcpout_hr0072.00.tif

Usage
-----
python split_all_samples.py                   # all samples, 8 workers
python split_all_samples.py --workers 4       # fewer parallel workers
python split_all_samples.py --overwrite       # redo already-split samples
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from tqdm import tqdm

CRS_GEO = CRS.from_epsg(4326)
SAMPLES_DIR = Path(__file__).parent / 'samples'


# --------------------------------------------------------------------------- #
# Core split (same logic as split_tifs.py)
# --------------------------------------------------------------------------- #

def _hr_name(prefix: str, desc: str) -> str:
    hr_str = desc.split('HR')[-1].strip().split()[0]
    return f'{prefix}_hr{float(hr_str):07.2f}.tif'


def _split(src_path: Path, out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, crs=CRS_GEO)
        for i in range(1, src.count + 1):
            desc  = src.descriptions[i - 1] or f'band {i}'
            fname = out_dir / _hr_name(prefix, desc)
            band  = src.read(i)
            with rasterio.open(fname, 'w', **profile) as dst:
                dst.write(band, 1)
                dst.update_tags(1, description=desc)
    return src.count


def _already_done(sample_dir: Path, prefix: str, expected: int = 289) -> bool:
    out_dir = sample_dir / f'{prefix}_timesteps'
    return out_dir.exists() and len(list(out_dir.glob(f'{prefix}_hr*.tif'))) >= expected


# --------------------------------------------------------------------------- #
# Per-sample worker (runs in subprocess)
# --------------------------------------------------------------------------- #

def _process_sample(sample_dir: Path, overwrite: bool) -> str:
    name = sample_dir.name

    # Locate depth and pcpout tifs (pattern: depth.<name>.tif)
    depth_tif  = sample_dir / f'depth.{name}.tif'
    pcpout_tif = sample_dir / f'pcpout.{name}.tif'

    if not depth_tif.exists():
        return f'[SKIP] {name}: depth tif not found'
    if not pcpout_tif.exists():
        return f'[SKIP] {name}: pcpout tif not found'

    msgs = []
    for tif, prefix in [(depth_tif, 'depth'), (pcpout_tif, 'pcpout')]:
        out_dir = sample_dir / f'{prefix}_timesteps'
        if not overwrite and _already_done(sample_dir, prefix):
            msgs.append(f'{prefix}=skipped')
            continue
        n = _split(tif, out_dir, prefix)
        msgs.append(f'{prefix}={n}')

    return f'[OK]   {name}: {", ".join(msgs)}'


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_dir', default=str(SAMPLES_DIR))
    p.add_argument('--workers',     type=int, default=8)
    p.add_argument('--overwrite',   action='store_true',
                   help='Re-split even if output files already exist')
    args = p.parse_args()

    samples_dir = Path(args.samples_dir)
    sample_dirs = sorted(samples_dir.iterdir(), key=lambda d: (len(d.name), d.name))
    sample_dirs = [d for d in sample_dirs if d.is_dir()]

    print(f'Found {len(sample_dirs)} samples in {samples_dir}')
    print(f'Workers : {args.workers}  |  overwrite={args.overwrite}')

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_sample, d, args.overwrite): d
            for d in sample_dirs
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Splitting', unit='sample'):
            results.append(fut.result())

    # Print summary
    ok   = [r for r in results if r.startswith('[OK]')]
    skip = [r for r in results if r.startswith('[SKIP]')]
    err  = [r for r in results if r.startswith('[ERR]')]

    print(f'\n=== Done: {len(ok)} split, {len(skip)} skipped, {len(err)} errors ===')
    if err:
        print('\nErrors:')
        for e in err:
            print(' ', e)


if __name__ == '__main__':
    main()
