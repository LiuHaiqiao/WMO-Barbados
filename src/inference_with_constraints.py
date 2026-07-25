"""
inference_with_constraints.py — Auto-regressive inference with physical constraints.

Identical to inference.py but adds a no-rise constraint at each rollout step:
if the local rainfall is below a threshold (default 0.1 mm/hr), the predicted
water depth at that pixel is capped at the current depth.  This prevents the
model from generating spurious inundation during dry or post-storm periods.

Usage
-----
python inference_with_constraints.py \\
    --ckpt_path  logs/tfno_.../checkpoints/best.ckpt \\
    --model_type fno \\
    --data_dir   /home/hl1138/TFNO/data/samples/evt001 \\
    --eval \\
    --rain_threshold 0.1
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Re-use every helper from the base inference module
from inference import (
    _set_deterministic,
    _hann2d,
    load_static,
    load_rain_sequence,
    load_depth_sequence,
    patched_forward,
    evaluate,
    save_geotiff,
    save_gif,
    save_max_depth_plot,
    load_model,
)
from data_loader import NORM_RAIN, NORM_DEPTH, _read


# --------------------------------------------------------------------------- #
# Constrained rollout
# --------------------------------------------------------------------------- #

def run_inference_constrained(
    model:           torch.nn.Module,
    data_dir:        Path,
    static_dir:      Path,
    patch_size:      int,
    stride:          int,
    device:          torch.device,
    rain_threshold:  float = 0.1,   # mm/hr; below this depth cannot rise
    start_step:      int   = 0,
    end_step:        int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Auto-regressive rollout with a no-rise-without-rain constraint.

    After every model call, pixels where rain_t < rain_threshold are clamped:
        depth_next[pixel] = min(depth_next[pixel], depth_t[pixel])

    Returns
    -------
    preds    : list of (1, H, W) float tensors in metres
    static   : (4, H, W)
    land_mask: (1, H, W)
    """
    static, land_mask = load_static(static_dir)
    rain_seq = load_rain_sequence(data_dir)
    T = len(rain_seq)
    if end_step is None:
        end_step = T - 1
    end_step = min(end_step, T - 1)

    if start_step >= end_step:
        raise ValueError(f'start_step ({start_step}) must be < end_step ({end_step})')

    hann = _hann2d(patch_size)

    if start_step == 0:
        depth_t   = torch.zeros_like(rain_seq[0])
        rain_prev = torch.zeros_like(rain_seq[0])
    else:
        depth_files = sorted((data_dir / 'depth_timesteps').glob('depth_hr????.00.tif'))
        depth_t   = torch.from_numpy(_read(depth_files[start_step])[None] / NORM_DEPTH)
        rain_prev = rain_seq[start_step - 1]

    # Normalised threshold: same units as rain_seq tensors (divided by NORM_RAIN)
    thresh_norm = rain_threshold / NORM_RAIN

    n_constrained_total = 0
    predictions = []

    for t in tqdm(range(start_step, end_step), desc='Rolling out (constrained)', unit='step'):
        rain_t = rain_seq[t]
        x_full = torch.cat([static, rain_prev, rain_t, depth_t], dim=0)  # (7,H,W)

        depth_next = patched_forward(
            model, x_full, patch_size, stride, device, hann
        )                                                                   # (1,H,W) norm

        # Physical constraint: no rise where rain is below threshold
        no_rain = rain_t < thresh_norm                                      # (1,H,W) bool
        n_constrained_total += int(no_rain.sum().item())
        depth_next = torch.where(no_rain, torch.min(depth_next, depth_t), depth_next)

        predictions.append(depth_next * NORM_DEPTH / 1000.0)               # → m
        rain_prev = rain_t
        depth_t   = depth_next

    H, W = depth_t.shape[-2], depth_t.shape[-1]
    print(f'Constraint applied: {n_constrained_total:,} pixel-steps '
          f'({100.0 * n_constrained_total / (len(predictions) * H * W):.1f}% of all pixel-steps)')

    return predictions, static, land_mask


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description='Constrained auto-regressive flood depth inference'
    )

    p.add_argument('--ckpt_path',  required=True)
    p.add_argument('--data_dir',   required=True)
    p.add_argument('--static_dir', default='/home/hl1138/surrogate/data/parms_bands')
    p.add_argument('--out_dir',    default=None,
                   help='Output directory (default: predictions/<exp_name>_constrained derived from ckpt path)')
    p.add_argument('--out_path',   default=None)
    p.add_argument('--model_type', default='fno', choices=['fno', 'cnn', 'gnn', 'gno'])
    p.add_argument('--start_step', type=int, default=0)
    p.add_argument('--end_step',   type=int, default=None)
    p.add_argument('--patch_size', type=int, default=512)
    p.add_argument('--stride',     type=int, default=256)
    p.add_argument('--eval',       action='store_true')
    p.add_argument('--gif',        action='store_true')
    p.add_argument('--gif_path',   default=None)
    p.add_argument('--fps',        type=int,   default=5)
    p.add_argument('--max_depth',  type=float, default=None)
    p.add_argument('--max_rain',   type=float, default=None)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',       type=int, default=42)

    # Constraint
    p.add_argument('--rain_threshold', type=float, default=0.1,
                   help='Rainfall threshold in mm/hr below which depth cannot rise (default 0.1)')

    # FNO args
    p.add_argument('--in_channels',    type=int,   default=7)
    p.add_argument('--hidden_dim',     type=int,   default=64)
    p.add_argument('--n_layers',       type=int,   default=4)
    p.add_argument('--modes1',         type=int,   default=16)
    p.add_argument('--modes2',         type=int,   default=16)
    p.add_argument('--rank',           type=float, default=0.42)
    p.add_argument('--domain_padding', type=float, default=0.1)

    # CNN args
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--depth',         type=int, default=4)

    # GNO args
    p.add_argument('--num_heads',    type=int,   default=4)
    p.add_argument('--num_phys',     type=int,   default=8)
    p.add_argument('--global_ratio', type=float, default=0.1)
    p.add_argument('--global_k',     type=int,   default=8)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    _set_deterministic(args.seed)
    device = torch.device(args.device)

    rain_files = sorted(
        (Path(args.data_dir) / 'pcpout_timesteps').glob('pcpout_hr????.00.tif')
    )
    T = len(rain_files)
    end_step = min(args.end_step if args.end_step is not None else T - 1, T - 1)

    sample  = Path(args.data_dir).name
    stem    = f'{sample}_{args.model_type}_constrained_s{args.start_step:04d}_e{end_step:04d}'
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        exp_name = Path(args.ckpt_path).parent.parent.parent.name
        out_dir  = Path('predictions') / f'{exp_name}_constrained'
    out_tif = Path(args.out_path) if args.out_path else out_dir / f'{stem}.tif'
    out_gif = (Path(args.gif_path) if args.gif_path
               else out_dir / f'{stem}.gif' if args.gif
               else None)

    print(f'Model           : {args.model_type}  |  checkpoint: {args.ckpt_path}')
    print(f'Data dir        : {args.data_dir}')
    print(f'Steps           : {args.start_step} → {end_step}')
    print(f'Rain threshold  : {args.rain_threshold} mm/hr')
    print(f'Device          : {device}')
    print(f'TIF out         : {out_tif}')
    if out_gif:
        print(f'GIF out         : {out_gif}')

    model = load_model(Path(args.ckpt_path), args.model_type, args).to(device)

    preds, _, land_mask = run_inference_constrained(
        model          = model,
        data_dir       = Path(args.data_dir),
        static_dir     = Path(args.static_dir),
        patch_size     = args.patch_size,
        stride         = args.stride,
        device         = device,
        rain_threshold = args.rain_threshold,
        start_step     = args.start_step,
        end_step       = end_step,
    )
    print(f'Predicted {len(preds)} steps  '
          f'(hr {args.start_step+1}–{args.start_step+len(preds)})')

    save_geotiff(preds, Path(args.data_dir), out_tif)

    if args.eval:
        metrics = evaluate(preds, Path(args.data_dir), land_mask,
                           start_step=args.start_step)
        print(f'\n=== Evaluation (land pixels only) ===')
        print(f'Mean RMSE : {metrics["rmse_mean"]:.4f} m')
        print(f'Mean MAE  : {metrics["mae_mean"]:.4f} m')
        print(f'Peak RMSE : {max(metrics["rmse_per_step"]):.4f} m  '
              f'(step {np.argmax(metrics["rmse_per_step"])+1})')
        print(f'\n--- Temporal-max depth map (per-pixel max over all timesteps) ---')
        print(f'GT   max depth : {metrics["max_depth_gt"].max().item():.4f} m')
        print(f'Pred max depth : {metrics["max_depth_pred"].max().item():.4f} m')
        print(f'Max-depth RMSE : {metrics["max_depth_rmse"]:.4f} m')
        print(f'Max-depth MAE  : {metrics["max_depth_mae"]:.4f} m')

        max_depth_img = out_dir / f'{stem}_max_depth.png'
        save_max_depth_plot(
            max_depth_gt   = metrics['max_depth_gt'],
            max_depth_pred = metrics['max_depth_pred'],
            land_mask      = land_mask,
            out_path       = max_depth_img,
            data_dir       = Path(args.data_dir),
        )

    if out_gif:
        save_gif(
            preds      = preds,
            data_dir   = Path(args.data_dir),
            land_mask  = land_mask,
            gif_path   = out_gif,
            fps        = args.fps,
            max_depth  = args.max_depth,
            max_rain   = args.max_rain,
            start_step = args.start_step,
        )


if __name__ == '__main__':
    main()
