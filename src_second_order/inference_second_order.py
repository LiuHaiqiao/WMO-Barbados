"""
inference_second_order.py — Autoregressive full-sequence inference for Diff2LitModel.

Starting from t0, uses the Δ²D-space TFNO to autoregressively predict water depth
for every subsequent hour, driven by actual rainfall files as forcing.

Reconstruction at each step:
    D̂_{t+k+1} = Δ²D̂_{t+k+1} + 2·D_{t+k} - D_{t+k-1}    (clamp ≥ 0)

Outputs
-------
<out_dir>/predictions.npy      (T, H, W) float32  — normalised depth (× NORM_DEPTH/1000 → m)
<out_dir>/pred_hr%04d.tif      one GeoTIF per step, float32, depth in metres
<out_dir>/metrics.csv          per-step RMSE / MAE / Bias vs ground truth (when available)
<out_dir>/summary.png          mean-depth time series + snapshots at peak step

Usage
-----
python src_second_order/inference_second_order.py \\
    --ckpt logs/tfno_diff2_lb4_.../checkpoints/best.ckpt \\
    --sim_dir /home/hl1138/TFNO/data/samples/sample1 \\
    --t0 5 \\
    --out_dir outputs/sample1_pred

# predict all available steps from first valid t0
python src_second_order/inference_second_order.py \\
    --ckpt logs/.../best.ckpt --sim_dir /path/to/sim --out_dir outputs/run1
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import torch

from fno_model import TFNOFlood
from train_second_order_fno import Diff2LitModel, NORM_DEPTH, NORM_RAIN, _read, _load_static


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #

def load_from_checkpoint(ckpt_path: str, device: torch.device) -> tuple[Diff2LitModel, dict]:
    """
    Reconstruct TFNOFlood from the model_cfg stored in the checkpoint's
    hyper_parameters, then load the full LightningModule weights.

    Uses manual state_dict loading to avoid the PyTorch _metadata key that
    some versions serialize into the checkpoint and that Lightning strict-mode
    rejects as an unexpected key.

    Returns (lit_model_in_eval_mode, hparams_dict).
    """
    ckpt    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = ckpt['hyper_parameters']
    cfg     = hparams['model_cfg']

    model = TFNOFlood(
        in_channels    = cfg['in_channels'],
        out_channels   = 1,
        hidden_dim     = cfg['hidden_dim'],
        n_layers       = cfg['n_layers'],
        modes1         = cfg['modes1'],
        modes2         = cfg['modes2'],
        rank           = cfg['rank'],
        domain_padding = cfg['domain_padding'],
    )

    lit = Diff2LitModel(
        model           = model,
        lookback        = hparams['lookback'],
        lr              = hparams['lr'],
        weight_decay    = hparams['weight_decay'],
        patience        = hparams['patience'],
        n_rollout_steps = hparams['n_rollout_steps'],
        lambda_phys     = hparams['lambda_phys'],
        truncate_bptt   = hparams['truncate_bptt'],
        model_cfg       = cfg,
    )

    state_dict = ckpt['state_dict']
    state_dict.pop('_metadata', None)   # PyTorch internal key, not a real parameter
    lit.load_state_dict(state_dict, strict=True)

    lit.to(device).eval()
    return lit, hparams


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def run_inference(
    lit:         Diff2LitModel,
    depth_files: list[Path],
    rain_files:  list[Path],
    static:      torch.Tensor,   # (4, H, W)
    land_mask:   torch.Tensor,   # (1, H, W)
    t0:          int,
    n_steps:     int,
    lookback:    int,
    device:      torch.device,
    alpha:       float = 0.0,
) -> np.ndarray:
    """
    Autoregressive inference over the full spatial domain with optional
    residual adjustment to suppress error drift.

    Each step follows the 5-stage loop:
      1. Base prediction    : Ŷ_base  = model(x)
      2. Residual adjust    : Ŷ_reg   = Ŷ_base + α·e          (e=0 at step 0)
      3. Depth reconstruct  : D̂_next = Ŷ_reg + 2·D_curr - D_prev   (clamp ≥ 0)
      4. Virtual Δ²D        : Y_virt  = D̂_next - 2·D_curr + D_prev
      5. Update residual    : e       = Y_virt - Ŷ_base

    Y_virt ≠ Ŷ_reg whenever the clamp fires; e captures that physical violation
    so α·e pre-corrects the next step.  alpha=0 disables the mechanism entirely.

    Returns
    -------
    preds : (n_steps, H, W) float32 array of normalised predicted depths.
            Multiply by NORM_DEPTH / 1000 to convert to metres.
    """
    def load_depth(i: int) -> torch.Tensor:
        return torch.from_numpy(_read(depth_files[i]) / NORM_DEPTH)

    def load_rain(i: int) -> torch.Tensor:
        return torch.from_numpy(_read(rain_files[i]) / NORM_RAIN)

    # ── Build initial windows ──────────────────────────────────────────────
    # depths[j] = D_{t0 - L - 1 + j},  j ∈ [0, L+1]
    # depths[L]   = D_{t0-1}
    # depths[L+1] = D_{t0}
    depths = [load_depth(t0 - lookback - 1 + j) for j in range(lookback + 2)]

    # Δ²D history: (L, H, W)
    d2_win = torch.stack(
        [depths[k+1] - 2.0*depths[k] + depths[k-1] for k in range(1, lookback + 1)],
        dim=0,
    ).to(device)

    # Current rain: R_{t0}  (H, W) → (1, H, W)
    rain_curr = load_rain(t0).unsqueeze(0).to(device)

    d_prev = depths[lookback].unsqueeze(0).to(device)      # (1, H, W)  D_{t0-1}
    d_curr = depths[lookback + 1].unsqueeze(0).to(device)  # (1, H, W)  D_{t0}

    static_b = static.unsqueeze(0).to(device)  # (1, 4, H, W)

    # Residual e tracks drift from the clamp: shape (1, H, W), init to zero
    e = torch.zeros_like(d_curr)

    preds: list[np.ndarray] = []

    with torch.no_grad():
        for k in range(n_steps):
            x = torch.cat(
                [static_b, d2_win.unsqueeze(0), rain_curr.unsqueeze(0)], dim=1
            )  # (1, 4+L+1, H, W)

            # Step 1 — base prediction
            pred_d2_base = lit.model(x)                                    # (1, 1, H, W)

            # Step 2 — residual adjustment  (e is (1,H,W); unsqueeze to (1,1,H,W))
            pred_d2_reg = pred_d2_base + alpha * e.unsqueeze(1)            # (1, 1, H, W)

            # Step 3 — inverse Δ²D → depth reconstruction (clamp enforces D ≥ 0)
            d_next = (pred_d2_reg[:, 0] + 2.0*d_curr - d_prev).clamp(min=0)  # (1, H, W)

            preds.append(d_next.squeeze(0).cpu().numpy())  # (H, W)

            # Step 4 — virtual Δ²D actually applied after clamping
            y_virtual = d_next - 2.0*d_curr + d_prev                      # (1, H, W)

            # Step 5 — update residual for next step
            e = y_virtual - pred_d2_base[:, 0]                            # (1, H, W)

            # Advance depth state
            d_prev = d_curr
            d_curr = d_next

            # Slide Δ²D window with the actually-applied virtual value
            d2_win = torch.cat([d2_win[1:], y_virtual[0].unsqueeze(0)], dim=0)

            # Advance current rain: R_{t0+k+1}
            future_t = t0 + k + 1
            if future_t < len(rain_files):
                rain_curr = load_rain(future_t).unsqueeze(0).to(device)
            # else: keep last known rain (no file available — end of simulation)

    return np.stack(preds, axis=0)  # (n_steps, H, W)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(
    preds:       np.ndarray,   # (T, H, W) normalised
    depth_files: list[Path],
    land_mask:   np.ndarray,   # (H, W) bool
    t0:          int,
    n_steps:     int,
) -> list[dict]:
    """Per-step RMSE, MAE, Bias in metres vs ground-truth depth files."""
    rows = []
    for k in range(n_steps):
        gt_idx = t0 + k + 1
        if gt_idx >= len(depth_files):
            break
        gt  = _read(depth_files[gt_idx]) / NORM_DEPTH  # (H, W) normalised
        pr  = preds[k]                                  # (H, W) normalised

        # Convert diff to metres for reporting
        scale = NORM_DEPTH / 1000.0
        err   = (pr[land_mask] - gt[land_mask]) * scale

        rows.append({
            'step':      k + 1,
            'time_idx':  gt_idx,
            'rmse_m':    float(np.sqrt((err**2).mean())),
            'mae_m':     float(np.abs(err).mean()),
            'bias_m':    float(err.mean()),
            'gt_mean_m': float(gt[land_mask].mean() * scale),
            'pr_mean_m': float(pr[land_mask].mean() * scale),
        })
    return rows


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def save_geotiffs(
    preds:       np.ndarray,  # (T, H, W) normalised
    ref_path:    Path,         # any input TIF for CRS / transform
    out_dir:     Path,
    t0:          int,
) -> None:
    """Write one float32 GeoTIF per step, depth in metres."""
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()

    profile.update(dtype='float32', count=1, nodata=-9999.0)
    scale = NORM_DEPTH / 1000.0

    for k, pred in enumerate(preds):
        out_path = out_dir / f'pred_hr{t0 + k + 1:04d}.tif'
        data     = pred * scale  # → metres
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data.astype('float32'), 1)


def save_metrics(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    path = out_dir / 'metrics.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Metrics saved : {path}')


def save_summary_figure(
    preds:       np.ndarray,   # (T, H, W) normalised
    metrics:     list[dict],
    land_mask:   np.ndarray,   # (H, W) bool
    out_dir:     Path,
    t0:          int,
) -> None:
    scale = NORM_DEPTH / 1000.0
    mean_pred = np.array([p[land_mask].mean() * scale for p in preds])
    steps     = np.arange(1, len(preds) + 1)

    peak_k = int(np.argmax(mean_pred))

    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)

    # ── Left: mean depth time series ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(steps + t0, mean_pred, color='steelblue', lw=1.5, label='Predicted mean depth')
    if metrics:
        gt_mean = [r['gt_mean_m'] for r in metrics]
        ax0.plot([r['time_idx'] for r in metrics], gt_mean,
                 color='coral', lw=1.5, linestyle='--', label='GT mean depth')
    ax0.axvline(peak_k + t0 + 1, color='gray', lw=0.8, linestyle=':')
    ax0.set_xlabel('Time step')
    ax0.set_ylabel('Mean depth (m)')
    ax0.set_title('Domain-mean depth over land')
    ax0.legend(fontsize=8)

    # ── Middle: predicted depth at peak step ──────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    snap = preds[peak_k] * scale
    snap_masked = np.where(land_mask, snap, np.nan)
    vmax = float(np.nanpercentile(snap_masked, 99))
    im1  = ax1.imshow(snap_masked, cmap='Blues', vmin=0, vmax=max(vmax, 1e-6),
                      origin='upper', aspect='auto')
    ax1.set_title(f'Predicted depth\n(step {peak_k+1}, t={t0+peak_k+1})', fontsize=9)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='m')

    # ── Right: RMSE over time (if metrics available) ──────────────────────
    ax2 = fig.add_subplot(gs[2])
    if metrics:
        rmse_steps = [r['step'] + t0 for r in metrics]
        rmse_vals  = [r['rmse_m'] for r in metrics]
        ax2.plot(rmse_steps, rmse_vals, color='darkorange', lw=1.5)
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('RMSE (m)')
        ax2.set_title('Per-step RMSE vs GT')
    else:
        ax2.text(0.5, 0.5, 'No ground truth\navailable',
                 ha='center', va='center', transform=ax2.transAxes, color='gray')
        ax2.set_title('Per-step RMSE vs GT')

    fig.suptitle(f'Inference from t0={t0}  ({len(preds)} steps predicted)', fontsize=11)
    fig.tight_layout()
    out_path = out_dir / 'summary.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary figure: {out_path}')


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='Autoregressive inference — Diff2 TFNO')

    p.add_argument('--ckpt',       required=True,
                   help='Path to Lightning checkpoint (.ckpt)')
    p.add_argument('--sim_dir',    required=True,
                   help='Simulation directory containing depth_timesteps/ and pcpout_timesteps/')
    p.add_argument('--static_dir', default='/home/hl1138/TFNO/data/parms_bands',
                   help='Shared static features directory')
    p.add_argument('--out_dir',    required=True,
                   help='Output directory (created if absent)')
    p.add_argument('--t0',         type=int, default=None,
                   help='Initial time index (default: first valid = lookback+1)')
    p.add_argument('--n_steps',    type=int, default=None,
                   help='Steps to predict (default: all remaining from t0)')
    p.add_argument('--no_tifs',    action='store_true',
                   help='Skip per-step GeoTIF output (faster, saves disk)')
    p.add_argument('--no_metrics', action='store_true',
                   help='Skip ground-truth comparison')
    p.add_argument('--alpha',      type=float, default=0.0,
                   help='Residual adjustment strength α ∈ [0, 1]. '
                        '0 = disabled (pure autoregressive); '
                        '0.1–0.5 typical; tunes on validation set.')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Torch device (e.g. cuda, cuda:1, cpu)')
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_dir = Path(args.sim_dir)
    depth_files = sorted((sim_dir / 'depth_timesteps').glob('depth_hr????.00.tif'))
    rain_files  = sorted((sim_dir / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))

    if not depth_files:
        raise FileNotFoundError(f'No depth_hr????.00.tif files found in {sim_dir}/depth_timesteps/')
    if not rain_files:
        raise FileNotFoundError(f'No pcpout_hr????.00.tif files found in {sim_dir}/pcpout_timesteps/')

    print(f'Depth files   : {len(depth_files)}  (t=0 … {len(depth_files)-1})')
    print(f'Rain  files   : {len(rain_files)}')

    # ── Load model ────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.ckpt}')
    lit, hparams = load_from_checkpoint(args.ckpt, device)
    lookback     = hparams['lookback']
    print(f'Lookback      : {lookback}')
    print(f'Device        : {device}')

    # ── Resolve t0 and n_steps ────────────────────────────────────────────
    t0_min  = lookback + 1
    t0      = args.t0 if args.t0 is not None else t0_min
    if t0 < t0_min:
        raise ValueError(f'--t0 must be ≥ {t0_min} (lookback+1) to have enough history; got {t0}')

    max_steps = min(len(depth_files), len(rain_files)) - t0 - 1
    n_steps   = args.n_steps if args.n_steps is not None else max_steps
    n_steps   = min(n_steps, max_steps)
    if n_steps <= 0:
        raise ValueError(f'No steps to predict from t0={t0} with {len(depth_files)} files.')

    print(f't0            : {t0}')
    print(f'Steps         : {n_steps}  (predicting t={t0+1} … t={t0+n_steps})')
    print(f'Alpha (resid) : {args.alpha}')

    # ── Load static features ──────────────────────────────────────────────
    static, land_mask = _load_static(Path(args.static_dir))
    land_np = land_mask[0].numpy().astype(bool)  # (H, W)

    # ── Run inference ─────────────────────────────────────────────────────
    print('Running inference …')
    preds = run_inference(
        lit, depth_files, rain_files,
        static, land_mask,
        t0=t0, n_steps=n_steps, lookback=lookback, device=device,
        alpha=args.alpha,
    )
    print(f'Predictions shape: {preds.shape}')

    # ── Save NPY ──────────────────────────────────────────────────────────
    npy_path = out_dir / 'predictions.npy'
    np.save(npy_path, preds)
    print(f'Predictions saved : {npy_path}')

    # ── Save GeoTIFs ──────────────────────────────────────────────────────
    if not args.no_tifs:
        print('Writing GeoTIFs …')
        save_geotiffs(preds, ref_path=depth_files[0], out_dir=out_dir, t0=t0)
        print(f'GeoTIFs saved in  : {out_dir}')

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = []
    if not args.no_metrics:
        print('Computing metrics vs ground truth …')
        metrics = compute_metrics(preds, depth_files, land_np, t0=t0, n_steps=n_steps)
        if metrics:
            save_metrics(metrics, out_dir)
            rmse_all = [r['rmse_m'] for r in metrics]
            mae_all  = [r['mae_m']  for r in metrics]
            print(f'Mean RMSE: {np.mean(rmse_all):.4f} m   '
                  f'Mean MAE: {np.mean(mae_all):.4f} m   '
                  f'(over {len(metrics)} steps)')

    # ── Summary figure ────────────────────────────────────────────────────
    save_summary_figure(preds, metrics, land_np, out_dir, t0)
    print('Done.')


if __name__ == '__main__':
    main()
