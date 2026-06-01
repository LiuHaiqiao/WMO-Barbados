"""
train_second_order_fno.py — TFNO trained in second-order-difference (Δ²D) space.

Instead of predicting water depth D_{t+1} directly, the model predicts:

    Δ²D_{t+1} = D_{t+1} - 2·D_t + D_{t-1}          ("depth acceleration")

Depth is recovered via:

    D̂_{t+1} = Δ²D̂_{t+1} + 2·D_t - D_{t-1}

Working in Δ²-space removes the slow mean-drift component, leaving a
zero-centred signal that is easier to learn, especially during dry periods
where the signal is near zero.

Input channels : [static(4) | Δ²D_hist(lookback) | Rain_t(1)]  = 4+L+1
Output channels: Δ²D_{t+1}                                     = 1

Dataset note
------------
A valid time step t requires:
  depth files  : t-L-1  …  t+N      (to build Δ²D history and targets)
  rain  files  : t  …  t+N-1        (current rain + future sliding)
so t_min = L+1,  t_max = min(T_depth - N, T_rain - N + 1)

Usage
-----
CUDA_VISIBLE_DEVICES=2 python src/train_second_order_fno.py
CUDA_VISIBLE_DEVICES=2 python src/train_second_order_fno.py \\
    --lookback 4 --n_steps 6 --batch_size 2
"""

import argparse
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from fno_model import TFNOFlood


# --------------------------------------------------------------------------- #
# Normalisation constants
# --------------------------------------------------------------------------- #
NORM_DEM   = 400.0
NORM_MAN   = 0.16
NORM_SLOPE = 40.0
NORM_RAIN  = 100.0
NORM_DEPTH = 30000.0


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


def _load_static(static_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    dem      = _read(static_dir / 'dem.tif')           / NORM_DEM
    manning  = _read(static_dir / 'manning_coef.tif')  / NORM_MAN
    pervious = _read(static_dir / 'pervious_cover.tif')
    slope    = _read(static_dir / 'slope.tif')         / NORM_SLOPE
    static   = torch.from_numpy(np.stack([dem, manning, pervious, slope], axis=0))
    mask     = torch.from_numpy(_read(static_dir / 'land_mask.tif')[None])
    return static, mask


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
        print(f'[data] {len(ready)}/{len(dirs)} samples ready '
              f'({skipped} skipped — not yet split)')
    return ready


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class Diff2FloodDataset(Dataset):
    """
    Per-item shapes (P = patch_size, L = lookback, N = n_steps):

        static       (4,   P, P)  DEM / Manning / Pervious / Slope
        diff2_hist   (L,   P, P)  Δ²D_{t-L+1} … Δ²D_t
        rain_curr    (1,   P, P)  R_t  — current rainfall only
        depth_tm1    (1,   P, P)  D_{t-1}  — reconstruction anchor
        depth_t      (1,   P, P)  D_t      — reconstruction anchor
        diff2_tgt    (N,   P, P)  Δ²D_{t+1} … Δ²D_{t+N}   (targets)
        depth_tgt    (N,   P, P)  D_{t+1}  … D_{t+N}       (depth-space eval)
        rain_future  (N-1, P, P)  R_{t+1}  … R_{t+N-1}     (future forcing)
        land         (1,   P, P)  land mask
    """

    def __init__(
        self,
        sample_dirs: list[Path],
        static:      torch.Tensor,
        land_mask:   torch.Tensor,
        patch_size:  int = 512,
        stride:      int = 568,
        lookback:    int = 4,
        n_steps:     int = 4,
    ):
        self.patch_size = patch_size
        self.lookback   = lookback
        self.n_steps    = n_steps
        self.static     = static
        self.land_mask  = land_mask
        self.samples: list[tuple] = []

        H, W    = static.shape[1], static.shape[2]
        offsets = _patch_offsets(H, W, patch_size, stride)

        for d in sample_dirs:
            depth_files = sorted((d / 'depth_timesteps').glob('depth_hr????.00.tif'))
            rain_files  = sorted((d / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))

            if len(depth_files) != len(rain_files):
                print(f'[data] WARNING: skipping {d.name} — '
                      f'file count mismatch '
                      f'({len(depth_files)} depth vs {len(rain_files)} rain)')
                continue

            # depth: need indices [t-L-1 … t+N]  → t_max = T - N (exclusive)
            # rain : need indices [t-L+1 … t+N-1] → t_max = T - N + 1 (exclusive)
            t_min = lookback + 1
            t_max = min(len(depth_files) - n_steps, len(rain_files) - n_steps + 1)
            for t in range(t_min, t_max):
                for (r, c) in offsets:
                    self.samples.append((rain_files, depth_files, t, r, c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rain_files, depth_files, t, r, c = self.samples[idx]
        L = self.lookback
        N = self.n_steps
        p = self.patch_size

        def crop(arr: np.ndarray) -> np.ndarray:
            return arr[r:r+p, c:c+p]

        def load_depth(i: int) -> np.ndarray:
            return crop(_read(depth_files[i])) / NORM_DEPTH

        def load_rain(i: int) -> np.ndarray:
            return crop(_read(rain_files[i])) / NORM_RAIN

        # ── Δ²D history ────────────────────────────────────────────────────
        # depths[j] = D_{t-L-1+j}  for j in [0, L+1]
        # depths[0] = D_{t-L-1},  depths[L] = D_{t-1},  depths[L+1] = D_t
        depths = [load_depth(t - L - 1 + j) for j in range(L + 2)]

        # Δ²D_{t-L+k} = depths[k+1] - 2·depths[k] + depths[k-1], k ∈ [1, L]
        diff2_hist = np.stack(
            [depths[k+1] - 2.0*depths[k] + depths[k-1] for k in range(1, L + 1)],
            axis=0,
        )  # (L, P, P)

        # ── Current rain ───────────────────────────────────────────────────
        rain_curr = load_rain(t)  # R_t  (P, P)

        # ── Reconstruction anchors ─────────────────────────────────────────
        depth_tm1 = depths[L]      # D_{t-1}  (P, P)
        depth_t   = depths[L + 1]  # D_t      (P, P)

        # ── N-step targets ─────────────────────────────────────────────────
        future_d = [load_depth(t + k) for k in range(1, N + 1)]  # D_{t+1}…D_{t+N}

        d_a, d_b = depth_tm1, depth_t
        diff2_tgt_list = []
        for d_c in future_d:
            diff2_tgt_list.append(d_c - 2.0*d_b + d_a)
            d_a, d_b = d_b, d_c
        diff2_tgt = np.stack(diff2_tgt_list, axis=0)  # (N, P, P)
        depth_tgt = np.stack(future_d,       axis=0)  # (N, P, P)

        # ── Future rain for window sliding (N-1 steps) ─────────────────────
        rain_future = (
            np.stack([load_rain(t + k) for k in range(1, N)], axis=0)
            if N > 1
            else np.empty((0, p, p), dtype=np.float32)
        )  # (N-1, P, P)

        # ── Patch static / mask ────────────────────────────────────────────
        static_p = self.static[:, r:r+p, c:c+p]    # (4, P, P)
        land_p   = self.land_mask[:, r:r+p, c:c+p]  # (1, P, P)

        return (
            static_p,
            torch.from_numpy(diff2_hist).unsqueeze(1),                    # (L, 1, P, P)
            torch.from_numpy(rain_curr[np.newaxis]),                      # (1, P, P)
            torch.from_numpy(depth_tm1[np.newaxis]),                      # (1, P, P)
            torch.from_numpy(depth_t[np.newaxis]),                        # (1, P, P)
            torch.from_numpy(diff2_tgt).unsqueeze(1),                     # (N, 1, P, P)
            torch.from_numpy(depth_tgt).unsqueeze(1),                     # (N, 1, P, P)
            torch.from_numpy(rain_future).unsqueeze(1),                   # (N-1, 1, P, P)
            land_p,                                                        # (1, P, P)
        )


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #

def build_diff2_loaders(
    samples_dir: str | Path = '/home/hl1138/TFNO/data/samples',
    static_dir:  str | Path = '/home/hl1138/TFNO/data/parms_bands',
    patch_size:  int   = 512,
    stride:      int   = 568,
    lookback:    int   = 4,
    n_steps:     int   = 4,
    val_split:   float = 0.15,
    test_split:  float = 0.15,
    batch_size:  int   = 4,
    num_workers: int   = 4,
    seed:        int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
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

    def _make_loader(dirs, shuffle: bool) -> DataLoader:
        ds = Diff2FloodDataset(
            dirs, static, land_mask,
            patch_size=patch_size, stride=stride,
            lookback=lookback, n_steps=n_steps,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        _make_loader(train_dirs, shuffle=True),
        _make_loader(val_dirs,   shuffle=False),
        _make_loader(test_dirs,  shuffle=False),
    )


# --------------------------------------------------------------------------- #
# Lightning module
# --------------------------------------------------------------------------- #

class Diff2LitModel(pl.LightningModule):
    """
    Trains TFNOFlood to predict Δ²D instead of raw depth.

    Rollout at step k:
        x        = [static | Δ²D_window | rain_curr]    # 4+L+1 channels
        pred_d2  = model(x)                              # Δ²D_{t+k+1}
        d_next   = pred_d2 + 2·d_curr - d_prev          # reconstructed D_{t+k+1}
        slide Δ²D window ← pred_d2
        rain_curr ← R_{t+k+1}  (from rain_future)
    """

    def __init__(
        self,
        model:           nn.Module,
        lookback:        int   = 4,
        lr:              float = 1e-3,
        weight_decay:    float = 1e-4,
        patience:        int   = 5,
        n_rollout_steps: int   = 4,
        lambda_phys:     float = 0.0,
        truncate_bptt:   bool  = True,
        model_cfg:       dict  = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

    # ------------------------------------------------------------------ #

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        diff2 = (pred - target) ** 2 * mask
        return diff2.sum() / mask.sum().clamp(min=1)

    @staticmethod
    def _masked_mae(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        (static, diff2_hist, rain_curr,
         depth_tm1, depth_t,
         diff2_tgt, depth_tgt, rain_fut, land) = batch

        B  = static.shape[0]
        L  = self.hparams.lookback
        N  = self.hparams.n_rollout_steps
        on_step = (stage == 'train')
        land = land.float()

        d2_win    = diff2_hist[:, :, 0]  # (B, L, P, P)
        rain_curr = rain_curr            # (B, 1, P, P)
        d_prev    = depth_tm1            # (B, 1, P, P)
        d_curr    = depth_t              # (B, 1, P, P)

        total_loss = torch.zeros(1, device=static.device, dtype=static.dtype).squeeze()

        for k in range(N):
            x = torch.cat([static, d2_win, rain_curr], dim=1)  # (B, 4+L+1, P, P)
            pred_d2 = self(x)                                    # (B, 1, P, P)
            tgt_d2  = diff2_tgt[:, k]                            # (B, 1, P, P)

            loss_k = self._masked_mse(pred_d2, tgt_d2, land)
            total_loss = total_loss + loss_k

            # Reconstruct depth; clamp so predictions stay physically valid
            d_next = (pred_d2 + 2.0 * d_curr - d_prev).clamp(min=0.0)

            # Physics: reconstructed depth should not rise when rain-free
            if self.hparams.lambda_phys > 0.0:
                no_rain  = (rain_curr < 1e-3).float()
                increase = nn.functional.relu(d_next - d_curr)
                l_phys   = (increase * no_rain * land).sum() / land.sum().clamp(min=1)
                total_loss = total_loss + self.hparams.lambda_phys * l_phys
                self.log(f'{stage}/phys{k+1}', l_phys, on_epoch=True, on_step=False)

            self.log(f'{stage}/diff2_mse{k+1}', loss_k,
                     prog_bar=(k == 0), on_epoch=True, on_step=on_step)
            self.log(f'{stage}/depth_mse{k+1}',
                     self._masked_mse(d_next, depth_tgt[:, k], land),
                     on_epoch=True, on_step=False)
            self.log(f'{stage}/depth_mae{k+1}',
                     self._masked_mae(d_next, depth_tgt[:, k], land),
                     on_epoch=True, on_step=False)

            # Advance state; detach to keep O(1) memory per step
            if self.hparams.truncate_bptt:
                d_prev = d_curr.detach()
                d_curr = d_next.detach()
                d2_new = pred_d2.detach()[:, 0]  # (B, P, P)
            else:
                d_prev = d_curr
                d_curr = d_next
                d2_new = pred_d2[:, 0]

            # Slide Δ²D window: drop oldest, append predicted
            d2_win = torch.cat([d2_win[:, 1:], d2_new.unsqueeze(1)], dim=1)

            # Advance rain: R_{t+k+1} becomes next current rain
            if k < N - 1:
                rain_curr = rain_fut[:, k]  # (B, 1, P, P)

        self.log(f'{stage}/loss', total_loss, prog_bar=True, on_epoch=True, on_step=on_step)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')
        if batch_idx == 0:
            self._log_depth_figures(batch)

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, 'test')

    def _log_depth_figures(self, batch) -> None:
        """Log a 2×3 grid (step1/step2) × (pred/gt/diff) of reconstructed depth."""
        (static, diff2_hist, rain_curr,
         depth_tm1, depth_t,
         _, depth_tgt, rain_fut, land) = batch

        d2_win    = diff2_hist[0:1, :, 0]  # (1, L, P, P) — first sample only
        rain_curr = rain_curr[0:1]          # (1, 1, P, P)
        d_prev    = depth_tm1[0:1]
        d_curr    = depth_t[0:1]

        preds = []
        with torch.no_grad():
            for k in range(min(2, self.hparams.n_rollout_steps)):
                x = torch.cat([static[0:1], d2_win, rain_curr], dim=1)
                pred_d2 = self(x)
                d_next  = (pred_d2 + 2.0 * d_curr - d_prev).clamp(min=0.0)
                preds.append(d_next)

                d_prev = d_curr
                d_curr = d_next
                d2_win = torch.cat([d2_win[:, 1:], pred_d2[:, 0].unsqueeze(1)], dim=1)
                if k < self.hparams.n_rollout_steps - 1 and rain_fut.shape[1] > k:
                    rain_curr = rain_fut[0:1, k]  # (1, 1, P, P)

        land_np = land[0, 0].cpu().numpy().astype(bool)

        def _np(t: torch.Tensor) -> np.ndarray:
            arr = t[0, 0].cpu().float().numpy() * NORM_DEPTH / 1000
            arr[~land_np] = np.nan
            return arr

        rows = [(preds[k], depth_tgt[0:1, k]) for k in range(len(preds))]
        step_labels = [f'Step {k+1}' for k in range(len(rows))]

        vmax = max(
            float(np.nanmax(np.abs(_np(gt)))) for _, gt in rows
        )
        vmax = max(vmax, 1e-6)
        dmax = max(
            float(np.nanmax(np.abs(_np(p) - _np(gt)))) for p, gt in rows
        )
        dmax = max(dmax, 1e-6)

        fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
        if len(rows) == 1:
            axes = axes[np.newaxis, :]

        col_titles = ['Prediction', 'Ground Truth', 'Difference']
        for row_i, ((pred, gt), label) in enumerate(zip(rows, step_labels)):
            p_np, g_np = _np(pred), _np(gt)
            diff_np    = p_np - g_np
            for col_i, (data, title) in enumerate(
                zip([p_np, g_np, diff_np], col_titles)
            ):
                ax   = axes[row_i, col_i]
                cmap = 'RdBu_r' if col_i == 2 else 'Blues'
                vmin_ = -dmax if col_i == 2 else 0.0
                vmax_ =  dmax if col_i == 2 else vmax
                im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_,
                               origin='upper', aspect='auto')
                ax.set_title(f'{label} — {title}', fontsize=9)
                ax.axis('off')
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label('m', fontsize=8)

        fig.suptitle(f'Epoch {self.current_epoch} — reconstructed depth (m)', fontsize=11)
        fig.tight_layout()
        self.logger.experiment.add_figure(
            'val/depth_predictions', fig, global_step=self.current_epoch
        )
        plt.close(fig)

    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5,
            patience=self.hparams.patience,
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss'},
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='Train TFNOFlood in Δ²D space')

    # Data
    p.add_argument('--samples_dir', default='/home/hl1138/TFNO/data/samples')
    p.add_argument('--static_dir',  default='/home/hl1138/TFNO/data/parms_bands')
    p.add_argument('--patch_size',  type=int,   default=512)
    p.add_argument('--stride',      type=int,   default=568)
    p.add_argument('--val_split',   type=float, default=0.15)
    p.add_argument('--test_split',  type=float, default=0.15)
    p.add_argument('--num_workers', type=int,   default=4)

    # Differencing
    p.add_argument('--lookback', type=int, default=4,
                   help='Number of historical Δ²D and Rain steps fed as input (L). '
                        'Sets in_channels = 4 + 2·L.')

    # Model
    p.add_argument('--hidden_dim',     type=int,   default=64)
    p.add_argument('--n_layers',       type=int,   default=4)
    p.add_argument('--modes1',         type=int,   default=16)
    p.add_argument('--modes2',         type=int,   default=16)
    p.add_argument('--rank',           type=float, default=0.1)
    p.add_argument('--domain_padding', type=float, default=0.1)

    # Training
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_epochs',   type=int,   default=100)
    p.add_argument('--patience',     type=int,   default=5,
                   help='ReduceLROnPlateau patience')
    p.add_argument('--early_stop',   type=int,   default=15,
                   help='EarlyStopping patience in epochs; 0 to disable')
    p.add_argument('--n_steps',      type=int,   default=4,
                   help='Autoregressive rollout steps in training loss')
    p.add_argument('--lambda_phys',  type=float, default=0.0,
                   help='Physics regularisation weight (0 = disabled)')
    p.add_argument('--full_bptt',    action='store_true', default=False,
                   help='Keep full computation graph across rollout steps '
                        '(high memory; default: truncated BPTT)')

    # Compilation
    p.add_argument('--compile',      action='store_true', default=False,
                   help='Compile the model with torch.compile (requires PyTorch ≥ 2.0)')
    p.add_argument('--compile_mode', default='default',
                   choices=['default', 'max-autotune-no-cudagraphs'],
                   help='torch.compile mode. reduce-overhead / max-autotune are excluded '
                        'because neuralop grid tensors are incompatible with CUDA Graphs.')

    # Infra
    p.add_argument('--log_dir',   default='logs')
    p.add_argument('--exp_name',  default=None)
    p.add_argument('--ckpt_path', default=None, help='Resume from checkpoint')
    p.add_argument('--precision', default='32',
                   choices=['32', '16-mixed', 'bf16-mixed'])
    p.add_argument('--devices',   type=int, default=1)

    return p.parse_args()


def _auto_exp_name(args) -> str:
    return (
        f"tfno_diff2"
        f"_lb{args.lookback}"
        f"_h{args.hidden_dim}"
        f"_l{args.n_layers}"
        f"_m{args.modes1}x{args.modes2}"
        f"_p{args.patch_size}"
        f"_n{args.n_steps}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    if args.exp_name is None:
        args.exp_name = _auto_exp_name(args)
    print(f'Experiment    : {args.exp_name}')

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_diff2_loaders(
        samples_dir = args.samples_dir,
        static_dir  = args.static_dir,
        patch_size  = args.patch_size,
        stride      = args.stride,
        lookback    = args.lookback,
        n_steps     = args.n_steps,
        val_split   = args.val_split,
        test_split  = args.test_split,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    print(f'Train samples : {len(train_loader.dataset)}')
    print(f'Val   samples : {len(val_loader.dataset)}')
    print(f'Test  samples : {len(test_loader.dataset)}')

    # ── Model ─────────────────────────────────────────────────────────────
    in_channels = 4 + args.lookback + 1
    model = TFNOFlood(
        in_channels    = in_channels,
        out_channels   = 1,
        hidden_dim     = args.hidden_dim,
        n_layers       = args.n_layers,
        modes1         = args.modes1,
        modes2         = args.modes2,
        rank           = args.rank,
        domain_padding = args.domain_padding,
    )
    print(f'Model params  : {sum(p.numel() for p in model.parameters()):,}')
    print(f'Input channels: {in_channels}  (4 static + {args.lookback} Δ²D + 1 Rain_t)')

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        print(f'torch.compile : enabled  (mode={args.compile_mode})')

    lit = Diff2LitModel(
        model           = model,
        lookback        = args.lookback,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        patience        = args.patience,
        n_rollout_steps = args.n_steps,
        lambda_phys     = args.lambda_phys,
        truncate_bptt   = not args.full_bptt,
        model_cfg       = dict(
            model_type     = 'tfno_diff2',
            in_channels    = in_channels,
            lookback       = args.lookback,
            hidden_dim     = args.hidden_dim,
            n_layers       = args.n_layers,
            modes1         = args.modes1,
            modes2         = args.modes2,
            rank           = args.rank,
            domain_padding = args.domain_padding,
        ),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor   = 'val/loss',
            mode      = 'min',
            filename  = 'best',
            save_last = True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    if args.early_stop > 0:
        callbacks.append(
            EarlyStopping(monitor='val/loss', patience=args.early_stop, mode='min')
        )

    # ── Logger ────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs        = args.max_epochs,
        accelerator       = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices           = args.devices,
        precision         = args.precision,
        callbacks         = callbacks,
        logger            = logger,
        log_every_n_steps = 10,
        gradient_clip_val = 1.0,
    )

    trainer.fit(lit, train_loader, val_loader, ckpt_path=args.ckpt_path)

    print(f'\nBest checkpoint : {trainer.checkpoint_callback.best_model_path}')
    trainer.test(lit, test_loader, ckpt_path='best')

    print(f'TensorBoard logs: {logger.log_dir}')
    print(f'  tensorboard --logdir {args.log_dir}')


if __name__ == '__main__':
    main()
