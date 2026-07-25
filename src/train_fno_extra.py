"""
train_fno_extra.py — PyTorch Lightning training script for TFNOFlood, with
two additional static features: flow accumulation and soil type.

Identical to train_fno.py in every respect (losses, rollout, logging) except
the static feature stack grows from 4 to 6 channels via data_loader_extra.py:

    [DEM, Manning, Pervious, Slope, FlowAccLog, SoilType, Rain_{t-1}, Rain_t, Depth_t]

so in_channels is 9 instead of 7. Checkpoints record extra_static=True in
model_cfg for provenance.

Usage
-----
# single GPU
python train_fno_extra.py --data_dirs /home/hl1138/TFNO/data

# multiple simulation dirs
python train_fno_extra.py --data_dirs /path/sim1 /path/sim2 --batch_size 8

# resume from checkpoint
python train_fno_extra.py --data_dirs /home/hl1138/TFNO/data --ckpt_path logs/tfno_extra/version_0/checkpoints/best.ckpt
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics.functional.image import structural_similarity_index_measure as ssim_metric

from data_loader import NORM_DEPTH
from data_loader_extra import build_loaders
from fno_model import TFNOFlood


# --------------------------------------------------------------------------- #
# Lightning module
# --------------------------------------------------------------------------- #

class FNOLitModel(pl.LightningModule):
    """
    Parameters
    ----------
    model      : instantiated TFNOFlood
    lr         : initial learning rate
    weight_decay
    patience   : ReduceLROnPlateau patience (epochs)
    """

    def __init__(
        self,
        model:            nn.Module,
        lr:               float = 1e-3,
        weight_decay:     float = 1e-4,
        patience:         int   = 5,
        model_cfg:        dict  = None,
        n_rollout_steps:  int   = 2,
        lambda_phys:      float = 0.0,
        truncate_bptt:    bool  = True,
        loss_fn:          str   = 'mse',  # 'mse' | 'peak_weighted_rmse' | 'hybrid'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint['state_dict'].pop('_metadata', None)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _mse(pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Land-masked MSE over the full rollout sequence.
        pred, target : (B, T, 1, H, W)
        mask         : (B, 1, H, W)  float, 1=land
        """
        mask   = mask.unsqueeze(1)
        n_land = mask.sum().clamp(min=1)
        T      = pred.shape[1]
        return (((pred - target) ** 2) * mask).sum() / (n_land * T)

    @staticmethod
    def _peak_weighted_rmse(pred: torch.Tensor, target: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
        """Per-pixel temporal Peak-Weighted RMSE over land pixels.

        For each pixel, q_bar is its mean depth across all T timesteps.
        Timesteps where a pixel exceeds its own temporal mean are upweighted.

        pred, target : (B, T, 1, H, W)
        mask         : (B, 1, H, W)  float, 1=land
        """
        T    = target.shape[1]
        mask = mask.unsqueeze(1)                              # (B, 1, 1, H, W) → broadcast over T
        n_land = mask.sum().clamp(min=1)

        # Per-pixel temporal mean: q_bar[b, 1, 1, h, w]
        q_bar = (target * mask).sum(dim=1, keepdim=True) / \
                mask.sum(dim=1, keepdim=True).clamp(min=1)   # (B, 1, 1, H, W)

        w     = (target + q_bar) / (2.0 * q_bar + 1e-3)     # (B, T, 1, H, W)
        w     = w.clamp(min=0.0)
        diff2 = w * (pred - target) ** 2 * mask
        return torch.sqrt(diff2.sum() / (n_land * T) + 1e-8)

    @staticmethod
    def _peak_aware_mse(pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """MSE weighted by per-pixel temporal peak — same weighting as
        _peak_weighted_rmse but without the sqrt so it stays in MSE units.

        pred, target : (B, T, 1, H, W)
        mask         : (B, 1, H, W)  float, 1=land
        """
        T    = target.shape[1]
        mask = mask.unsqueeze(1)
        n_land = mask.sum().clamp(min=1)
        q_bar = (target * mask).sum(dim=1, keepdim=True) / \
                mask.sum(dim=1, keepdim=True).clamp(min=1)
        w     = (target + q_bar) / (2.0 * q_bar + 1e-3)
        w     = w.clamp(min=0.0)
        return (w * (pred - target) ** 2 * mask).sum() / (n_land * T)

    @staticmethod
    def _ssim_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """1 - SSIM averaged over land-masked (B*T) frames.

        pred, target : (B, T, 1, H, W)
        mask         : (B, 1, H, W)
        """
        B, T = pred.shape[:2]
        # Flatten batch and time → (B*T, 1, H, W) so ssim_metric sees 2-D images
        p = pred.reshape(B * T, 1, pred.shape[-2], pred.shape[-1])
        t = target.reshape(B * T, 1, target.shape[-2], target.shape[-1])
        ssim_val = ssim_metric(p, t, data_range=t.max() - t.min() + 1e-8)
        return 1.0 - ssim_val

    # ---- flood-focused helpers (threshold in metres) ---------------------- #
    #
    # pred/target here are normalised depth (raw_mm / NORM_DEPTH), NOT metres —
    # thresholds expressed in physical units must be divided by NORM_DEPTH
    # before comparing against these tensors, or they silently apply at the
    # wrong physical scale.
    _WET_THRESH_MM   = 10.0   # 1 cm — pixels with GT depth above this are "wet"
    _WET_THRESH      = _WET_THRESH_MM / NORM_DEPTH
    _EXTENT_SIGMA_MM = 50.0   # 5 cm — soft wet/dry transition width for dice
    _EXTENT_SIGMA    = _EXTENT_SIGMA_MM / NORM_DEPTH

    @staticmethod
    def _wet_mse(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """MSE normalised over wet pixels only (GT depth > WET_THRESH).

        Avoids the dry-pixel dilution that causes MSE to underestimate peaks.
        pred, target : (B, T, 1, H, W)   mask : (B, 1, H, W)
        """
        T      = target.shape[1]
        mask4  = mask.unsqueeze(1)
        wet    = (target > FNOLitModel._WET_THRESH).float() * mask4
        n_wet  = wet.sum().clamp(min=1)
        return ((pred - target) ** 2 * wet).sum() / (n_wet * T)

    @staticmethod
    def _dry_mse(pred: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """MSE on dry pixels — penalises false-positive flooding.

        Only the positive part of pred is penalised (pred already ≥ 0 after
        ReLU/clamp, but we clamp here for safety).
        pred, target : (B, T, 1, H, W)   mask : (B, 1, H, W)
        """
        T      = target.shape[1]
        mask4  = mask.unsqueeze(1)
        dry    = (target <= FNOLitModel._WET_THRESH).float() * mask4
        n_dry  = dry.sum().clamp(min=1)
        return (pred.clamp(min=0) ** 2 * dry).sum() / (n_dry * T)

    @staticmethod
    def _extent_dice(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss on flood extent (wet/dry boundary).

        Uses a sigmoid around WET_THRESH to give a differentiable wet
        probability for pred while GT wet/dry is a hard mask.
        pred, target : (B, T, 1, H, W)   mask : (B, 1, H, W)
        """
        mask4   = mask.unsqueeze(1)
        gt_wet  = (target > FNOLitModel._WET_THRESH).float() * mask4
        pred_wet = torch.sigmoid(
            (pred - FNOLitModel._WET_THRESH) / FNOLitModel._EXTENT_SIGMA
        ) * mask4
        inter   = (pred_wet * gt_wet).sum()
        union   = pred_wet.sum() + gt_wet.sum()
        return 1.0 - (2.0 * inter + 1e-8) / (union + 1e-8)

    @staticmethod
    def _hybrid_loss(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """mse + 0.1*ssim_loss + 0.5*peak_aware_mse"""
        mse       = FNOLitModel._mse(pred, target, mask)
        ssim_loss = FNOLitModel._ssim_loss(pred, target, mask)
        pa_mse    = FNOLitModel._peak_aware_mse(pred, target, mask)
        return mse + 0.1 * ssim_loss + 0.5 * pa_mse

    @staticmethod
    def _hybrid_loss_2(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor, return_components: bool = False):
        """Flood-focused hybrid loss.

        wet_rmse       — depth accuracy where water exists (main term)
        0.5*dry_rmse   — suppress false-positive flooding
        0.3*extent     — inundation boundary accuracy (soft Dice)
        0.1*ssim       — spatial structure
        0.5*peak_rmse  — extra pressure on high-depth pixels

        wet/dry/peak terms are sqrt'd (RMSE, not MSE) and denormalised to
        metres. Left as raw normalised MSE they sit 4-5 orders of magnitude
        below extent/ssim (~O(0.1-1)) and are numerically invisible in the
        weighted sum — which silently defeats the depth-accuracy and
        false-positive terms this loss is supposed to enforce. Denormalising
        (rather than picking an arbitrary multiplier) puts them in the same
        physical currency (metres) that _WET_THRESH/_EXTENT_SIGMA are already
        defined in — still ~3-5x smaller than extent/ssim, but no longer
        negligible. Inspect the newly-logged val/hybrid_* components and
        retune these weights against real data if that residual gap matters.
        """
        eps = 1e-8
        to_metres = NORM_DEPTH / 1000.0
        wet_rmse  = torch.sqrt(FNOLitModel._wet_mse(pred, target, mask) + eps) * to_metres
        dry_rmse  = torch.sqrt(FNOLitModel._dry_mse(pred, target, mask) + eps) * to_metres
        extent    = FNOLitModel._extent_dice(pred, target, mask)
        ssim_loss = FNOLitModel._ssim_loss(pred, target, mask)
        pa_rmse   = torch.sqrt(FNOLitModel._peak_aware_mse(pred, target, mask) + eps) * to_metres

        total = wet_rmse + 0.5 * dry_rmse + 0.3 * extent + 0.1 * ssim_loss + 0.5 * pa_rmse
        if return_components:
            return total, dict(wet_rmse=wet_rmse, dry_rmse=dry_rmse, extent=extent,
                                ssim=ssim_loss, peak_rmse=pa_rmse)
        return total

    @staticmethod
    def _masked_mae(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).abs() * mask
        return diff.sum() / mask.sum().clamp(min=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, rain_future, depth_future, land = batch
        land      = land.float()
        N         = self.hparams.n_rollout_steps
        on_step   = (stage == 'train')

        static    = x[:, :6]    # (B, 6, P, P)  — static channels, constant across steps
        rain_prev = x[:, 6:7]   # Rain_{t-1}
        rain_curr = x[:, 7:8]   # Rain_t
        depth_in  = x[:, 8:9]   # h_t (initial condition)

        preds_list = []
        phys_loss  = torch.zeros(1, device=x.device, dtype=x.dtype).squeeze()

        for k in range(N):
            x_k  = torch.cat([static, rain_prev, rain_curr, depth_in], dim=1)
            pred = self(x_k)
            preds_list.append(pred)       # keep for loss before any detach

            # Physics: penalise depth increases when there is no rainfall.
            if self.hparams.lambda_phys > 0:
                no_rain  = (rain_curr < 1e-3).float()
                increase = torch.nn.functional.relu(pred - depth_in)
                l_phys   = (increase * no_rain * land).sum() / land.sum().clamp(min=1)
                phys_loss = phys_loss + l_phys
                self.log(f'{stage}/phys{k+1}', l_phys, on_epoch=True, on_step=False)

            # Advance state for next step
            rain_prev = rain_curr
            # Detach to avoid retaining all N computation graphs simultaneously.
            depth_in  = pred.detach() if self.hparams.truncate_bptt else pred
            if k < N - 1:
                rain_curr = rain_future[:, k]  # Rain_{t+k+1}

        # Stack rollout into (B, T, 1, H, W) and compute loss
        preds_all = torch.stack(preds_list, dim=1)            # (B, T, 1, H, W)
        if self.hparams.loss_fn == 'mse':
            total_loss = self._mse(preds_all, depth_future, land)
        elif self.hparams.loss_fn == 'hybrid':
            total_loss = self._hybrid_loss(preds_all, depth_future, land)
        elif self.hparams.loss_fn == 'hybrid_2':
            total_loss, components = self._hybrid_loss_2(
                preds_all, depth_future, land, return_components=True)
            for name, val in components.items():
                self.log(f'{stage}/hybrid_{name}', val, on_epoch=True, on_step=on_step)
        else:
            total_loss = self._peak_weighted_rmse(preds_all, depth_future, land)

        if self.hparams.lambda_phys > 0:
            total_loss = total_loss + self.hparams.lambda_phys * phys_loss

        # Per-step plain RMSE for diagnostic logging
        n_land = land.sum().clamp(min=1)
        for k in range(N):
            tgt    = depth_future[:, k]
            rmse_k = torch.sqrt(((preds_list[k] - tgt) ** 2 * land).sum() / n_land + 1e-8)
            self.log(f'{stage}/loss{k+1}', rmse_k,
                     prog_bar=(k == 0), on_epoch=True, on_step=on_step)
            self.log(f'{stage}/mae{k+1}',
                     self._masked_mae(preds_list[k], tgt, land), on_epoch=True, on_step=False)

        self.log(f'{stage}/loss', total_loss, prog_bar=True, on_epoch=True, on_step=on_step)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, 'test')

    def validation_step(self, batch, batch_idx):
        x, rain_future, depth_future, land = batch
        self._shared_step(batch, 'val')

        # Log 2-step prediction plots once per epoch (first batch only)
        if batch_idx == 0:
            with torch.no_grad():
                pred1 = self(x)
                x2    = torch.cat([x[:, :6], x[:, 7:8], rain_future[:, 0], pred1], dim=1)
                pred2 = self(x2)
            self._log_depth_figures(pred1, depth_future[:, 0],
                                    pred2, depth_future[:, 1], land)

    def _log_depth_figures(
        self,
        pred1: torch.Tensor, y1: torch.Tensor,
        pred2: torch.Tensor, y2: torch.Tensor,
        land:  torch.Tensor,
    ) -> None:
        """Log a 2×3 figure grid (step1 / step2) × (pred / gt / diff) to TensorBoard.
        Ocean pixels are masked to NaN so only land errors are visible."""
        from data_loader import NORM_DEPTH

        # Take first sample, denormalise to mm, mask ocean to NaN
        def _np(t: torch.Tensor) -> "np.ndarray":
            arr = t[0, 0].cpu().float().numpy() * NORM_DEPTH / 1000
            arr[land[0, 0].cpu().numpy() == 0] = np.nan
            return arr

        p1, g1 = _np(pred1), _np(y1)
        p2, g2 = _np(pred2), _np(y2)

        vmax = float(np.nanmax([g1, g2]))
        vmax = max(vmax, 1e-6)
        dmax = float(np.nanmax([np.abs(p1 - g1), np.abs(p2 - g2)]))
        dmax = max(dmax, 1e-6)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        titles = [
            ('Step 1 — Prediction', 'Step 1 — Ground Truth', 'Step 1 — Difference'),
            ('Step 2 — Prediction', 'Step 2 — Ground Truth', 'Step 2 — Difference'),
        ]
        rows = [(p1, g1), (p2, g2)]

        for row_i, ((pred, gt), row_titles) in enumerate(zip(rows, titles)):
            diff = pred - gt
            for col_i, (data, title) in enumerate(zip([pred, gt, diff], row_titles)):
                ax = axes[row_i, col_i]
                cmap  = 'RdBu_r' if col_i == 2 else 'Blues'
                vmin_ = -dmax   if col_i == 2 else 0.0
                vmax_ =  dmax   if col_i == 2 else vmax
                im = ax.imshow(data, cmap=cmap, vmin=vmin_, vmax=vmax_,
                               origin='upper', aspect='auto')
                ax.set_title(title, fontsize=9)
                ax.axis('off')
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label('m', fontsize=8)

        fig.suptitle(f'Epoch {self.current_epoch}', fontsize=11)
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
    p = argparse.ArgumentParser(description='Train TFNOFlood')

    # Data
    p.add_argument('--samples_dir', default='/home/hl1138/surrogate/data/samples',
                   help='Root directory containing all simulation sample subdirs')
    p.add_argument('--static_dir',  default='/home/hl1138/surrogate/data/parms_bands',
                   help='Shared static features directory (DEM, Manning, Slope, '
                        'Pervious, flow_accum.tif, soil_type.tif, etc.)')
    p.add_argument('--patch_size',  type=int,   default=512)
    p.add_argument('--stride',      type=int,   default=568)
    p.add_argument('--val_split',   type=float, default=0.15)
    p.add_argument('--test_split',  type=float, default=0.15)
    p.add_argument('--num_workers', type=int,   default=4)

    # Model
    p.add_argument('--in_channels',    type=int,   default=9,
                   help='6 static (DEM, Manning, Pervious, Slope, FlowAccLog, '
                        'SoilType) + 3 dynamic (Rain_t-1, Rain_t, Depth_t)')
    p.add_argument('--hidden_dim',     type=int,   default=64)
    p.add_argument('--n_layers',       type=int,   default=4)
    p.add_argument('--modes1',         type=int,   default=16)
    p.add_argument('--modes2',         type=int,   default=16)
    p.add_argument('--rank',           type=float, default=0.1)
    p.add_argument('--domain_padding', type=float, default=0.1)

    # Training
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_epochs',   type=int,   default=100)
    p.add_argument('--patience',     type=int,   default=5,
                   help='ReduceLROnPlateau patience')
    p.add_argument('--early_stop',   type=int,   default=15,
                   help='EarlyStopping patience (epochs); 0 to disable')
    p.add_argument('--n_steps',      type=int,   default=4,
                   help='Number of autoregressive rollout steps in training loss')
    p.add_argument('--full_bptt',    action='store_true', default=False,
                   help='Keep full computation graph across all rollout steps (high memory; default: truncated BPTT)')
    p.add_argument('--lambda_phys',  type=float, default=0.0,
                   help='Weight for no-rain drainage physics loss (0 = disabled)')
    p.add_argument('--loss',         default='mse',
                   choices=['mse', 'peak_weighted_rmse', 'hybrid', 'hybrid_2'],
                   help='Training loss function (default: mse)')

    # Infra
    p.add_argument('--log_dir',   default='logs')
    p.add_argument('--exp_name',  default=None,
                   help='TensorBoard experiment name. Auto-generated from settings if omitted.')
    p.add_argument('--ckpt_path', default=None,
                   help='Resume from checkpoint')
    p.add_argument('--precision', default='32',
                   choices=['32', '16-mixed', 'bf16-mixed'])
    p.add_argument('--devices',   type=int, default=1)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _auto_exp_name(args) -> str:
    return (
        f"tfno_extra_fullsamples"
        f"_h{args.hidden_dim}"
        f"_l{args.n_layers}"
        f"_m{args.modes1}x{args.modes2}"
        f"_r{args.rank}"
        f"_p{args.patch_size}"
        f"_n{args.n_steps}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_{args.loss}"
    )


def main():
    args = parse_args()

    if args.exp_name is None:
        args.exp_name = _auto_exp_name(args)
    print(f'Experiment    : {args.exp_name}')

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(
        samples_dir = args.samples_dir,
        static_dir  = args.static_dir,
        patch_size  = args.patch_size,
        stride      = args.stride,
        val_split   = args.val_split,
        test_split  = args.test_split,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        n_steps     = args.n_steps,
    )
    print(f'Train samples : {len(train_loader.dataset)}')
    print(f'Val   samples : {len(val_loader.dataset)}')
    print(f'Test  samples : {len(test_loader.dataset)}')

    # ── Model ─────────────────────────────────────────────────────────────
    model = TFNOFlood(
        in_channels    = args.in_channels,
        out_channels   = 1,
        hidden_dim     = args.hidden_dim,
        n_layers       = args.n_layers,
        modes1         = args.modes1,
        modes2         = args.modes2,
        rank           = args.rank,
        domain_padding = args.domain_padding,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model params  : {total_params:,}')

    lit = FNOLitModel(
        model            = model,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        patience         = args.patience,
        n_rollout_steps  = args.n_steps,
        lambda_phys      = args.lambda_phys,
        truncate_bptt    = not args.full_bptt,
        loss_fn          = args.loss,
        model_cfg        = dict(
            model_type     = 'fno',
            in_channels    = args.in_channels,
            hidden_dim     = args.hidden_dim,
            n_layers       = args.n_layers,
            modes1         = args.modes1,
            modes2         = args.modes2,
            rank           = args.rank,
            domain_padding = args.domain_padding,
            extra_static   = True,   # 6 static channels via data_loader_extra
        ),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor   = 'val/loss',
            mode      = 'min',
            filename  = 'best',
            save_last = True,
            verbose   = True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    if args.early_stop > 0:
        callbacks.append(
            EarlyStopping(monitor='val/loss', patience=args.early_stop, mode='min')
        )

    # ── Logger ────────────────────────────────────────────────────────────
    logger = TensorBoardLogger(
        save_dir = args.log_dir,
        name     = args.exp_name,
    )

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
