"""
train_gno.py — PyTorch Lightning training script for GNOFlood.

Shares the same LightningModule, masked loss, 2-step rollout, and TensorBoard
plots as train_fno.py — only the model and CLI args differ.

Usage
-----
# single GPU
python train_gno.py --data_dirs /home/hl1138/TFNO/data

# multiple simulation dirs
python train_gno.py --data_dirs /path/sim1 /path/sim2 --batch_size 4

# resume from checkpoint
python train_gno.py --data_dirs /home/hl1138/TFNO/data --ckpt_path logs/gno/version_0/checkpoints/best.ckpt
"""

import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from gno_model import GNOFlood
from data_loader import build_loaders
from train_fno import FNOLitModel


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='Train GNOFlood')

    # Data
    p.add_argument('--samples_dir', default='/home/hl1138/TFNO/data/samples',
                   help='Root directory containing all simulation sample subdirs')
    p.add_argument('--static_dir',  default='/home/hl1138/TFNO/data/parms_bands',
                   help='Shared static features directory (DEM, Manning, etc.)')
    p.add_argument('--patch_size',  type=int,   default=512)
    p.add_argument('--stride',      type=int,   default=568)
    p.add_argument('--val_split',   type=float, default=0.15)
    p.add_argument('--test_split',  type=float, default=0.15)
    p.add_argument('--num_workers', type=int,   default=4)

    # Model
    p.add_argument('--in_channels',  type=int,   default=6)
    p.add_argument('--hidden_dim',   type=int,   default=64,
                   help='Node feature dimension throughout GNO layers')
    p.add_argument('--n_layers',     type=int,   default=2,
                   help='Number of MultiscaleGraphBlock layers')
    p.add_argument('--num_heads',    type=int,   default=4,
                   help='GATv2 attention heads')
    p.add_argument('--num_phys',     type=int,   default=8,
                   help='Number of latent physics tokens per sample')
    p.add_argument('--global_ratio', type=float, default=0.1,
                   help='FPS ratio for global node sampling (precomputed at init)')
    p.add_argument('--global_k',     type=int,   default=8,
                   help='k-NN degree for global graph')

    # Training
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_epochs',   type=int,   default=100)
    p.add_argument('--patience',     type=int,   default=5,
                   help='ReduceLROnPlateau patience')
    p.add_argument('--early_stop',   type=int,   default=15,
                   help='EarlyStopping patience (epochs); 0 to disable')
    p.add_argument('--n_steps',      type=int,   default=2,
                   help='Number of autoregressive rollout steps in training loss')
    p.add_argument('--lambda_phys',  type=float, default=0.0,
                   help='Weight for no-runoff drainage physics loss (0 = disabled)')
    p.add_argument('--loss',         default='mse',
                   choices=['mse', 'peak_weighted_rmse', 'hybrid', 'hybrid_2'],
                   help='Training loss function (default: mse)')

    # Infra
    p.add_argument('--log_dir',   default='logs')
    p.add_argument('--exp_name',  default=None,
                   help='TensorBoard experiment name. Auto-generated if omitted.')
    p.add_argument('--ckpt_path', default=None,
                   help='Resume from checkpoint')
    p.add_argument('--precision', default='32',
                   choices=['32', '16-mixed', 'bf16-mixed'])
    p.add_argument('--devices',   type=int, default=1)

    return p.parse_args()


def _auto_exp_name(args) -> str:
    return (
        f"gno_runoff_only"
        f"_h{args.hidden_dim}"
        f"_l{args.n_layers}"
        f"_nh{args.num_heads}"
        f"_np{args.num_phys}"
        f"_p{args.patch_size}"
        f"_n{args.n_steps}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_{args.loss}"
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
    model = GNOFlood(
        in_channels  = args.in_channels,
        out_channels = 1,
        hidden_dim   = args.hidden_dim,
        n_layers     = args.n_layers,
        patch_size   = args.patch_size,
        num_heads    = args.num_heads,
        num_phys     = args.num_phys,
        global_ratio = args.global_ratio,
        global_k     = args.global_k,
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
        model_cfg        = dict(
            model_type   = 'gno',
            in_channels  = args.in_channels,
            hidden_dim   = args.hidden_dim,
            n_layers     = args.n_layers,
            patch_size   = args.patch_size,
            num_heads    = args.num_heads,
            num_phys     = args.num_phys,
            global_ratio = args.global_ratio,
            global_k     = args.global_k,
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
