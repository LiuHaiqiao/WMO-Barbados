"""
cnn_model.py — U-Net baseline for flood depth prediction.

Drop-in compatible with TFNOFlood: accepts (B, C, H, W) and returns (B, 1, H, W).

Architecture: 4-level encoder-decoder with skip connections.
    Encoder: Conv(3×3) × 2 + BN + ReLU, then MaxPool(2×2)
    Decoder: BilinearUpsample + Conv(3×3) × 2 + BN + ReLU
    Output : Conv(1×1) → depth

Default capacity (base_channels=64, depth=4) gives ~31M parameters,
comparable to TFNOFlood at equivalent hidden_dim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetFlood(nn.Module):
    """
    U-Net flood depth predictor.

    Parameters
    ----------
    in_channels   : input channels (7 = 4 static + 3 dynamic)
    out_channels  : output channels (1 = water depth)
    base_channels : feature maps at the first encoder level (doubles each level)
    depth         : number of encoder/decoder levels (default 4)
    """

    def __init__(
        self,
        in_channels:   int = 7,
        out_channels:  int = 1,
        base_channels: int = 64,
        depth:         int = 4,
    ):
        super().__init__()
        ch = [base_channels * (2 ** i) for i in range(depth)]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for c in ch:
            self.encoders.append(_ConvBlock(prev, c))
            self.pools.append(nn.MaxPool2d(2))
            prev = c

        # Bottleneck
        self.bottleneck = _ConvBlock(ch[-1], ch[-1] * 2)

        # Decoder
        self.decoders = nn.ModuleList()
        prev = ch[-1] * 2
        for c in reversed(ch):
            self.decoders.append(_UpBlock(prev, c, c))
            prev = c

        self.head = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    device = 'cpu'
    P = 384

    model = UNetFlood(
        in_channels=7,
        base_channels=64,
        depth=4,
    ).to(device)

    x = torch.randn(2, 7, P, P, device=device)
    y = model(x)

    total = sum(p.numel() for p in model.parameters())
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y.shape)}')
    print(f'Params : {total:,}')
    print(f'Device : {device}')
