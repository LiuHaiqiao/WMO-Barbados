"""
gno_model.py — GNO (Graph Neural Operator) flood-depth predictor.

Adapted from /home/hl1138/AMG/models/grapher/ (Grapher architecture).

Speed improvements over the original Grapher:
  - Local edges  : precomputed 4-connected grid (no per-step FPS + knn_interpolate)
  - Global edges : precomputed FPS + k-NN on grid positions once at __init__
  - PhysicsBlock : fully vectorised (B,N,D) batch ops — no Python for-loop over samples

Architecture (per layer):
  1. PhysicsGraphBlock  : pool to num_phys latent tokens → batched multi-head
                          self-attention → unpool (global context, no edges needed)
  2. Local GATv2        : 4-connected grid neighbours (spatial locality)
  3. Global GATv2       : FPS-sampled k-NN (long-range coverage)
  4. MLP + residual

Drop-in compatible with TFNOFlood / UNetFlood / GNNFlood:
  (B, C, H, W) → (B, 1, H, W)
"""

import einops
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import fps
from torch_geometric.nn import knn_graph


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #

class _MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, act='relu'):
        super().__init__()
        _acts = {
            'relu': nn.ReLU(), 'gelu': nn.GELU(),
            'elu':  nn.ELU(),  'tanh': nn.Tanh(),
        }
        act_fn = _acts[act]
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_layers):
            layers += [act_fn, nn.Linear(hidden_size, hidden_size)]
        layers += [act_fn, nn.Linear(hidden_size, output_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Grid helpers
# --------------------------------------------------------------------------- #

def _build_grid_pos(H: int, W: int) -> torch.Tensor:
    """Normalised (row, col) positions for an H×W grid → (H*W, 2)."""
    r = torch.linspace(0, 1, H)
    c = torch.linspace(0, 1, W)
    rr, cc = torch.meshgrid(r, c, indexing='ij')
    return torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=1)


def _build_grid_edges(H: int, W: int) -> torch.Tensor:
    """4-connected edge_index for H×W grid → (2, E)."""
    idx = torch.arange(H * W).reshape(H, W)
    src, dst = [], []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r_src = torch.arange(max(0, -dr), H - max(0, dr))
        c_src = torch.arange(max(0, -dc), W - max(0, dc))
        rr, cc = torch.meshgrid(r_src, c_src, indexing='ij')
        src.append(idx[rr, cc].reshape(-1))
        dst.append(idx[rr + dr, cc + dc].reshape(-1))
    return torch.stack([torch.cat(src), torch.cat(dst)])


def _precompute_global_edges(grid_pos: torch.Tensor,
                              ratio: float, k: int) -> torch.Tensor:
    """
    FPS-sample grid positions, build k-NN, return full-resolution edge_index (2, E).
    Called once at __init__ — result is stored as a buffer.
    """
    sampled_idx = fps(grid_pos, ratio=ratio)               # (M,) indices into N
    sampled_ei  = knn_graph(grid_pos[sampled_idx], k=k, loop=False)  # (2, E')
    return torch.stack([sampled_idx[sampled_ei[0]],
                         sampled_idx[sampled_ei[1]]])       # (2, E')


def _tile_edges(ei: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """Offset a single-sample edge_index for a batch of B samples."""
    offsets = torch.arange(B, device=ei.device) * N        # (B,)
    return (ei.unsqueeze(0) + offsets[:, None, None]
            ).permute(1, 0, 2).reshape(2, -1)              # (2, B*E)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class _AttentionGraphBlock(nn.Module):
    """GATv2 + FFN with pre-norm residual."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.conv  = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads,
                               concat=False, negative_slope=0.2, dropout=0.0)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn   = _MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=0)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.conv(x, edge_index))
        x = self.norm2(x + self.ffn(x))
        return x


class _PhysicsGraphBlock(nn.Module):
    """
    Fully-vectorised physics token block (no Python loop over batch samples).

    Each sample is pooled into num_phys latent tokens via soft assignment.
    Multi-head self-attention is applied across those tokens, then they are
    scattered back to node features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_phys: int = 16):
        super().__init__()
        D, H, M = hidden_dim, num_heads, num_phys
        self.D, self.H, self.M = D, H, M
        self.scale   = D ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        # temperature: (1, H, 1, 1) for broadcasting with (B, H, N, M)
        self.temperature = nn.Parameter(torch.ones(1, H, 1, 1) * 0.5)

        self.l_in    = nn.Linear(D, D * H)
        self.l_token = nn.Linear(D, D * H)
        self.l_phy   = nn.Linear(D, M)
        nn.init.orthogonal_(self.l_phy.weight)

        self.q     = nn.Linear(D, D, bias=False)
        self.k     = nn.Linear(D, D, bias=False)
        self.v     = nn.Linear(D, D, bias=False)
        self.l_out = nn.Linear(D * H, D)

        self.norm1 = nn.LayerNorm(D)
        self.ffn   = _MLP(D, D, D, num_layers=0)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor, B: int) -> torch.Tensor:
        # x: (B*N, D)  →  reshape to (B, N, D) for batch ops
        BN, D = x.shape
        N = BN // B
        x3 = x.reshape(B, N, D)
        shortcut = x

        # ── Pool to physics tokens ────────────────────────────────────────
        # (B, N, H*D) → (B, H, N, D)
        phy_x = einops.rearrange(self.l_in(x3), 'b n (h c) -> b h n c', h=self.H)
        # soft assignment weights (B, H, N, M)
        w = self.softmax(self.l_phy(phy_x) / self.temperature)

        # weighted mean → physics tokens (B, H, M, D)
        tok = einops.rearrange(self.l_token(x3), 'b n (h c) -> b h n c', h=self.H)
        norm  = w.sum(dim=2, keepdim=True) + 1e-5          # (B, H, 1, M)
        token = torch.einsum('bhnc,bhnm->bhmc', tok, w) / norm.squeeze(2).unsqueeze(-1)

        # ── Self-attention across tokens ──────────────────────────────────
        q, k, v = self.q(token), self.k(token), self.v(token)
        attn  = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        token = torch.matmul(attn, v)                       # (B, H, M, D)

        # ── Unpool back to nodes ──────────────────────────────────────────
        out = torch.einsum('bhmc,bhnm->bhnc', token, w)    # (B, H, N, D)
        out = einops.rearrange(out, 'b h n c -> b n (h c)')
        out = self.l_out(out).reshape(BN, D)                # (B*N, D)

        x = self.norm1(out + shortcut)
        x = self.norm2(x  + self.ffn(x))
        return x


class _MultiscaleGraphBlock(nn.Module):
    """Physics tokens + local GATv2 + global GATv2 + MLP residual.

    Edge indices are passed in (precomputed outside) — no dynamic graph
    construction at forward time.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_phys: int = 16):
        super().__init__()
        self.phy_aggr    = _PhysicsGraphBlock(hidden_dim, num_heads, num_phys)
        self.local_aggr  = _AttentionGraphBlock(hidden_dim, num_heads)
        self.global_aggr = _AttentionGraphBlock(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn  = _MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=0)

    def forward(self, x: torch.Tensor, local_ei: torch.Tensor,
                global_ei: torch.Tensor, B: int) -> torch.Tensor:
        x_in = x
        x = self.phy_aggr(x, B)
        x = self.local_aggr(x, local_ei)
        x = self.global_aggr(x, global_ei)
        x = self.ffn(self.norm(x + x_in))
        return x


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #

class GNOFlood(nn.Module):
    """
    GNO flood-depth predictor (Grapher-style multiscale graph neural operator).

    Parameters
    ----------
    in_channels  : C (default 7)
    out_channels : 1
    hidden_dim   : node feature dimension throughout GNO layers
    n_layers     : number of MultiscaleGraphBlock layers
    patch_size   : H = W of expected input patch (precomputes edges once)
    num_heads    : GATv2 / physics-block attention heads
    num_phys     : number of latent physics tokens per sample
    global_ratio : FPS ratio for global node sampling (fraction of H*W)
    global_k     : k-NN degree for global graph
    """

    def __init__(
        self,
        in_channels:  int   = 7,
        out_channels: int   = 1,
        hidden_dim:   int   = 128,
        n_layers:     int   = 4,
        patch_size:   int   = 384,
        num_heads:    int   = 4,
        num_phys:     int   = 16,
        global_ratio: float = 0.1,
        global_k:     int   = 8,
    ):
        super().__init__()
        self.patch_size = patch_size

        # ── Precompute edges (once, CPU) ───────────────────────────────────
        grid_pos  = _build_grid_pos(patch_size, patch_size)   # (N, 2)
        local_ei  = _build_grid_edges(patch_size, patch_size) # (2, E_local)
        global_ei = _precompute_global_edges(grid_pos, global_ratio, global_k)

        self.register_buffer('_grid_pos',  grid_pos)
        self.register_buffer('_local_ei',  local_ei)
        self.register_buffer('_global_ei', global_ei)

        # ── Network ────────────────────────────────────────────────────────
        self.encoder = _MLP(in_channels + 2, hidden_dim * 2, hidden_dim,
                            num_layers=0)
        self.blocks  = nn.ModuleList([
            _MultiscaleGraphBlock(hidden_dim, num_heads, num_phys)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(hidden_dim)
        self.decoder = _MLP(hidden_dim, hidden_dim, out_channels, num_layers=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W)  →  (B, 1, H, W)"""
        B, C, H, W = x.shape
        N = H * W

        # Node features (B*N, C) and positions (B*N, 2)
        h   = x.permute(0, 2, 3, 1).reshape(B * N, C)
        pos = self._grid_pos.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 2)

        # Tile precomputed edges for this batch
        local_ei  = _tile_edges(self._local_ei,  B, N)
        global_ei = _tile_edges(self._global_ei, B, N)

        h    = self.encoder(torch.cat([h, pos], dim=-1))
        h_in = h

        for block in self.blocks:
            h = block(h, local_ei, global_ei, B)

        out = self.decoder(self.norm(h + h_in))             # (B*N, out_channels)
        return out.reshape(B, H, W, -1).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import time

    P = 64
    model = GNOFlood(
        in_channels  = 7,
        hidden_dim   = 64,
        n_layers     = 2,
        patch_size   = P,
        num_heads    = 4,
        num_phys     = 8,
        global_ratio = 0.1,
        global_k     = 8,
    ).cpu()

    x = torch.randn(2, 7, P, P)
    t0 = time.time()
    y = model(x)
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y.shape)}')
    print(f'Params : {sum(p.numel() for p in model.parameters()):,}')
    print(f'Forward: {(time.time() - t0) * 1000:.1f} ms  (cpu)')

    N = P * P
    n_local  = model._local_ei.shape[1]
    n_global = model._global_ei.shape[1]
    print(f'Local edges  : {n_local}  ({n_local / N:.1f} per node)')
    print(f'Global edges : {n_global}  ({n_global / N:.1f} per node, '
          f'{round(0.1 * N)} sampled nodes)')
