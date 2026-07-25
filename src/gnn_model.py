"""
gnn_model.py — GNN flood-depth predictor.

Builds a k=4 nearest-neighbour graph on the regular pixel grid (giving the
4-connected cardinal neighbours N/S/E/W) and applies stacked message-passing
layers with DEM-difference edge features.

Drop-in compatible with TFNOFlood / UNetFlood: (B, C, H, W) → (B, 1, H, W).

Graph construction
------------------
- Nodes    : one per pixel, feature vector = input channels at that pixel
- Edges    : 4-connected neighbours (precomputed once for a given H×W)
- Edge attr: Δdem = dem[j] - dem[i]  (elevation change src→dst, shape (E,1))
             Computed dynamically from channel 0 of the input (normalised DEM).
- Batch    : edge indices are offset by b·H·W for each sample in the batch
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# --------------------------------------------------------------------------- #
# Graph construction helpers
# --------------------------------------------------------------------------- #

def build_grid_edges(H: int, W: int) -> torch.Tensor:
    """
    Precompute edge_index for a single H×W grid with k=4 nearest neighbours.

    Returns
    -------
    edge_index : (2, E) long — source / target node indices
    """
    node_idx = torch.arange(H * W).reshape(H, W)
    src, dst = [], []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:   # N S W E
        r_src = torch.arange(max(0, -dr), H - max(0, dr))
        c_src = torch.arange(max(0, -dc), W - max(0, dc))
        rr, cc = torch.meshgrid(r_src, c_src, indexing='ij')
        src.append(node_idx[rr, cc].reshape(-1))
        dst.append(node_idx[rr + dr, cc + dc].reshape(-1))
    return torch.stack([torch.cat(src), torch.cat(dst)], dim=0)


# --------------------------------------------------------------------------- #
# Message-passing layer
# --------------------------------------------------------------------------- #

class _GNNLayer(MessagePassing):
    """
    h_i' = MLP_update( h_i || mean_j( MLP_msg( h_j || edge_attr_ij ) ) )
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 1):
        super().__init__(aggr='mean')
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        agg = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return self.norm(h + self.upd_mlp(torch.cat([h, agg], dim=-1)))

    def message(self, h_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.msg_mlp(torch.cat([h_j, edge_attr], dim=-1))


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #

class GNNFlood(nn.Module):
    """
    GNN flood-depth predictor.

    Parameters
    ----------
    in_channels  : C (default 7)
    out_channels : 1
    hidden_dim   : node feature dimension in GNN layers
    n_layers     : number of message-passing layers
    patch_size   : H=W of the expected input patch (used to precompute edges)
    """

    def __init__(
        self,
        in_channels:  int = 7,
        out_channels: int = 1,
        hidden_dim:   int = 128,
        n_layers:     int = 6,
        patch_size:   int = 384,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.patch_size  = patch_size

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [_GNNLayer(hidden_dim) for _ in range(n_layers)]
        )
        # self.decoder = nn.Linear(hidden_dim, out_channels) # gnn one
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_channels),
            nn.Softplus(),
        )

        # Precompute topology for the expected patch size (no edge_attr — computed per sample)
        self.register_buffer('_edge_index', build_grid_edges(patch_size, patch_size))  # (2, E)

    def _batch_edges(self, B: int, N: int) -> torch.Tensor:
        """Tile edge_index across the batch by offsetting node indices."""
        offsets = torch.arange(B, device=self._edge_index.device) * N   # (B,)
        ei = self._edge_index.unsqueeze(0) + offsets[:, None, None]      # (B, 2, E)
        return ei.permute(1, 0, 2).reshape(2, -1)                        # (2, B*E)

    def _dem_edge_attr(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute DEM difference for every edge: Δdem = dem[dst] - dem[src].

        x          : (B, C, H, W) — channel 0 is normalised DEM
        edge_index : (2, B*E)
        Returns    : (B*E, 1)
        """
        dem_flat = x[:, 0, :, :].reshape(-1)   # (B*N,)
        delta = dem_flat[edge_index[1]] - dem_flat[edge_index[0]]
        return delta.unsqueeze(1)               # (B*E, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W)  →  (B, 1, H, W)"""
        B, C, H, W = x.shape
        assert H == self.patch_size and W == self.patch_size, (
            f"GNNFlood expects {self.patch_size}×{self.patch_size} input, got {H}×{W}"
        )
        N = H * W

        if not hasattr(self, '_cached_ei') or self._cached_B != B:
            self._cached_ei = self._batch_edges(B, N)
            self._cached_B  = B
        edge_index = self._cached_ei

        edge_attr  = self._dem_edge_attr(x, edge_index)    # (B*E, 1)

        # Flatten to node features: (B*N, C)
        h = x.permute(0, 2, 3, 1).reshape(B * N, C)
        h = self.encoder(h)                                 # (B*N, hidden)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        out = self.decoder(h)                               # (B*N, out_channels)
        return out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_channels, H, W)


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import time

    device = 'cpu'
    P = 64   # use small patch for quick test

    # Rebuild edges for small patch
    model = GNNFlood(
        in_channels = 7,
        hidden_dim  = 64,
        n_layers    = 4,
        patch_size  = P,
    ).to(device)

    x = torch.randn(2, 7, P, P, device=device)
    t0 = time.time()
    y = model(x)
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y.shape)}')
    print(f'Params : {sum(p.numel() for p in model.parameters()):,}')
    print(f'Forward: {(time.time()-t0)*1000:.1f} ms  (device={device})')

    # Verify edge count: 4-connected grid has 2*(H*(W-1) + (H-1)*W) directed edges
    H = W = P
    expected_edges = 2 * (H * (W - 1) + (H - 1) * W)
    ei = build_grid_edges(H, W)
    print(f'Edges  : {ei.shape[1]}  (expected {expected_edges})')
