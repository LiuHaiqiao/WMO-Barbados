import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import BatchedDCRNN


# --------------------------------------------------------------------------- #
# DCRNN cell with custom initial hidden state
# --------------------------------------------------------------------------- #

class _DCRNNWithInit(BatchedDCRNN):
    """
    BatchedDCRNN extended to accept an optional initial hidden state h0.
    Only forward() is overridden; all gate layers are inherited unchanged.
    """

    def forward(
        self,
        X:           torch.Tensor,          # (B, T, N, in_channels)
        edge_index:  torch.Tensor,          # (2, E)
        edge_weight: torch.Tensor,          # (E,)
        h0:          torch.Tensor | None = None,  # (B, N, out_channels)
    ) -> torch.Tensor:                      # (B, T, N, out_channels)

        B, T, N, F = X.size()

        hidden = (h0 if h0 is not None
                  else torch.zeros(B, N, self.out_channels, device=X.device,
                                   dtype=X.dtype))

        # Rebuild expanded edge index when graph or batch size changes
        if (self._cached_edge_index is None
                or self._cached_batch_size != B
                or not torch.equal(self._cached_edge_index, edge_index)
                or not torch.equal(self._cached_edge_weight, edge_weight)):
            self._cached_batch_size           = B
            self._cached_edge_index           = edge_index
            self._cached_edge_weight          = edge_weight
            self._cached_expanded_edge_index  = self._replicate_edge_index(edge_index, B, N)
            self._cached_expanded_edge_weight = edge_weight.repeat(B)
            self._cached_idx = False
        else:
            self._cached_idx = True

        ei = self._cached_expanded_edge_index
        ew = self._cached_expanded_edge_weight

        outputs = []
        for t in range(T):
            x_t = X[:, t].reshape(B * N, F)
            H   = hidden.reshape(B * N, self.out_channels)
            Z   = self._calculate_update_gate(x_t, ei, ew, H, self._cached_idx)
            R   = self._calculate_reset_gate(x_t, ei, ew, H, self._cached_idx)
            H_c = self._calculate_candidate_state(x_t, ei, ew, H, R, self._cached_idx)
            H   = self._calculate_hidden_state(Z, H, H_c)
            hidden = H.reshape(B, N, self.out_channels)
            outputs.append(hidden)

        return torch.stack(outputs, dim=1)   # (B, T, N, out_channels)


# --------------------------------------------------------------------------- #
# Flood DCRNN
# --------------------------------------------------------------------------- #

class FloodDCRNN(nn.Module):
    """
    Diffusion Convolutional RNN for water depth prediction.

    Parameters
    ----------
    d_init     : feature dim of the initial-condition input  (D1)
    d_dyn      : feature dim of the per-step dynamic input   (D2)
    hidden_dim : DCRNN hidden channels  (default 64)
    n_layers   : number of stacked DCRNN layers  (default 2)
    K          : diffusion filter order  (default 3)
    dropout    : dropout probability between layers  (default 0.1)

    Inputs
    ------
    init_feat  : (B, N, D1)    current water depth + static features
    dyn_feat   : (B, T, N, D2) per-step rainfall + static features
    edge_index : (2, E)
    edge_weight: (E,)

    Output
    ------
    depth : (B, T, N, 1)  predicted water depth (clamped ≥ 0)

    Architecture
    ------------
    1. init_encoder  MLP  D1 → hidden_dim   produces h0 for layer-0
    2. input_proj    Linear D2 → hidden_dim  projects each dynamic step
    3. DCRNN stack   n_layers × _DCRNNWithInit
                     layer 0 starts from h0; deeper layers start from zeros
    4. output_proj   MLP  hidden_dim → hidden_dim//2 → 1
    """

    def __init__(
        self,
        d_init:     int,
        d_dyn:      int,
        hidden_dim: int   = 64,
        n_layers:   int   = 2,
        K:          int   = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Encode initial condition → h0
        self.init_encoder = nn.Sequential(
            nn.Linear(d_init, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project dynamic features to DCRNN width
        self.input_proj = nn.Linear(d_dyn, hidden_dim)

        # Stacked DCRNN cells
        self.dcrnn_layers = nn.ModuleList([
            _DCRNNWithInit(in_channels=hidden_dim, out_channels=hidden_dim, K=K)
            for _ in range(n_layers)
        ])

        self.drop = nn.Dropout(dropout)

        # Depth output head
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        init_feat:   torch.Tensor,   # (B, N, D1)
        dyn_feat:    torch.Tensor,   # (B, T, N, D2)
        edge_index:  torch.Tensor,   # (2, E)
        edge_weight: torch.Tensor,   # (E,)
    ) -> torch.Tensor:               # (B, T, N, 1)

        # h0: encode the initial water depth + static snapshot
        h0 = self.init_encoder(init_feat)   # (B, N, hidden_dim)

        # Project dynamic inputs
        x = self.drop(self.input_proj(dyn_feat))   # (B, T, N, hidden_dim)

        # DCRNN layers; only layer 0 receives h0
        for i, layer in enumerate(self.dcrnn_layers):
            x = layer(x, edge_index, edge_weight, h0=(h0 if i == 0 else None))
            x = self.drop(x)

        # Project to depth and enforce non-negativity
        return self.output_proj(x).clamp(min=0.0)   # (B, T, N, 1)
