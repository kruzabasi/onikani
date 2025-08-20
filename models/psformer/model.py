# models/psformer/model.py
import torch
import torch.nn as nn
from .blocks import PSBlock, SegAtt

class PSFormer(nn.Module):
    """
    PSFormer with decoder/inverse transform:
      - input: (batch, C, N)
      - encoder: repeated SegAtt layers (sharing PSBlock)
      - inverse patch: (batch, C, N) -> (batch, M, L)
      - linear mapping W_F: L -> F (applied per variable) to produce (batch, M, F)
    """
    def __init__(self, M: int, L: int, P: int, horizon: int = 1, n_layers: int = 2):
        super().__init__()
        assert L % P == 0, "L must be divisible by P"
        self.M = M
        self.L = L
        self.P = P
        self.F = horizon
        self.N = L // P
        self.C = M * P
        self.n_layers = n_layers

        # shared PSBlock across layers
        self.ps_block = PSBlock(self.N)
        self.layers = nn.ModuleList([SegAtt(self.ps_block, d_k=self.N) for _ in range(n_layers)])

        # Decoder linear mapping W_F: map L -> F for each channel.
        # Implemented as a single Linear applied to flattened (batch*M, L)
        self.W_F = nn.Linear(self.L, self.F)

    def _segments_to_series(self, seg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert seg_tensor shape (batch, C, N) -> (batch, M, L)
        where C = M * P and L = N * P
        """
        if seg_tensor.dim() != 3:
            raise ValueError("_segments_to_series expects shape (batch, C, N)")
        batch = seg_tensor.size(0)
        # seg_tensor: (batch, C, N) with C = M*P
        # reshape to (batch, M, P, N)
        # ensure contiguous before view
        x = seg_tensor.contiguous().view(batch, self.M, self.P, self.N)
        # permute to (batch, M, N, P) then reshape to (batch, M, L)
        x = x.permute(0, 1, 3, 2).contiguous().view(batch, self.M, self.L)
        return x

    def forward(self, x):
        """
        x: Tensor shape (batch, C, N)
        returns: Tensor shape (batch, M, F)
        """
        if x.dim() != 3:
            raise ValueError("Input must be (batch, C, N)")
        out = x
        for layer in self.layers:
            out = layer(out)  # (batch, C, N)
        # inverse to series shape
        series = self._segments_to_series(out)  # (batch, M, L)
        # map time dimension L -> F using linear W_F applied per channel
        b, m, l = series.shape
        # flatten channels into batch dimension for a single linear transform
        series_flat = series.view(b * m, l)  # (b*M, L)
        pred_flat = self.W_F(series_flat)     # (b*M, F)
        pred = pred_flat.view(b, m, self.F)   # (b, M, F)
        return pred
