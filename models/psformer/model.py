# models/psformer/model.py
import torch
import torch.nn as nn
from .blocks import PSBlock, SegAtt

class PSFormer(nn.Module):
    """
    Minimal PSformer that:
      - takes input (batch, C, N)
      - passes through n_layers of SegAtt (sharing a PSBlock instance)
      - flattens and maps to output dimension M * F
    It's purposely simple to match the needs of unit tests (overfitting toy data).
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
        # build layers using the same ps_block (shared parameters)
        self.layers = nn.ModuleList([SegAtt(self.ps_block, d_k=self.N) for _ in range(n_layers)])

        # after encoder -> flatten and a simple MLP to map to M*F
        flattened_dim = self.C * self.N
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, max(16, flattened_dim // 8)),
            nn.ReLU(),
            nn.Linear(max(16, flattened_dim // 8), M * self.F)
        )

    def forward(self, x):
        """
        x: Tensor shape (batch, C, N)
        returns: Tensor shape (batch, M, F)
        """
        if x.dim() != 3:
            raise ValueError("Input must be (batch, C, N)")
        out = x
        for layer in self.layers:
            out = layer(out)
        # head expects (batch, C, N)
        y = self.head(out)  # (batch, M*F)
        y = y.view(x.size(0), self.M, self.F)
        return y
