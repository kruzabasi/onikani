# models/psformer/blocks.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSBlock(nn.Module):
    """
    Parameter-shared block described in the paper:
    three linear layers with residual connections, applied across the last dimension (N).
    Accepts input shape (batch, C, N) and returns same shape.
    """
    def __init__(self, N: int):
        super().__init__()
        if N <= 0:
            raise ValueError("N (patch count dimension) must be > 0")
        self.fc1 = nn.Linear(N, N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, C, N) or (C, N)
        returns: same shape (batch, C, N) or (C, N)
        """
        squeeze_back = False
        if x.dim() == 2:
            # treat as single batch
            x = x.unsqueeze(0)
            squeeze_back = True
        if x.dim() != 3:
            raise ValueError("Input tensor must have shape (batch, C, N) or (C, N)")
        # apply fc over last dim
        out = F.gelu(self.fc1(x))        # (batch, C, N)
        out = self.fc2(out) + x         # residual connection over last dim
        out = self.fc3(out)
        if squeeze_back:
            out = out.squeeze(0)
        return out


class SegAtt(nn.Module):
    """
    Spatial-Temporal Segment Attention
    - Uses a shared PSBlock to produce Q,K,V
    - Computes attention across the segment dimension C:
      Q,K,V: (batch, C, N) -> att = softmax( Q @ K^T / sqrt(d_k) )  -> (batch, C, C)
      out = att @ V -> (batch, C, N)
    - Adds residual and passes through PSBlock again (optional final transform).
    """
    def __init__(self, ps_block: PSBlock, d_k: int = None):
        """
        ps_block: an instance of PSBlock; will be reused (parameter sharing).
        d_k: dimension used for scaling (defaults to last dim N)
        """
        super().__init__()
        if not isinstance(ps_block, PSBlock):
            raise TypeError("ps_block must be an instance of PSBlock")
        self.ps = ps_block
        self.d_k = d_k

    def forward(self, xin: torch.Tensor) -> torch.Tensor:
        """
        xin: (batch, C, N)
        returns: (batch, C, N)
        """
        if xin.dim() == 2:
            xin = xin.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        if xin.dim() != 3:
            raise ValueError("xin must have shape (batch, C, N) or (C, N)")

        q = self.ps(xin)  # (batch, C, N)
        k = self.ps(xin)
        v = self.ps(xin)

        dk = self.d_k if (self.d_k is not None) else xin.size(-1)
        scale = math.sqrt(dk) if dk > 0 else 1.0

        # attention across C dimension: matmul(q, k.transpose(-2,-1)) -> (batch, C, C)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        att = torch.softmax(scores, dim=-1)  # along the last dim (C)
        out = torch.matmul(att, v)           # (batch, C, N)

        # residual and final transform via shared PSBlock
        out = out + xin
        out = self.ps(out)

        if squeeze_back:
            out = out.squeeze(0)
        return out
