# tests/test_psblock.py
import torch
from models.psformer.blocks import PSBlock, SegAtt

def test_psblock_forward_shape():
    batch, C, N = 2, 6, 8
    x = torch.randn(batch, C, N)
    ps = PSBlock(N)
    out = ps(x)
    assert out.shape == (batch, C, N)

def test_segatt_forward_shape():
    batch, C, N = 2, 6, 8
    x = torch.randn(batch, C, N)
    ps = PSBlock(N)
    seg = SegAtt(ps, d_k=N)
    out = seg(x)
    assert out.shape == (batch, C, N)
