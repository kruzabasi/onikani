# tests/test_model_overfit.py
import numpy as np
import torch
import random
from models.psformer.model import PSFormer
from training.datasets import SlidingWindowDataset
from training.trainer import Trainer

def seed_everything(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def test_model_forward_shape():
    seed_everything(0)
    M = 2
    L = 16
    P = 4
    F = 2
    batch = 4
    C = M * P
    N = L // P
    x = torch.randn(batch, C, N)
    model = PSFormer(M=M, L=L, P=P, horizon=F, n_layers=1)
    out = model(x)
    assert out.shape == (batch, M, F)

def test_model_can_overfit_small_batch():
    seed_everything(42)
    # create tiny synthetic dataset where model can memorize
    M = 2
    T = 80
    L = 16
    P = 4
    F = 1
    t = np.arange(T)
    X = np.vstack([
        (np.sin(0.1 * t) + 0.01 * np.random.randn(T)),
        (np.cos(0.08 * t) + 0.01 * np.random.randn(T))
    ]).astype(float)
    ds = SlidingWindowDataset(X, input_length=L, patch_size=P, horizon=F)
    model = PSFormer(M=M, L=L, P=P, horizon=F, n_layers=2)
    trainer = Trainer(model, device="cpu")
    hist = trainer.fit(ds, epochs=80, batch_size=8, lr=1e-3, verbose=False)
    final_loss = hist["loss"][-1]
    assert final_loss < 0.02, f"Model did not overfit enough, final_loss={final_loss}"
