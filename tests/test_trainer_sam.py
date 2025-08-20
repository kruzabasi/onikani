# tests/test_trainer_sam.py
import numpy as np
import torch
import random
from models.psformer.model import PSFormer
from training.datasets import SlidingWindowDataset
from training.trainer import Trainer
import torch.optim as optim

def seed_everything(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def make_tiny_dataset():
    seed_everything(0)
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
    return ds, M, L, P, F

def test_trainer_with_sam_reduces_loss():
    ds, M, L, P, F = make_tiny_dataset()
    model = PSFormer(M=M, L=L, P=P, horizon=F, n_layers=2)
    trainer = Trainer(model, device="cpu")
    hist = trainer.fit(ds, epochs=40, batch_size=8, lr=1e-3, verbose=False, use_sam=True, rho=0.05)
    final_loss = hist["loss"][-1]
    assert final_loss < 0.05, f"SAM trainer didn't reduce loss enough, final_loss={final_loss}"

def test_trainer_with_scheduler_steps_lr():
    ds, M, L, P, F = make_tiny_dataset()
    model = PSFormer(M=M, L=L, P=P, horizon=F, n_layers=1)
    trainer = Trainer(model, device="cpu")
    # Use StepLR with step_size=1, gamma=0.5
    hist = trainer.fit(
        ds,
        epochs=3,
        batch_size=8,
        lr=1e-3,
        verbose=False,
        use_sam=False,
        scheduler=optim.lr_scheduler.StepLR,
        scheduler_kwargs={"step_size": 1, "gamma": 0.5},
    )
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    initial_lr = 1e-3
    final_lr = trainer.optimizer.param_groups[0]["lr"]
    assert final_lr < initial_lr, "Scheduler did not decrease lr"
