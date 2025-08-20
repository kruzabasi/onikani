# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math

class Trainer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def fit(self, train_dataset, epochs=100, batch_size=8, lr=1e-3, verbose=False):
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = {"loss": []}
        self.model.train()
        for ep in range(epochs):
            running = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                count += xb.size(0)
            epoch_loss = running / count if count else float("nan")
            history["loss"].append(epoch_loss)
            if verbose and (ep % max(1, epochs//5) == 0 or ep == epochs-1):
                print(f"Epoch {ep+1}/{epochs} loss={epoch_loss:.6f}")
        return history
