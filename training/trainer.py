# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any
from training.sam import SAM

class Trainer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = None
        self.scheduler = None
        self._use_sam = False
        self._sam = None

    def fit(
        self,
        train_dataset,
        epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        verbose: bool = False,
        use_sam: bool = False,
        rho: float = 0.05,
        scheduler: Optional[Callable[..., torch.optim.lr_scheduler._LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        train_dataset: PyTorch Dataset
        scheduler: a scheduler class (callable) like torch.optim.lr_scheduler.StepLR
        scheduler_kwargs: kwargs passed to scheduler constructor
        """
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._use_sam = use_sam
        if use_sam:
            # create SAM wrapper
            self._sam = SAM(self.optimizer, self.model, rho=rho)

        if scheduler is not None:
            scheduler_kwargs = scheduler_kwargs or {}
            # instantiate scheduler with optimizer
            self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)
        else:
            self.scheduler = None

        loss_fn = nn.MSELoss()
        history = {"loss": []}
        self.model.train()
        for ep in range(epochs):
            running = 0.0
            count = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                if not use_sam:
                    self.optimizer.zero_grad()
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    self.optimizer.step()
                else:
                    # SAM two-step
                    self.model.zero_grad()
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    # first step: ascend to perturb weights
                    self._sam.first_step(zero_grad=True)

                    # compute loss at perturbed weights
                    pred2 = self.model(xb)  # forward again
                    loss2 = loss_fn(pred2, yb)
                    loss2.backward()
                    # second step: descent using gradients at perturbed weights
                    self._sam.second_step(zero_grad=True)

                running += loss.item() * xb.size(0)
                count += xb.size(0)
            epoch_loss = running / count if count else float("nan")
            history["loss"].append(epoch_loss)
            if self.scheduler is not None:
                # step scheduler once per epoch
                try:
                    self.scheduler.step()
                except Exception:
                    # some schedulers (like ReduceLROnPlateau) require a metric
                    # in that case the user can call scheduler.step(metric) later
                    pass
            if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
                print(f"Epoch {ep+1}/{epochs} loss={epoch_loss:.6f}")
        return history
