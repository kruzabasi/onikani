# training/sam.py
import torch

class SAM:
    """
    A simple Sharpness-Aware Minimization (SAM) wrapper.
    Usage pattern:
      sam = SAM(optimizer, model, rho=0.05)
      loss.backward()
      sam.first_step(zero_grad=True)
      # compute loss on perturbed weights
      loss_perturbed.backward()
      sam.second_step(zero_grad=True)
    This implementation stores per-parameter epsilon in self._e_w.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module, rho: float = 0.05):
        if rho <= 0.0:
            raise ValueError("rho must be > 0")
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        # store per-param perturbations
        self._e_w = {}

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        self._e_w = {}
        for p in self.model.parameters():
            if p.grad is None:
                continue
            e_w = p.grad * scale
            p.add_(e_w)
            self._e_w[p] = e_w.clone()  # store clone for subtraction later
        if zero_grad:
            self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        # restore original weights
        for p in self.model.parameters():
            if p.grad is None:
                continue
            e_w = self._e_w.pop(p, None)
            if e_w is None:
                continue
            p.sub_(e_w)
        # do optimizer step with gradients computed at perturbed weights
        self.optimizer.step()
        if zero_grad:
            self.optimizer.zero_grad()

    def _grad_norm(self):
        # compute norm over all parameter gradients
        device = None
        norms = []
        for p in self.model.parameters():
            if p.grad is None:
                continue
            if device is None:
                device = p.grad.device
            norms.append(p.grad.detach().norm(2))
        if not norms:
            return torch.tensor(0.0, device=device)
        norms = torch.stack(norms)
        return norms.norm(2)
