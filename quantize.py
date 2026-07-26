import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-5


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------

ScaleMode = Literal["absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor"]


def compute_scale(
    w: torch.Tensor,
    mode: str = "absmean_channel",
) -> torch.Tensor:
    """Compute the per-layer or per-channel scale factor ``gamma``.

    The returned tensor is detached from the autograd graph (no gradient flow
    through the scale).  For channel modes the shape is ``(d_out, 1)`` so it
    broadcasts correctly against ``W`` of shape ``(d_out, d_in)``; for tensor
    modes it is a scalar (``shape == ()``).
    """
    w_abs = w.abs()
    if mode == "absmean_channel":
        gamma = w_abs.mean(dim=1, keepdim=True)
    elif mode == "absmean_tensor":
        gamma = w_abs.mean()
    elif mode == "absmax_channel":
        gamma = w_abs.max(dim=1, keepdim=True).values
    elif mode == "absmax_tensor":
        gamma = w_abs.max()
    else:
        raise ValueError(f"Unknown scale_mode: {mode}")

    return gamma.detach().clamp(min=_EPS)


# ---------------------------------------------------------------------------
# Core quantization function
# ---------------------------------------------------------------------------

def quaternary_quant(w: torch.Tensor, c: float, scale: torch.Tensor) -> torch.Tensor:
    """Map ``W`` to the quaternary grid ``{-1, -c, c, 1}`` scaled by ``gamma``.

    ``scale`` can be a scalar (tensor-mode) or broadcastable to ``w``
    (e.g. ``(d_out, 1)`` for channel-mode).
    """
    t = (1.0 + c) / 2.0
    x = w / scale  # broadcasts correctly for both scalar and channel scales

    q = torch.where(x < -t, -1.0, torch.zeros_like(x))
    q = torch.where((x >= -t) & (x < 0), -c, q)
    q = torch.where((x >= 0) & (x < t), c, q)
    q = torch.where(x >= t, 1.0, q)

    return q * scale  # scale broadcasts to match w shape


# ---------------------------------------------------------------------------
# Quantized Linear layer
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """Linear layer with quaternary forward pass and configurable STE."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        c: float = 0.25,
        scale_mode: str = "absmean_channel",
        ste_mode: str = "identity",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.scale_mode = scale_mode
        self.ste_mode = ste_mode
        self.lambda_ = 1.0  # can be overridden by the trainer

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = compute_scale(self.weight, self.scale_mode)
        w_q = quaternary_quant(self.weight, self.c, gamma)

        # ----- Straight-through estimator -----
        # Forward value: (1-λ)W + λ Q(W). Backward: identity STE through W.
        w_eff = self.weight + self.lambda_ * (w_q - self.weight).detach()

        if self.ste_mode == "identity":
            pass
        elif self.ste_mode == "clip":
            # Keep the same forward values, but zero gradients where |W/γ| > 1.
            # w_eff * mask + w_eff.detach() * (~mask) leaves values unchanged
            # while blocking grad on outlier positions (RESEARCH.md §3.2).
            mask = (self.weight.detach() / gamma).abs() <= 1.0
            w_eff = w_eff * mask + w_eff.detach() * (~mask)
        else:
            raise ValueError(f"Unknown ste_mode: {self.ste_mode}")

        return F.linear(x, w_eff, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, c={self.c}, "
            f"scale_mode={self.scale_mode}, ste_mode={self.ste_mode}"
        )


def quant_commitment_loss(model: nn.Module) -> torch.Tensor:
    """Mean squared commitment ``||W - sg(Q(W))||^2`` over all ``QuantizedLinear`` weights.

    Pulls latent weights toward the discrete grid so the STE path is less dishonest.
    Returns a scalar on the model's device (0 if no quantized layers).
    """
    total = None
    n = 0
    for module in model.modules():
        if not isinstance(module, QuantizedLinear):
            continue
        w = module.weight
        gamma = compute_scale(w, module.scale_mode)
        w_q = quaternary_quant(w, module.c, gamma)
        err = (w - w_q.detach()).pow(2).mean()
        total = err if total is None else total + err
        n += 1
    if total is None:
        # No quantized layers — zero that still participates in autograd graphs if needed.
        p = next(model.parameters(), None)
        device = p.device if p is not None else torch.device("cpu")
        return torch.zeros((), device=device)
    return total / n


def quant_bin_stats(model: nn.Module) -> dict:
    """Fraction of weights on each quaternary code (global over all QuantizedLinear)."""
    tallies = {"-1": 0, "-c": 0, "+c": 0, "+1": 0}
    total = 0
    c_ref = None
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, QuantizedLinear):
                continue
            c_ref = module.c
            gamma = compute_scale(module.weight, module.scale_mode)
            w_q = quaternary_quant(module.weight, module.c, gamma)
            codes = (w_q / gamma).reshape(-1)
            c = float(module.c)
            tallies["-1"] += int((codes + 1.0).abs().le(1e-4).sum().item())
            tallies["-c"] += int((codes + c).abs().le(1e-4).sum().item())
            tallies["+c"] += int((codes - c).abs().le(1e-4).sum().item())
            tallies["+1"] += int((codes - 1.0).abs().le(1e-4).sum().item())
            total += codes.numel()
    if total == 0:
        return {"n": 0, "frac": {}, "c": c_ref}
    frac = {k: v / total for k, v in tallies.items()}
    return {"n": total, "counts": tallies, "frac": frac, "c": c_ref}
