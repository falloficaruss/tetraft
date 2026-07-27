import math
import re
from typing import Any, Dict, List, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-5

_BLOCK_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")
_ROLE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


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


def quant_commitment_loss(
    model: nn.Module,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Mean squared commitment ``||W - sg(Q(W))||^2`` over all ``QuantizedLinear`` weights.

    Pulls latent weights toward the discrete grid so the STE path is less dishonest.
    Returns a scalar on *device* (or the first layer's device). Layer errs are moved
    onto that device before summing so ``device_map`` multi-GPU shards do not crash.
    """
    total = None
    n = 0
    reduce_device = torch.device(device) if device is not None else None
    for module in model.modules():
        if not isinstance(module, QuantizedLinear):
            continue
        w = module.weight
        gamma = compute_scale(w, module.scale_mode)
        w_q = quaternary_quant(w, module.c, gamma)
        err = (w - w_q.detach()).pow(2).mean()
        if reduce_device is None:
            reduce_device = err.device
        elif err.device != reduce_device:
            err = err.to(reduce_device)
        total = err if total is None else total + err
        n += 1
    if total is None:
        p = next(model.parameters(), None)
        dev = reduce_device if reduce_device is not None else (
            p.device if p is not None else torch.device("cpu")
        )
        return torch.zeros((), device=dev)
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


def parse_module_role(name: str) -> str:
    """Infer projection role from a module path (Qwen-style suffixes)."""
    leaf = name.rsplit(".", 1)[-1] if name else ""
    if leaf in _ROLE_SUFFIXES:
        return leaf
    for role in _ROLE_SUFFIXES:
        if name.endswith("." + role) or name == role:
            return role
    return "other"


def parse_block_index(name: str) -> Optional[int]:
    """Decoder block index from paths like ``model.layers.12.mlp.up_proj``."""
    m = _BLOCK_RE.search(name or "")
    if m is None:
        return None
    return int(m.group(1))


def _bin_fracs(codes: torch.Tensor, c: float) -> Dict[str, float]:
    n = max(1, int(codes.numel()))
    return {
        "frac_-1": float((codes + 1.0).abs().le(1e-4).sum().item()) / n,
        "frac_-c": float((codes + c).abs().le(1e-4).sum().item()) / n,
        "frac_+c": float((codes - c).abs().le(1e-4).sum().item()) / n,
        "frac_+1": float((codes - 1.0).abs().le(1e-4).sum().item()) / n,
    }


@torch.no_grad()
def per_module_quant_stats(model: nn.Module) -> List[Dict[str, Any]]:
    """Per-``QuantizedLinear`` quantization error and bin mass (RESULTS.md §5.8 D0)."""
    rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantizedLinear):
            continue
        w = module.weight.detach().float()
        gamma = compute_scale(w, module.scale_mode).float()
        w_q = quaternary_quant(w, float(module.c), gamma)
        diff = w - w_q
        n = int(w.numel())
        mse = float(diff.pow(2).mean().item())
        mae = float(diff.abs().mean().item())
        w_norm = float(w.norm().item())
        d_norm = float(diff.norm().item())
        rel_l2 = d_norm / (w_norm + _EPS)
        mean_abs_err_over_gamma = float((diff.abs() / gamma).mean().item())
        codes = (w_q / gamma).reshape(-1)
        c = float(module.c)
        fracs = _bin_fracs(codes, c)
        rows.append(
            {
                "name": name,
                "role": parse_module_role(name),
                "block": parse_block_index(name),
                "n_params": n,
                "out_features": int(module.out_features),
                "in_features": int(module.in_features),
                "c": c,
                "scale_mode": str(module.scale_mode),
                "mse": mse,
                "rmse": math.sqrt(mse),
                "mae": mae,
                "rel_l2": rel_l2,
                "mean_abs_err_over_gamma": mean_abs_err_over_gamma,
                **fracs,
            }
        )
    return rows


def aggregate_layer_map(
    rows: Sequence[Dict[str, Any]],
    *,
    top_k: int = 16,
    sort_key: str = "rel_l2",
) -> Dict[str, Any]:
    """Role / block aggregates and top-k modules for D0 readout."""
    if not rows:
        return {
            "n_modules": 0,
            "n_params": 0,
            "by_role": {},
            "by_block": {},
            "top_modules": [],
            "median_rel_l2": None,
            "median_mean_abs_err_over_gamma": None,
        }

    def _wmean(vals: List[float], weights: List[int]) -> float:
        tw = float(sum(weights)) or 1.0
        return float(sum(v * w for v, w in zip(vals, weights)) / tw)

    by_role: Dict[str, Dict[str, Any]] = {}
    role_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        role_groups.setdefault(str(r["role"]), []).append(r)
    for role, group in sorted(role_groups.items()):
        ws = [int(g["n_params"]) for g in group]
        by_role[role] = {
            "n_modules": len(group),
            "n_params": int(sum(ws)),
            "rel_l2_wmean": _wmean([float(g["rel_l2"]) for g in group], ws),
            "mean_abs_err_over_gamma_wmean": _wmean(
                [float(g["mean_abs_err_over_gamma"]) for g in group], ws
            ),
            "mse_wmean": _wmean([float(g["mse"]) for g in group], ws),
        }

    by_block: Dict[str, Dict[str, Any]] = {}
    block_groups: Dict[Optional[int], List[Dict[str, Any]]] = {}
    for r in rows:
        block_groups.setdefault(r.get("block"), []).append(r)
    for block, group in sorted(block_groups.items(), key=lambda kv: (-1 if kv[0] is None else kv[0])):
        ws = [int(g["n_params"]) for g in group]
        key = "none" if block is None else str(block)
        by_block[key] = {
            "n_modules": len(group),
            "n_params": int(sum(ws)),
            "rel_l2_wmean": _wmean([float(g["rel_l2"]) for g in group], ws),
            "mean_abs_err_over_gamma_wmean": _wmean(
                [float(g["mean_abs_err_over_gamma"]) for g in group], ws
            ),
        }

    key = sort_key if sort_key in rows[0] else "rel_l2"
    ranked = sorted(rows, key=lambda r: float(r[key]), reverse=True)
    top = ranked[: max(0, int(top_k))]
    top_modules = [
        {
            "name": t["name"],
            "role": t["role"],
            "block": t["block"],
            "n_params": t["n_params"],
            "rel_l2": t["rel_l2"],
            "mean_abs_err_over_gamma": t["mean_abs_err_over_gamma"],
            "mse": t["mse"],
        }
        for t in top
    ]

    rels = sorted(float(r["rel_l2"]) for r in rows)
    maeg = sorted(float(r["mean_abs_err_over_gamma"]) for r in rows)
    mid = len(rels) // 2
    median_rel = rels[mid] if len(rels) % 2 == 1 else 0.5 * (rels[mid - 1] + rels[mid])
    median_maeg = maeg[mid] if len(maeg) % 2 == 1 else 0.5 * (maeg[mid - 1] + maeg[mid])

    return {
        "n_modules": len(rows),
        "n_params": int(sum(int(r["n_params"]) for r in rows)),
        "sort_key": key,
        "by_role": by_role,
        "by_block": by_block,
        "top_modules": top_modules,
        "median_rel_l2": float(median_rel),
        "median_mean_abs_err_over_gamma": float(median_maeg),
    }


def suggest_d0_next(
    summary: Dict[str, Any],
    *,
    fp_mask_ppl: Optional[float] = None,
    baseline_ppl: Optional[float] = None,
) -> Dict[str, Any]:
    """Heuristic next step from D0 aggregates (RESULTS.md §5.8)."""
    by_role = summary.get("by_role") or {}
    top = summary.get("top_modules") or []
    median = summary.get("median_rel_l2")
    reasons: List[str] = []
    suggest = "D1_heal_kl_100m"

    if not by_role:
        return {
            "suggest": "unclear",
            "reasons": ["no QuantizedLinear modules found"],
        }

    role_scores = sorted(
        ((r, float(v["rel_l2_wmean"])) for r, v in by_role.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_role, top_score = role_scores[0]
    other_scores = [s for _, s in role_scores[1:]] or [top_score]
    mean_other = sum(other_scores) / len(other_scores)

    attn_roles = {"q_proj", "k_proj", "o_proj"}
    if top_role in attn_roles and top_score > 1.25 * mean_other:
        suggest = "D2_scout_skip_qo"
        reasons.append(
            f"role {top_role} rel_l2_wmean={top_score:.4f} >> other mean {mean_other:.4f}"
        )

    if median is not None and top:
        top8 = top[:8]
        top8_mean = sum(float(t["rel_l2"]) for t in top8) / len(top8)
        if top8_mean > 1.4 * float(median):
            if suggest == "D1_heal_kl_100m":
                suggest = "D2_scout_skip_qo"
            reasons.append(
                f"top-8 mean rel_l2={top8_mean:.4f} >> median {float(median):.4f}"
            )
        elif top8_mean <= 1.15 * float(median) and suggest == "D1_heal_kl_100m":
            reasons.append(
                f"errors relatively flat (top-8 {top8_mean:.4f} vs median {float(median):.4f})"
            )

    if (
        fp_mask_ppl is not None
        and baseline_ppl is not None
        and baseline_ppl > 0
        and fp_mask_ppl < 0.92 * baseline_ppl
    ):
        suggest = "D2_scout_skip_qo"
        reasons.append(
            f"fp_mask PPL {fp_mask_ppl:.3f} << baseline {baseline_ppl:.3f} "
            "(latent-W upper bound on worst modules)"
        )

    if not reasons:
        reasons.append("no strong concentration signal; default to longer KL")

    return {
        "suggest": suggest,
        "reasons": reasons,
        "top_role": top_role,
        "top_role_rel_l2_wmean": top_score,
        "role_rel_l2_ranking": [
            {"role": r, "rel_l2_wmean": s} for r, s in role_scores
        ],
    }
