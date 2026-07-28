import logging
from typing import Any, Dict, List, Optional, Set

import torch.nn as nn

from quantize import QuantizedLinear, apply_weight_calib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skip patterns — names / partial names that should never be quantized
# ---------------------------------------------------------------------------
SKIP_VISION_PATTERNS: Set[str] = {
    "vision_tower",
    "vision_model",
    "visual",
}

SKIP_MTP_PATTERNS: Set[str] = {
    "mtp",
    "MTP",
}

# Qwen3.5 hybrid: Gated DeltaNet lives under decoder ``linear_attn``
# (Linears: in_proj_qkv/z/b/a, out_proj). Full attn is ``self_attn``; FFN is ``mlp``.
SKIP_LINEAR_ATTN_PATTERNS: Set[str] = {
    "linear_attn",
}


def _matches_any(name: str, patterns: Set[str]) -> bool:
    """Return ``True`` if *name* (or any dot-separated component) is in *patterns*."""
    for part in name.split("."):
        if part in patterns:
            return True
    return False


def _skip_reason(
    name: str,
    module: nn.Module,
    skip_names: Set[str],
    skip_vision: bool,
    skip_mtp: bool,
    skip_linear_attn: bool = False,
) -> Optional[str]:
    """Return a human-readable skip reason, or ``None`` if the module is eligible."""
    leaf = name.rsplit(".", 1)[-1] if name else name
    if leaf in skip_names or name in skip_names:
        if leaf == "lm_head" or name.endswith("lm_head"):
            return "skip_lm_head"
        if leaf == "embed_tokens" or name.endswith("embed_tokens"):
            return "skip_embed_tokens"
        return f"skip_name:{leaf}"
    if skip_vision and _matches_any(name, SKIP_VISION_PATTERNS):
        return "skip_vision"
    if skip_mtp and _matches_any(name, SKIP_MTP_PATTERNS):
        return "skip_mtp"
    if skip_linear_attn and _matches_any(name, SKIP_LINEAR_ATTN_PATTERNS):
        return "skip_linear_attn"
    if isinstance(module, QuantizedLinear):
        return "already_quantized"
    if not isinstance(module, nn.Linear):
        return "not_linear"
    return None


def dump_linear_inventory(
    model: nn.Module,
    skip_lm_head: bool = True,
    skip_embed_tokens: bool = True,
    skip_vision: bool = True,
    skip_mtp: bool = True,
    skip_linear_attn: bool = False,
) -> Dict[str, Any]:
    """Inventory all ``nn.Linear`` / ``QuantizedLinear`` modules and skip policy.

    Returns a dict with per-module rows and a summary suitable for JSON dump
    (Phase 1.2 language-only policy check).
    """
    skip_names: Set[str] = set()
    if skip_lm_head:
        skip_names.add("lm_head")
    if skip_embed_tokens:
        skip_names.add("embed_tokens")

    rows: List[Dict[str, Any]] = []
    n_linear = 0
    n_eligible = 0
    n_skipped = 0
    eligible_params = 0
    skipped_params = 0

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            n_linear += 1
            n_params = module.weight.numel() + (
                module.bias.numel() if module.bias is not None else 0
            )
            reason = _skip_reason(
                name, module, skip_names, skip_vision, skip_mtp, skip_linear_attn
            )
            # Already quantized counts as eligible (would be / was replaced).
            status = "quantized"
            n_eligible += 1
            eligible_params += n_params
            rows.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                    "n_params": n_params,
                    "status": status,
                    "skip_reason": reason if reason == "already_quantized" else None,
                }
            )
            continue

        if not isinstance(module, nn.Linear):
            continue

        n_linear += 1
        n_params = module.weight.numel() + (
            module.bias.numel() if module.bias is not None else 0
        )
        reason = _skip_reason(
            name, module, skip_names, skip_vision, skip_mtp, skip_linear_attn
        )
        if reason is None:
            status = "eligible"
            n_eligible += 1
            eligible_params += n_params
        else:
            status = "skipped"
            n_skipped += 1
            skipped_params += n_params

        rows.append(
            {
                "name": name,
                "type": type(module).__name__,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                "n_params": n_params,
                "status": status,
                "skip_reason": reason,
            }
        )

    summary = {
        "n_linear": n_linear,
        "n_eligible": n_eligible,
        "n_skipped": n_skipped,
        "eligible_params": eligible_params,
        "skipped_params": skipped_params,
    }
    logger.info(
        "Linear inventory: %d linear, %d eligible (%s params), %d skipped (%s params)",
        n_linear,
        n_eligible,
        f"{eligible_params:,}",
        n_skipped,
        f"{skipped_params:,}",
    )
    return {"summary": summary, "modules": rows}


def set_quant_lambda(model: nn.Module, lambda_val: float) -> int:
    """Set ``lambda_`` on every ``QuantizedLinear``. Returns number of modules updated."""
    n = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.lambda_ = float(lambda_val)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Parameter counting helpers
# ---------------------------------------------------------------------------

def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Recursive replacement
# ---------------------------------------------------------------------------

def replace_from_config(model: nn.Module, config, verbose: bool = True) -> nn.Module:
    """Apply ``replace_linear_layers`` using knobs from a ``QAFTConfig``."""
    return replace_linear_layers(
        model,
        c=config.quaternary_c,
        scale_mode=config.scale_mode,
        ste_mode=config.ste_mode,
        trust_softness=float(getattr(config, "trust_softness", 1.0)),
        skip_lm_head=config.skip_lm_head,
        skip_embed_tokens=config.skip_embed_tokens,
        skip_vision=config.skip_vision,
        skip_mtp=config.skip_mtp,
        skip_linear_attn=bool(getattr(config, "skip_linear_attn", False)),
        pre_rms=bool(getattr(config, "pre_rms", False)),
        lora_rank=int(getattr(config, "lora_rank", 0) or 0),
        lora_alpha=getattr(config, "lora_alpha", None),
        weight_calib=str(getattr(config, "weight_calib", "none") or "none"),
        verbose=verbose,
    )


def replace_linear_layers(
    model: nn.Module,
    c: float = 0.25,
    scale_mode: str = "absmean_channel",
    ste_mode: str = "identity",
    trust_softness: float = 1.0,
    skip_lm_head: bool = True,
    skip_embed_tokens: bool = True,
    skip_vision: bool = True,
    skip_mtp: bool = True,
    skip_linear_attn: bool = False,
    pre_rms: bool = False,
    lora_rank: int = 0,
    lora_alpha: Optional[float] = None,
    weight_calib: str = "none",
    verbose: bool = True,
) -> nn.Module:
    """Replace all eligible ``nn.Linear`` submodules with ``QuantizedLinear``.

    Idempotent — already-replaced ``QuantizedLinear`` modules are left untouched.
    Returns the model (modified in-place).

    When ``skip_linear_attn`` is True, any module path containing ``linear_attn``
    is left in full precision (Qwen3.5 Gated DeltaNet scope ablation).

    Bundle adapters (optional): ``pre_rms``, ``lora_rank``, ``weight_calib``.
    """
    # Build a set of module names to skip (exact or wildcard).
    skip_names: Set[str] = set()
    if skip_lm_head:
        skip_names.add("lm_head")
    if skip_embed_tokens:
        skip_names.add("embed_tokens")

    # Count total parameters before replacement
    total_params = _count_params(model)

    _replace_in_module(
        model,
        c=c,
        scale_mode=scale_mode,
        ste_mode=ste_mode,
        trust_softness=float(trust_softness),
        skip_names=skip_names,
        skip_vision=skip_vision,
        skip_mtp=skip_mtp,
        skip_linear_attn=skip_linear_attn,
        pre_rms=pre_rms,
        lora_rank=int(lora_rank or 0),
        lora_alpha=lora_alpha,
        weight_calib=weight_calib or "none",
        prefix="",
    )

    # Report quaternary weight mass only (exclude LoRA / pre_rms adapters)
    quantized_params = sum(
        int(sub.weight.numel())
        for sub in model.modules()
        if isinstance(sub, QuantizedLinear)
    )
    adapter_params = 0
    for sub in model.modules():
        if not isinstance(sub, QuantizedLinear):
            continue
        if sub.pre_rms is not None:
            adapter_params += int(sub.pre_rms.weight.numel())
        if sub.lora_A is not None:
            adapter_params += int(sub.lora_A.numel())
        if sub.lora_B is not None:
            adapter_params += int(sub.lora_B.numel())
    ratio = quantized_params / total_params * 100 if total_params > 0 else 0.0

    if verbose:
        logger.info(
            f"Replace report: {quantized_params:,} / {total_params:,} weight parameters "
            f"quantized ({ratio:.1f}%); adapters={adapter_params:,} "
            f"(pre_rms={pre_rms}, lora_rank={int(lora_rank or 0)}, "
            f"weight_calib={weight_calib})"
        )

    return model


def _replace_in_module(
    module: nn.Module,
    c: float,
    scale_mode: str,
    ste_mode: str,
    trust_softness: float,
    skip_names: Set[str],
    skip_vision: bool,
    skip_mtp: bool,
    skip_linear_attn: bool,
    pre_rms: bool,
    lora_rank: int,
    lora_alpha: Optional[float],
    weight_calib: str,
    prefix: str,
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        # ----- skip policies -----
        if name in skip_names:
            continue
        if skip_vision and _matches_any(full_name, SKIP_VISION_PATTERNS):
            continue
        if skip_mtp and _matches_any(full_name, SKIP_MTP_PATTERNS):
            continue
        if skip_linear_attn and _matches_any(full_name, SKIP_LINEAR_ATTN_PATTERNS):
            continue
        # ----- idempotency -----
        if isinstance(child, QuantizedLinear):
            continue

        if isinstance(child, nn.Linear):
            qlinear = QuantizedLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                c=c,
                scale_mode=scale_mode,
                ste_mode=ste_mode,
                trust_softness=trust_softness,
                pre_rms=pre_rms,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            w = child.weight.data.clone().to(child.weight.dtype)
            w = apply_weight_calib(w, weight_calib)
            qlinear.weight.data = w
            if child.bias is not None:
                qlinear.bias.data = child.bias.data.clone()
            # Keep adapters on same device/dtype as the source linear
            dev, dt = child.weight.device, child.weight.dtype
            if qlinear.pre_rms is not None:
                qlinear.pre_rms.to(device=dev, dtype=dt)
            if qlinear.lora_A is not None:
                qlinear.lora_A.data = qlinear.lora_A.data.to(device=dev, dtype=dt)
                qlinear.lora_B.data = qlinear.lora_B.data.to(device=dev, dtype=dt)
            setattr(module, name, qlinear)
        else:
            _replace_in_module(
                child,
                c=c,
                scale_mode=scale_mode,
                ste_mode=ste_mode,
                trust_softness=trust_softness,
                skip_names=skip_names,
                skip_vision=skip_vision,
                skip_mtp=skip_mtp,
                skip_linear_attn=skip_linear_attn,
                pre_rms=pre_rms,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                weight_calib=weight_calib,
                prefix=full_name,
            )
