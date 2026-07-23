import logging
from typing import Any, Dict, List, Optional, Set

import torch.nn as nn

from quantize import QuantizedLinear

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
        skip_lm_head=config.skip_lm_head,
        skip_embed_tokens=config.skip_embed_tokens,
        skip_vision=config.skip_vision,
        skip_mtp=config.skip_mtp,
        skip_linear_attn=bool(getattr(config, "skip_linear_attn", False)),
        verbose=verbose,
    )


def replace_linear_layers(
    model: nn.Module,
    c: float = 0.25,
    scale_mode: str = "absmean_channel",
    ste_mode: str = "identity",
    skip_lm_head: bool = True,
    skip_embed_tokens: bool = True,
    skip_vision: bool = True,
    skip_mtp: bool = True,
    skip_linear_attn: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Replace all eligible ``nn.Linear`` submodules with ``QuantizedLinear``.

    Idempotent — already-replaced ``QuantizedLinear`` modules are left untouched.
    Returns the model (modified in-place).

    When ``skip_linear_attn`` is True, any module path containing ``linear_attn``
    is left in full precision (Qwen3.5 Gated DeltaNet scope ablation).
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
        skip_names=skip_names,
        skip_vision=skip_vision,
        skip_mtp=skip_mtp,
        skip_linear_attn=skip_linear_attn,
        prefix="",
    )

    # Report
    quantized_params = sum(
        p.numel() for sub in model.modules() if isinstance(sub, QuantizedLinear)
        for p in sub.parameters()
    )
    ratio = quantized_params / total_params * 100 if total_params > 0 else 0.0

    if verbose:
        logger.info(
            f"Replace report: {quantized_params:,} / {total_params:,} parameters "
            f"quantized ({ratio:.1f}%)"
        )

    return model


def _replace_in_module(
    module: nn.Module,
    c: float,
    scale_mode: str,
    ste_mode: str,
    skip_names: Set[str],
    skip_vision: bool,
    skip_mtp: bool,
    skip_linear_attn: bool,
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
            )
            qlinear.weight.data = child.weight.data.clone().to(child.weight.dtype)
            if child.bias is not None:
                qlinear.bias.data = child.bias.data.clone()
            setattr(module, name, qlinear)
        else:
            _replace_in_module(
                child,
                c=c,
                scale_mode=scale_mode,
                ste_mode=ste_mode,
                skip_names=skip_names,
                skip_vision=skip_vision,
                skip_mtp=skip_mtp,
                skip_linear_attn=skip_linear_attn,
                prefix=full_name,
            )
