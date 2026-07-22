import logging
from typing import Set

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


def _matches_any(name: str, patterns: Set[str]) -> bool:
    """Return ``True`` if *name* (or any dot-separated component) is in *patterns*."""
    for part in name.split("."):
        if part in patterns:
            return True
    return False


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
    verbose: bool = True,
) -> nn.Module:
    """Replace all eligible ``nn.Linear`` submodules with ``QuantizedLinear``.

    Idempotent — already-replaced ``QuantizedLinear`` modules are left untouched.
    Returns the model (modified in-place).
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
                prefix=full_name,
            )
