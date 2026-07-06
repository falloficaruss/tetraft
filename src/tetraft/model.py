import torch.nn as nn

from .quantize import QuantizedLinear


def replace_linear_layers(
    model,
    c=0.5,
    skip_lm_head=True,
    skip_embed_tokens=True,
):
    skip_names = set()
    if skip_lm_head and hasattr(model, "lm_head"):
        skip_names.add("lm_head")
    if skip_embed_tokens and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        pass  # embed_tokens is nn.Embedding, not Linear, so no risk

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        pass

    _replace_in_module(model, c, skip_names)
    return model


def _replace_in_module(module, c, skip_names):
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            qlinear = QuantizedLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                c=c,
            )
            qlinear.weight.data = child.weight.data.clone().to(child.weight.dtype)
            if child.bias is not None:
                qlinear.bias.data = child.bias.data.clone()
            setattr(module, name, qlinear)
        else:
            _replace_in_module(child, c, skip_names)
