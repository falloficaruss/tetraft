"""2-bit packed quaternary weights — reference dequant path (systems v1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize import QuantizedLinear, compute_scale, quaternary_quant

FORMAT_VERSION = 1
FORMAT_NAME = "tetraft-packed-v1"
CODEBOOK_INDICES = (0, 1, 2, 3)

# index u -> code value factor (before * gamma): 0→-1, 1→-c, 2→+c, 3→+1


def codes_from_weight(
    weight: torch.Tensor,
    c: float,
    scale_mode: str = "absmean_channel",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (indices uint8 (out, in) in {0,1,2,3}, gamma float broadcastable)."""
    gamma = compute_scale(weight, scale_mode)
    t = (1.0 + float(c)) / 2.0
    x = weight.float() / gamma.float()
    # Match quaternary_quant thresholds with integer bins (no float re-bucket).
    indices = torch.full(x.shape, 2, dtype=torch.uint8, device=weight.device)  # +c default
    indices = torch.where(x < -t, torch.zeros_like(indices), indices)  # 0 → -1
    indices = torch.where((x >= -t) & (x < 0), torch.full_like(indices, 1), indices)  # -c
    indices = torch.where((x >= 0) & (x < t), torch.full_like(indices, 2), indices)  # +c
    indices = torch.where(x >= t, torch.full_like(indices, 3), indices)  # +1
    return indices, gamma


def pack_indices_u2(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices in {0,1,2,3} along last dim: 4 codes/byte, low bits first.

    byte = i0 | (i1<<2) | (i2<<4) | (i3<<6). Pad last dim with 0 so n % 4 == 0.
    """
    if indices.dtype != torch.uint8:
        indices = indices.to(torch.uint8)
    indices = indices.cpu().contiguous()
    n = indices.shape[-1]
    pad = (4 - (n % 4)) % 4
    if pad:
        pad_shape = list(indices.shape)
        pad_shape[-1] = pad
        indices = torch.cat(
            [indices, torch.zeros(pad_shape, dtype=torch.uint8)], dim=-1
        )
    flat_lead = indices.shape[:-1]
    n4 = indices.shape[-1]
    x = indices.reshape(*flat_lead, n4 // 4, 4)
    packed = (
        x[..., 0].to(torch.int64)
        | (x[..., 1].to(torch.int64) << 2)
        | (x[..., 2].to(torch.int64) << 4)
        | (x[..., 3].to(torch.int64) << 6)
    ).to(torch.uint8)
    return packed.contiguous()


def unpack_indices_u2(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of pack_indices_u2; return uint8 (..., n) cropped to n."""
    packed = packed.cpu().contiguous().to(torch.uint8)
    b = packed.to(torch.int64)
    i0 = (b & 0x3).to(torch.uint8)
    i1 = ((b >> 2) & 0x3).to(torch.uint8)
    i2 = ((b >> 4) & 0x3).to(torch.uint8)
    i3 = ((b >> 6) & 0x3).to(torch.uint8)
    indices = torch.stack([i0, i1, i2, i3], dim=-1)
    # (..., nbytes, 4) -> (..., nbytes*4)
    indices = indices.reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    return indices[..., :n].contiguous()


def dequant_from_indices(
    indices: torch.Tensor,
    gamma: torch.Tensor,
    c: float,
) -> torch.Tensor:
    """Map indices → {-1,-c,c,1} * gamma (float32)."""
    c = float(c)
    # codebook: 0:-1, 1:-c, 2:+c, 3:+1
    table = torch.tensor(
        [-1.0, -c, c, 1.0], dtype=torch.float32, device=indices.device
    )
    codes = table[indices.long()]
    g = gamma.float()
    if g.dim() == 1 and codes.dim() >= 2:
        g = g.view(-1, *([1] * (codes.dim() - 1)))
    elif g.dim() == 0:
        pass
    return codes * g


def pack_weight_matrix(
    weight: torch.Tensor,
    c: float,
    scale_mode: str = "absmean_channel",
) -> dict:
    """Pack one (out, in) weight matrix to 2-bit codes + gamma."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(weight.shape)}")
    out_f, in_f = int(weight.shape[0]), int(weight.shape[1])
    indices, gamma = codes_from_weight(weight, c, scale_mode)
    packed_codes = pack_indices_u2(indices)
    # float32 gamma: fp16 rounds channel scales and breaks bit-exact Q(W) parity
    g = gamma.detach().float().cpu().reshape(-1).contiguous()
    return {
        "out_features": out_f,
        "in_features": in_f,
        "c": float(c),
        "scale_mode": str(scale_mode),
        "gamma": g,
        "packed_codes": packed_codes.cpu().contiguous(),
    }


def unpack_weight_matrix(blob: dict) -> torch.Tensor:
    """packed blob → float32 Q(W) (out, in)."""
    in_f = int(blob["in_features"])
    indices = unpack_indices_u2(blob["packed_codes"], in_f)
    gamma = blob["gamma"]
    if isinstance(gamma, torch.Tensor):
        g = gamma.float()
    else:
        g = torch.tensor(gamma, dtype=torch.float32)
    if g.dim() == 1:
        g = g.view(-1, 1)
    return dequant_from_indices(indices, g, float(blob["c"]))


class PackedQuantizedLinear(nn.Module):
    """Inference-only quaternary linear: packed codes + gamma; reference dequant matmul."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        c: float,
        scale_mode: str,
        packed_codes: torch.Tensor,
        gamma: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.c = float(c)
        self.scale_mode = str(scale_mode)
        self.register_buffer("packed_codes", packed_codes.detach().cpu().contiguous().to(torch.uint8))
        g = gamma.detach().float().reshape(-1).cpu().contiguous()
        if g.numel() != self.out_features:
            raise ValueError(
                f"gamma length {g.numel()} != out_features {self.out_features}"
            )
        self.register_buffer("gamma", g)
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().cpu().contiguous())
        else:
            self.register_buffer("bias", None)

    def dequant_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        indices = unpack_indices_u2(self.packed_codes, self.in_features)
        w = dequant_from_indices(
            indices, self.gamma.float().view(-1, 1), self.c
        )
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequant_weight(dtype=torch.float32)
        if x.dtype != torch.float32:
            w = w.to(x.dtype)
        else:
            w = w.to(device=x.device)
        w = w.to(device=x.device, dtype=x.dtype)
        b = None
        if self.bias is not None:
            b = self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, b)

    @staticmethod
    def from_quantized_linear(mod: QuantizedLinear) -> "PackedQuantizedLinear":
        if int(getattr(mod, "lora_rank", 0) or 0) > 0:
            raise ValueError(
                "PackedQuantizedLinear refuses LoRA (lora_rank>0); export latent adapters separately"
            )
        if getattr(mod, "pre_rms", None) is not None:
            raise ValueError(
                "PackedQuantizedLinear refuses pre_rms; not supported in packed v1"
            )
        blob = pack_weight_matrix(mod.weight.data, float(mod.c), str(mod.scale_mode))
        bias = mod.bias.data if mod.bias is not None else None
        return PackedQuantizedLinear(
            out_features=blob["out_features"],
            in_features=blob["in_features"],
            c=blob["c"],
            scale_mode=blob["scale_mode"],
            packed_codes=blob["packed_codes"],
            gamma=blob["gamma"],
            bias=bias,
        )

    def to_blob(self) -> dict:
        d: Dict[str, Any] = {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "c": self.c,
            "scale_mode": self.scale_mode,
            "gamma": self.gamma.cpu().contiguous(),
            "packed_codes": self.packed_codes.cpu().contiguous(),
        }
        if self.bias is not None:
            d["bias"] = self.bias.cpu().contiguous()
        return d

    @staticmethod
    def from_blob(blob: dict) -> "PackedQuantizedLinear":
        return PackedQuantizedLinear(
            out_features=int(blob["out_features"]),
            in_features=int(blob["in_features"]),
            c=float(blob["c"]),
            scale_mode=str(blob["scale_mode"]),
            packed_codes=blob["packed_codes"],
            gamma=blob["gamma"],
            bias=blob.get("bias"),
        )


def _quant_param_key_prefixes(model: nn.Module) -> set:
    """State-dict key prefixes owned by QuantizedLinear modules (weight/bias/lora/pre_rms)."""
    prefixes = set()
    for name, mod in model.named_modules():
        if isinstance(mod, QuantizedLinear):
            prefixes.add(name)
    return prefixes


def _is_under_quant_module(key: str, quant_names: set) -> bool:
    for qn in quant_names:
        if key == qn or key.startswith(qn + "."):
            # only weight/bias of the QuantizedLinear itself (not nested if any)
            rest = key[len(qn) + 1 :] if key.startswith(qn + ".") else ""
            if rest in (
                "weight",
                "bias",
                "lora_A",
                "lora_B",
                "pre_rms.weight",
            ):
                return True
    return False


def export_packed_state(
    model: nn.Module,
    *,
    c: float,
    scale_mode: str,
    model_name: str,
    extra_meta: Optional[dict] = None,
) -> dict:
    """Export QuantizedLinear modules to packed codes; residual holds other tensors."""
    quant: Dict[str, Any] = {}
    quant_names = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, QuantizedLinear):
            continue
        if int(getattr(mod, "lora_rank", 0) or 0) > 0:
            raise ValueError(f"export refuses LoRA on module {name!r}")
        if getattr(mod, "pre_rms", None) is not None:
            raise ValueError(f"export refuses pre_rms on module {name!r}")
        packed_mod = PackedQuantizedLinear.from_quantized_linear(mod)
        quant[name] = packed_mod.to_blob()
        quant_names.add(name)

    residual: Dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        if _is_under_quant_module(key, quant_names):
            continue
        t = tensor.detach().cpu()
        residual[key] = t

    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "c": float(c),
        "scale_mode": str(scale_mode),
        "index_map": {"0": -1.0, "1": -float(c), "2": float(c), "3": 1.0},
        "quant": quant,
        "residual": residual,
        "meta": dict(extra_meta or {}),
    }


def export_packed_state_from_state_dict(
    topology: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    c: float,
    scale_mode: str,
    model_name: str,
    extra_meta: Optional[dict] = None,
    parity_modules: int = 3,
) -> dict:
    """Pack from ckpt tensors using topology only (no full weight materialize).

    ``topology`` must already have ``QuantizedLinear`` modules where export should
    pack (e.g. after ``replace_from_config`` on a ``from_config`` empty model).
    Weights are read from ``state_dict`` keys ``{name}.weight`` / ``.bias``.
    """
    quant: Dict[str, Any] = {}
    quant_names: set = set()
    n_parity = 0

    for name, mod in topology.named_modules():
        if not isinstance(mod, QuantizedLinear):
            continue
        if int(getattr(mod, "lora_rank", 0) or 0) > 0:
            raise ValueError(f"export refuses LoRA on module {name!r}")
        if getattr(mod, "pre_rms", None) is not None:
            raise ValueError(f"export refuses pre_rms on module {name!r}")
        w_key = f"{name}.weight"
        if w_key not in state_dict:
            raise KeyError(f"state_dict missing quant weight {w_key}")
        for bad in (f"{name}.lora_A", f"{name}.lora_B", f"{name}.pre_rms.weight"):
            if bad in state_dict:
                raise ValueError(f"export refuses adapter key present in ckpt: {bad}")

        W = state_dict[w_key]
        if not isinstance(W, torch.Tensor):
            raise TypeError(f"{w_key} is not a tensor")
        Wf = W.detach().float().cpu()
        mod_c = float(getattr(mod, "c", c))
        mod_mode = str(getattr(mod, "scale_mode", scale_mode))
        blob = pack_weight_matrix(Wf, mod_c, mod_mode)

        if n_parity < int(parity_modules):
            got = unpack_weight_matrix(blob)
            ref = quaternary_quant(Wf, mod_c, compute_scale(Wf, mod_mode)).float()
            if not torch.equal(got, ref):
                raise AssertionError(
                    f"parity fail on {name}: max_diff={(got - ref).abs().max().item()}"
                )
            n_parity += 1

        b_key = f"{name}.bias"
        if b_key in state_dict and state_dict[b_key] is not None:
            blob["bias"] = state_dict[b_key].detach().float().cpu().contiguous()
        quant[name] = blob
        quant_names.add(name)
        del Wf

    if not quant:
        raise RuntimeError("no QuantizedLinear modules on topology")

    residual: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if _is_under_quant_module(key, quant_names):
            continue
        if isinstance(tensor, torch.Tensor):
            residual[key] = tensor.detach().cpu()
        else:
            residual[key] = tensor

    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "c": float(c),
        "scale_mode": str(scale_mode),
        "index_map": {"0": -1.0, "1": -float(c), "2": float(c), "3": 1.0},
        "quant": quant,
        "residual": residual,
        "meta": dict(extra_meta or {}),
    }


def save_packed(path: Union[str, Path], packed: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, str(path))


def load_packed(path: Union[str, Path], map_location: str = "cpu") -> dict:
    path = Path(path)
    obj = torch.load(str(path), map_location=map_location, weights_only=False)
    if obj.get("format") != FORMAT_NAME:
        raise ValueError(f"unknown packed format: {obj.get('format')!r}")
    if int(obj.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version: {obj.get('format_version')}")
    return obj


def _set_submodule(root: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    if not dotted:
        raise ValueError("cannot replace empty module path")
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def materialize_packed_model(
    base_model: nn.Module,
    packed: dict,
    *,
    strict_residual: bool = True,
) -> nn.Module:
    """In-place replace modules listed in packed['quant']; load residual state."""
    if packed.get("format") != FORMAT_NAME:
        raise ValueError(f"bad format {packed.get('format')!r}")
    quant = packed["quant"]
    for name, blob in quant.items():
        # resolve current submodule
        mod = base_model
        try:
            for p in name.split("."):
                mod = getattr(mod, p)
        except AttributeError as e:
            raise KeyError(f"module path not found: {name}") from e
        if not isinstance(mod, (nn.Linear, QuantizedLinear, PackedQuantizedLinear)):
            raise TypeError(
                f"refuse replace {name}: type {type(mod).__name__} "
                f"(expected Linear/QuantizedLinear)"
            )
        packed_mod = PackedQuantizedLinear.from_blob(blob)
        _set_submodule(base_model, name, packed_mod)

    residual = packed.get("residual") or {}
    missing, unexpected = base_model.load_state_dict(residual, strict=False)
    # Quant weight keys missing is expected; residual should cover the rest.
    if strict_residual:
        # Filter missing keys that belong to packed quant modules
        quant_names = set(quant.keys())
        real_missing = [
            k
            for k in missing
            if not _is_under_quant_module(k, quant_names)
            and not any(
                k == n or k.startswith(n + ".")
                for n in quant_names
            )
        ]
        # Packed modules register packed_codes/gamma/bias — those won't be in residual
        real_missing = [
            k
            for k in real_missing
            if not k.endswith(".packed_codes")
            and not k.endswith(".gamma")
            and not (
                k.endswith(".bias")
                and any(k.rsplit(".", 1)[0] == n for n in quant_names)
            )
        ]
        if real_missing:
            raise RuntimeError(
                f"residual load missing {len(real_missing)} keys: {real_missing[:12]}"
            )
        if unexpected:
            raise RuntimeError(
                f"residual load unexpected {len(unexpected)} keys: {unexpected[:12]}"
            )
    return base_model


def estimate_packed_footprint(packed: dict) -> Dict[str, Any]:
    """Bits / bytes summary for a packed export dict."""
    n_q_params = 0
    bits_gamma = 0
    packed_code_bytes = 0
    for blob in packed.get("quant", {}).values():
        out_f = int(blob["out_features"])
        in_f = int(blob["in_features"])
        n_q_params += out_f * in_f
        g = blob.get("gamma")
        g_bits = 32
        if isinstance(g, torch.Tensor):
            g_bits = int(g.element_size() * 8)
        bits_gamma += g_bits * out_f
        pc = blob["packed_codes"]
        packed_code_bytes += int(pc.numel()) if isinstance(pc, torch.Tensor) else 0
        if blob.get("bias") is not None:
            b = blob["bias"]
            b_bits = 32
            if isinstance(b, torch.Tensor):
                b_bits = int(b.element_size() * 8)
            bits_gamma += b_bits * int(b.numel() if isinstance(b, torch.Tensor) else out_f)

    bits_idx = 2 * n_q_params
    residual_numel = 0
    residual_bytes = 0
    for t in (packed.get("residual") or {}).values():
        if isinstance(t, torch.Tensor):
            residual_numel += int(t.numel())
            residual_bytes += int(t.numel() * t.element_size())

    return {
        "n_quant_modules": len(packed.get("quant", {})),
        "n_quant_params": n_q_params,
        "bits_idx": bits_idx,
        "bits_gamma": bits_gamma,
        "packed_code_bytes": packed_code_bytes,
        "n_residual_params": residual_numel,
        "residual_bytes": residual_bytes,
        "bytes_estimate": packed_code_bytes
        + (bits_gamma // 8)
        + residual_bytes,
    }
