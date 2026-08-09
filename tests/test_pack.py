"""Unit tests for pack.py — no HF, no checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn

from pack import (
    PackedQuantizedLinear,
    export_packed_state,
    load_packed,
    materialize_packed_model,
    pack_indices_u2,
    pack_weight_matrix,
    save_packed,
    unpack_indices_u2,
    unpack_weight_matrix,
)
from quantize import QuantizedLinear, compute_scale, quaternary_quant


def test_pack_unpack_indices_roundtrip():
    for n in (1, 3, 4, 5, 128):
        idx = torch.randint(0, 4, (7, n), dtype=torch.uint8)
        packed = pack_indices_u2(idx)
        assert packed.dtype == torch.uint8
        assert packed.shape[-1] == (n + 3) // 4
        got = unpack_indices_u2(packed, n)
        assert got.shape == idx.shape
        assert torch.equal(got, idx)


def test_pack_weight_equals_quaternary_quant():
    torch.manual_seed(0)
    W = torch.randn(64, 128)
    c = 0.25
    mode = "absmean_channel"
    blob = pack_weight_matrix(W, c, mode)
    got = unpack_weight_matrix(blob)
    ref = quaternary_quant(W, c, compute_scale(W, mode)).float()
    assert torch.equal(got, ref)


def test_pack_weight_c05():
    torch.manual_seed(1)
    W = torch.randn(32, 17)
    c = 0.5
    mode = "absmean_channel"
    blob = pack_weight_matrix(W, c, mode)
    got = unpack_weight_matrix(blob)
    ref = quaternary_quant(W, c, compute_scale(W, mode)).float()
    assert torch.equal(got, ref)


def test_packed_linear_forward_matches_quantized():
    torch.manual_seed(2)
    ql = QuantizedLinear(16, 8, bias=True, c=0.25, scale_mode="absmean_channel")
    ql.lambda_ = 1.0
    pl = PackedQuantizedLinear.from_quantized_linear(ql)
    x = torch.randn(4, 16)
    y_q = ql(x)
    y_p = pl(x)
    assert torch.allclose(y_p, y_q, atol=1e-5, rtol=1e-5)


def test_export_import_tiny_sequential(tmp_path):
    torch.manual_seed(3)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = QuantizedLinear(8, 4, bias=False, c=0.25)
            self.fp = nn.Linear(4, 2, bias=True)

        def forward(self, x):
            return self.fp(self.q(x))

    m = Tiny()
    m.q.lambda_ = 1.0
    x = torch.randn(3, 8)
    y_ref = m(x).detach()

    packed = export_packed_state(
        m, c=0.25, scale_mode="absmean_channel", model_name="tiny"
    )
    assert "q" in packed["quant"]
    assert any(k.startswith("fp.") for k in packed["residual"])

    path = tmp_path / "tiny_packed.pt"
    save_packed(path, packed)
    loaded = load_packed(path)

    m2 = Tiny()
    # replace q with dummy Linear-shaped QuantizedLinear then materialize
    materialize_packed_model(m2, loaded, strict_residual=True)
    assert isinstance(m2.q, PackedQuantizedLinear)
    assert isinstance(m2.fp, nn.Linear)
    y_p = m2(x)
    assert torch.allclose(y_p, y_ref, atol=1e-5, rtol=1e-5)


def test_refuse_lora():
    ql = QuantizedLinear(8, 4, bias=False, c=0.25, lora_rank=4)
    try:
        PackedQuantizedLinear.from_quantized_linear(ql)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "LoRA" in str(e)


def test_export_from_state_dict_topology():
    """Pack from sd without materializing topology weights (RAM-safe path)."""
    from pack import export_packed_state_from_state_dict, materialize_packed_model

    torch.manual_seed(4)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = QuantizedLinear(8, 4, bias=True, c=0.25)
            self.fp = nn.Linear(4, 2, bias=True)

        def forward(self, x):
            return self.fp(self.q(x))

    src = Tiny()
    src.q.lambda_ = 1.0
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    x = torch.randn(2, 8)
    y_ref = src(x).detach()

    topo = Tiny()  # different random init; only structure used
    packed = export_packed_state_from_state_dict(
        topo,
        sd,
        c=0.25,
        scale_mode="absmean_channel",
        model_name="tiny",
        parity_modules=1,
    )
    assert "q" in packed["quant"]
    assert "fp.weight" in packed["residual"]

    dst = Tiny()
    materialize_packed_model(dst, packed, strict_residual=True)
    assert isinstance(dst.q, PackedQuantizedLinear)
    assert torch.allclose(dst(x), y_ref, atol=1e-5, rtol=1e-5)
