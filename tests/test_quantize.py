import torch
import pytest

import torch.nn.functional as F

from quantize import quaternary_quant, QuantizedLinear


class TestQuaternaryQuant:
    def test_quant_returns_quaternary_values(self):
        w = torch.tensor([[-2.0, -0.7, -0.3, 0.3, 0.7, 2.0]], dtype=torch.float32)
        scale = w.abs().max()
        c = 0.25
        q = quaternary_quant(w, c, scale)

        # x = w / scale = [-1.0, -0.35, -0.15, 0.15, 0.35, 1.0]
        # t = (1+0.25)/2 = 0.625
        # -1.0 < -t → -1.0 * scale = -2.0
        # -0.35 in [-t, 0) → -c * scale = -0.5
        # -0.15 in [-t, 0) → -c * scale = -0.5
        # 0.15 in [0, t) → c * scale = 0.5
        # 0.35 in [0, t) → c * scale = 0.5
        # 1.0 >= t → 1.0 * scale = 2.0
        expected = torch.tensor([[-2.0, -0.5, -0.5, 0.5, 0.5, 2.0]])
        assert torch.equal(q, expected)

    def test_quant_all_values_in_quaternary_grid(self):
        w = torch.randn(10, 10) * 2
        scale = w.abs().max()
        c = 0.25
        q = quaternary_quant(w, c, scale)

        q_scaled = q / scale
        unique = torch.unique(q_scaled.round(decimals=6))
        for val in unique.tolist():
            assert val in (-1.0, -c, c, 1.0), f"Unexpected quantized value: {val}"


class TestQuantizedLinear:
    def test_forward_output_shape(self):
        layer = QuantizedLinear(64, 128, bias=True, c=0.25)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 128)

    def test_forward_no_bias(self):
        layer = QuantizedLinear(64, 128, bias=False, c=0.25)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 128)

    def test_backward_computes_gradients(self):
        layer = QuantizedLinear(64, 128, bias=True, c=0.25)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert x.grad is not None
        if layer.bias is not None:
            assert layer.bias.grad is not None

        assert not torch.isnan(layer.weight.grad).any()
        assert not torch.isnan(x.grad).any()

    def test_backward_grads_flow_through_all_weights(self):
        layer = QuantizedLinear(32, 32, bias=False, c=0.25)
        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # With detach() STE, NO weights are frozen
        assert (layer.weight.grad != 0).all()

    def test_weight_copy_preserves_values(self):
        ref = torch.nn.Linear(32, 64)
        qlayer = QuantizedLinear(32, 64, bias=True, c=0.25)
        qlayer.weight.data = ref.weight.data.clone()
        qlayer.bias.data = ref.bias.data.clone()

        assert torch.equal(qlayer.weight, ref.weight)
        assert torch.equal(qlayer.bias, ref.bias)

    def test_lambda_zero_disables_quantization(self):
        layer = QuantizedLinear(32, 32, bias=False, c=0.25)
        layer.lambda_ = 0.0
        x = torch.randn(4, 32)
        ref = F.linear(x, layer.weight)
        out = layer(x)
        # With lambda=0, forward is pure FP32
        assert torch.equal(out, ref)

    def test_lambda_one_applies_full_quantization(self):
        layer = QuantizedLinear(32, 32, bias=False, c=0.25)
        layer.lambda_ = 1.0
        x = torch.randn(4, 32)
        out = layer(x)
        # Output should differ from pure FP32 (quantization changes it)
        ref = F.linear(x, layer.weight)
        assert not torch.equal(out, ref)

    def test_training_step_updates_weights(self):
        layer = QuantizedLinear(32, 32, bias=False, c=0.25)
        opt = torch.optim.SGD(layer.parameters(), lr=1.0)

        x = torch.randn(4, 32)
        y = torch.randn(4, 32)

        for _ in range(3):
            opt.zero_grad()
            out = layer(x)
            loss = (out - y).pow(2).mean()
            loss.backward()
            opt.step()

        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight), atol=1e-3)
