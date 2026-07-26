import torch
import torch.nn as nn
import pytest

import torch.nn.functional as F

from quantize import quaternary_quant, QuantizedLinear, compute_scale


# ===========================================================================
# compute_scale
# ===========================================================================

class TestComputeScale:
    def test_absmean_channel_shape(self):
        w = torch.randn(8, 16)
        gamma = compute_scale(w, "absmean_channel")
        assert gamma.shape == (8, 1), f"Expected (8, 1), got {gamma.shape}"
        assert not gamma.requires_grad

    def test_absmean_tensor_scalar(self):
        w = torch.randn(8, 16)
        gamma = compute_scale(w, "absmean_tensor")
        assert gamma.ndim == 0, f"Expected scalar, got shape {gamma.shape}"
        assert not gamma.requires_grad

    def test_absmax_channel_shape(self):
        w = torch.randn(8, 16)
        gamma = compute_scale(w, "absmax_channel")
        assert gamma.shape == (8, 1), f"Expected (8, 1), got {gamma.shape}"
        assert not gamma.requires_grad

    def test_absmax_tensor_scalar(self):
        w = torch.randn(8, 16)
        gamma = compute_scale(w, "absmax_tensor")
        assert gamma.ndim == 0, f"Expected scalar, got shape {gamma.shape}"
        assert not gamma.requires_grad

    def test_clamp_eps(self):
        """All-zero weights should not produce zero scale (would give NaN)."""
        w = torch.zeros(4, 8)
        for mode in ["absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor"]:
            gamma = compute_scale(w, mode)
            assert (gamma >= 1e-5).all(), f"Scale too small for mode={mode}"

    def test_values_absmean_channel(self):
        w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        gamma = compute_scale(w, "absmean_channel")
        expected = torch.tensor([[2.0], [5.0]])  # row means
        assert torch.allclose(gamma, expected, atol=1e-6)

    def test_values_absmax_channel(self):
        w = torch.tensor([[1.0, -5.0, 3.0], [-10.0, 2.0, 4.0]], dtype=torch.float32)
        gamma = compute_scale(w, "absmax_channel")
        expected = torch.tensor([[5.0], [10.0]])
        assert torch.allclose(gamma, expected, atol=1e-6)


# ===========================================================================
# quaternary_quant
# ===========================================================================

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

    def test_quant_with_channel_scale(self):
        """Per-channel scale should broadcast correctly."""
        w = torch.tensor([[1.0, -8.0], [4.0, 6.0]], dtype=torch.float32)
        scale = torch.tensor([[4.0], [5.0]])  # row-wise
        c = 0.25
        q = quaternary_quant(w, c, scale)
        # row 0: x = [0.25, -2.0] → t = 0.625 → [c, -1] * scale = [0.25*4=1, -1*4=-4]
        # row 1: x = [0.8, 1.2] → t = 0.625 → [1, 1] * scale = [1*5=5, 1*5=5]
        expected = torch.tensor([[1.0, -4.0], [5.0, 5.0]], dtype=torch.float32)
        assert torch.equal(q, expected)


# ===========================================================================
# QuantizedLinear
# ===========================================================================

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

        # With identity STE, all weights receive gradient
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

    # ----- Scale modes -----

    @pytest.mark.parametrize("scale_mode", [
        "absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor",
    ])
    def test_all_scale_modes_forward(self, scale_mode):
        layer = QuantizedLinear(32, 64, bias=True, c=0.25, scale_mode=scale_mode)
        x = torch.randn(2, 8, 32)
        out = layer(x)
        assert out.shape == (2, 8, 64)

    @pytest.mark.parametrize("scale_mode", [
        "absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor",
    ])
    def test_all_scale_modes_backward(self, scale_mode):
        layer = QuantizedLinear(32, 32, bias=False, c=0.25, scale_mode=scale_mode)
        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()

    # ----- STE clip mode -----

    def test_ste_clip_grads_zero_for_outliers(self):
        """With clip STE, weights with |w/gamma| > 1 get zero gradient."""
        layer = QuantizedLinear(8, 8, bias=False, c=0.25, ste_mode="clip")
        # Create a scenario where one specific weight is an outlier relative to
        # its channel scale.  Make most row-0 values small but one huge.
        with torch.no_grad():
            nn.init.normal_(layer.weight, std=0.02)
            layer.weight.data[0, 0] = 100.0  # single extreme outlier in row 0

        x = torch.randn(4, 8)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        # The outlier element should have zero gradient (clipped)
        assert layer.weight.grad[0, 0] == 0.0, \
            f"Outlier element should have zero grad, got {layer.weight.grad[0, 0]}"
        # Regular elements inside range should have non-zero gradient
        assert (layer.weight.grad[1:, :] != 0).any(), "Non-outlier rows should have some grad"

    def test_ste_clip_forward_matches_quantized_at_lambda_one(self):
        """Clip STE must not change forward values vs Q(W) at λ=1 (only grads)."""
        layer = QuantizedLinear(4, 1, bias=False, c=0.25, ste_mode="clip",
                                scale_mode="absmean_channel")
        with torch.no_grad():
            layer.weight.data = torch.tensor([[0.1, 0.1, 0.1, 10.0]])
        layer.lambda_ = 1.0
        x = torch.ones(1, 4)

        gamma = compute_scale(layer.weight, "absmean_channel")
        w_q = quaternary_quant(layer.weight, 0.25, gamma)
        expected = F.linear(x, w_q)
        out = layer(x)
        assert torch.allclose(out, expected), (
            f"clip STE forward should match Q(W), got {out} vs {expected}"
        )

    def test_ste_clip_lambda_zero(self):
        """STE clip + lambda=0 → pure FP forward."""
        layer = QuantizedLinear(32, 32, bias=False, c=0.25, ste_mode="clip")
        layer.lambda_ = 0.0
        x = torch.randn(4, 32)
        ref = F.linear(x, layer.weight)
        out = layer(x)
        assert torch.equal(out, ref)

    def test_ste_clip_lambda_one(self):
        """STE clip + lambda=1 → quantized forward (not pure FP)."""
        layer = QuantizedLinear(32, 32, bias=False, c=0.25, ste_mode="clip")
        layer.lambda_ = 1.0
        x = torch.randn(4, 32)
        ref = F.linear(x, layer.weight)
        out = layer(x)
        assert not torch.equal(out, ref)
        # Also equals explicit Q(W) matmul
        gamma = compute_scale(layer.weight, layer.scale_mode)
        w_q = quaternary_quant(layer.weight, layer.c, gamma)
        assert torch.allclose(out, F.linear(x, w_q))

    # ----- extra_repr -----

    def test_extra_repr_includes_scale_and_ste(self):
        layer = QuantizedLinear(16, 32, bias=True, c=0.5, scale_mode="absmax_channel", ste_mode="clip")
        r = layer.extra_repr()
        assert "scale_mode=absmax_channel" in r
        assert "ste_mode=clip" in r
        assert "c=0.5" in r


class TestCommitmentAndBins:
    def test_commitment_loss_zero_when_on_grid(self):
        from quantize import quant_commitment_loss

        layer = QuantizedLinear(8, 4, bias=False, c=0.25)
        with torch.no_grad():
            # Fixed point of absmean_channel Q: uniform |w| so γ=|w| and codes stay ±1.
            layer.weight.fill_(0.5)
        model = nn.Sequential(layer)
        loss = quant_commitment_loss(model)
        assert loss.ndim == 0
        assert float(loss.detach()) < 1e-10

    def test_commitment_loss_positive_off_grid(self):
        from quantize import quant_commitment_loss

        layer = QuantizedLinear(16, 8, bias=False, c=0.25)
        with torch.no_grad():
            layer.weight.normal_(0, 1.0)
        model = nn.Module()
        model.q = layer
        loss = quant_commitment_loss(model)
        assert float(loss) > 0.0
        loss.backward()
        assert layer.weight.grad is not None

    def test_bin_stats_sum_to_one(self):
        from quantize import quant_bin_stats

        layer = QuantizedLinear(32, 16, bias=False, c=0.25)
        model = nn.Module()
        model.q = layer
        stats = quant_bin_stats(model)
        assert stats["n"] == 32 * 16
        s = sum(stats["frac"].values())
        assert abs(s - 1.0) < 1e-5
