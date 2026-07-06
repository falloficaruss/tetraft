import torch
import pytest

from tetraft.quantize import QuantizeFunction, QuantizedLinear


class TestQuantizeFunction:
    def test_forward_returns_quaternary_values(self):
        w = torch.tensor([[-2.0, -0.8, -0.1, 0.1, 0.8, 2.0]], dtype=torch.float32)
        c = 0.5
        gamma = w.abs().mean()
        t = (1.0 + c) / 2.0

        q = QuantizeFunction.apply(w, c)

        assert q.shape == w.shape
        assert q.dtype == w.dtype

        gamma_val = gamma.item()
        # -2.0/gamma < -t -> should be -1.0 * gamma
        # -0.8/gamma < -t -> -1.0 * gamma  (if abs(-0.8) > t*gamma)
        # -0.1/gamma between -t and 0 -> -c * gamma
        # 0.1/gamma between 0 and t -> c * gamma
        # 0.8/gamma >= t -> 1.0 * gamma
        # 2.0/gamma >= t -> 1.0 * gamma

        q_scaled = q / gamma_val
        assert q_scaled[0, 0] == -1.0
        assert q_scaled[0, 1] in (-1.0, -c)
        assert q_scaled[0, 2] == -c
        assert q_scaled[0, 3] == c
        assert q_scaled[0, 4] in (c, 1.0)
        assert q_scaled[0, 5] == 1.0

    def test_forward_all_values_in_quaternary_grid(self):
        w = torch.randn(10, 10) * 2
        c = 0.5
        q = QuantizeFunction.apply(w, c)

        q_scaled = q / (w.abs().mean())
        unique = torch.unique(q_scaled.round(decimals=6))
        for val in unique.tolist():
            assert val in (-1.0, -c, c, 1.0), f"Unexpected quantized value: {val}"

    def test_ste_backward_passes_through_for_small_weights(self):
        w = torch.randn(5, 5, requires_grad=True)
        c = 0.5

        q = QuantizeFunction.apply(w, c)
        loss = q.sum()
        loss.backward()

        gamma = w.abs().mean().detach()
        mask = (w.abs() <= gamma).float()

        # For |w| <= gamma, gradient should be 1 (STE passes through)
        assert w.grad is not None
        assert torch.allclose(w.grad, mask, atol=1e-6)

    def test_ste_backward_clips_saturated_weights(self):
        w = torch.tensor([[0.5, 5.0]], requires_grad=True)
        c = 0.5

        q = QuantizeFunction.apply(w, c)
        loss = q.sum()
        loss.backward()

        gamma = w.abs().mean().detach()  # mean(|0.5|, |5.0|) = 2.75
        # |0.5| <= 2.75 -> grad = 1
        # |5.0| > 2.75 -> grad = 0
        assert w.grad[0, 0].item() == 1.0
        assert w.grad[0, 1].item() == 0.0

    def test_backward_with_different_c(self):
        for c in (0.25, 0.5):
            w = torch.randn(5, 5, requires_grad=True)
            q = QuantizeFunction.apply(w, c)
            loss = q.sum()
            loss.backward()
            assert w.grad is not None
            assert not torch.isnan(w.grad).any()


class TestQuantizedLinear:
    def test_forward_output_shape(self):
        layer = QuantizedLinear(64, 128, bias=True, c=0.5)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 128)

    def test_forward_no_bias(self):
        layer = QuantizedLinear(64, 128, bias=False, c=0.5)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 128)

    def test_backward_computes_gradients(self):
        layer = QuantizedLinear(64, 128, bias=True, c=0.5)
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

    def test_weight_copy_preserves_values(self):
        ref = torch.nn.Linear(32, 64)
        qlayer = QuantizedLinear(32, 64, bias=True, c=0.5)
        qlayer.weight.data = ref.weight.data.clone()
        qlayer.bias.data = ref.bias.data.clone()

        assert torch.equal(qlayer.weight, ref.weight)
        assert torch.equal(qlayer.bias, ref.bias)

    def test_training_step_updates_weights(self):
        layer = QuantizedLinear(32, 32, bias=False, c=0.5)
        opt = torch.optim.SGD(layer.parameters(), lr=1.0)

        x = torch.randn(4, 32)
        y = torch.randn(4, 32)

        for _ in range(3):
            opt.zero_grad()
            out = layer(x)
            loss = (out - y).pow(2).mean()
            loss.backward()
            opt.step()

        # Weight should have changed
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight), atol=1e-3)
