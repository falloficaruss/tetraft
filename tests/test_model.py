import torch
import torch.nn as nn
import pytest

from quantize import QuantizedLinear
from model import replace_linear_layers


class _DummyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, d * 4)
        self.up_proj = nn.Linear(d, d * 4)
        self.down_proj = nn.Linear(d * 4, d)


class _DummyModel(nn.Module):
    def __init__(self, d=64, n_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, d)
        self.layers = nn.ModuleList([_DummyBlock(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, 1000, bias=False)

    def forward(self, x):
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer.q_proj(h)
        h = self.norm(h)
        return self.lm_head(h)


class TestReplaceLinearLayers:
    def test_replaces_linear_in_submodules(self):
        model = _DummyModel(d=64)

        for layer in model.layers:
            assert isinstance(layer.q_proj, nn.Linear)
            assert not isinstance(layer.q_proj, QuantizedLinear)

        replace_linear_layers(model, c=0.5)

        for layer in model.layers:
            assert isinstance(layer.q_proj, QuantizedLinear)

        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, QuantizedLinear)

    def test_skips_lm_head_by_default(self):
        model = _DummyModel(d=64)
        replace_linear_layers(model, c=0.5)
        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, QuantizedLinear)

    def test_replaces_all_linear_when_skip_false(self):
        model = _DummyModel(d=64)
        replace_linear_layers(model, c=0.5, skip_lm_head=False)
        assert isinstance(model.lm_head, QuantizedLinear)

    def test_preserves_weight_values(self):
        model = _DummyModel(d=64)
        orig_weights = {}
        for name, param in model.named_parameters():
            orig_weights[name] = param.data.clone()

        replace_linear_layers(model, c=0.5, skip_lm_head=False)

        for name, param in model.named_parameters():
            assert torch.equal(param.data, orig_weights[name]), f"Weight mismatch: {name}"

    def test_forward_passes_with_replaced_layers(self):
        model = _DummyModel(d=64)
        replace_linear_layers(model, c=0.5)

        x = torch.randint(0, 1000, (2, 16))
        out = model(x)
        assert out.shape == (2, 16, 1000)
