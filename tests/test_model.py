import logging
import io

import torch
import torch.nn as nn
import pytest

from quantize import QuantizedLinear
from model import replace_linear_layers, replace_from_config
from config import QAFTConfig


# ===========================================================================
# Helpers
# ===========================================================================

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


class _DummyVisionModel(nn.Module):
    """A model with a vision tower that should be skipped by default."""
    def __init__(self, d=64):
        super().__init__()
        self.vision_tower = nn.Linear(d * 16, d)
        self.embed_tokens = nn.Embedding(1000, d)
        self.q_proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, 1000, bias=False)

    def forward(self, x):
        return self.lm_head(self.q_proj(self.embed_tokens(x)))


class _DummyMTPModel(nn.Module):
    """A model with an MTP head that should be skipped by default."""
    def __init__(self, d=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, d)
        self.q_proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, 1000, bias=False)
        self.mtp = nn.Linear(d, 1000, bias=False)

    def forward(self, x):
        return self.lm_head(self.q_proj(self.embed_tokens(x)))


# ===========================================================================
# Tests
# ===========================================================================

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

    # ----- Idempotency -----

    def test_replace_is_idempotent(self):
        """Applying replace_linear_layers twice should not change anything."""
        model = _DummyModel(d=64)
        replace_linear_layers(model, c=0.25)
        state_after_first = {n: p.data.clone() for n, p in model.named_parameters()}

        replace_linear_layers(model, c=0.25)
        for n, p in model.named_parameters():
            assert torch.equal(p.data, state_after_first[n]), f"Weight changed on second replace: {n}"

        # All Linears should still be QuantizedLinear (not re-replaced)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, QuantizedLinear):
                # Only lm_head and embed_tokens (which isn't Linear) should remain
                assert "lm_head" in name or "norm" in name, f"Unexpected non-quantized Linear: {name}"

    # ----- Vision skip -----

    def test_skips_vision_tower_by_default(self):
        model = _DummyVisionModel(d=64)
        replace_linear_layers(model, c=0.25)
        assert isinstance(model.vision_tower, nn.Linear), "vision_tower should remain nn.Linear"
        assert not isinstance(model.vision_tower, QuantizedLinear)

    def test_vision_skip_flag_controls(self):
        model = _DummyVisionModel(d=64)
        replace_linear_layers(model, c=0.25, skip_vision=False)
        assert isinstance(model.vision_tower, QuantizedLinear), "vision_tower should be quantized when skip_vision=False"

    # ----- MTP skip -----

    def test_skips_mtp_by_default(self):
        model = _DummyMTPModel(d=64)
        replace_linear_layers(model, c=0.25)
        assert isinstance(model.mtp, nn.Linear), "mtp should remain nn.Linear"
        assert not isinstance(model.mtp, QuantizedLinear)

    def test_mtp_skip_flag_controls(self):
        model = _DummyMTPModel(d=64)
        replace_linear_layers(model, c=0.25, skip_mtp=False)
        assert isinstance(model.mtp, QuantizedLinear), "mtp should be quantized when skip_mtp=False"

    # ----- scale_mode passthrough -----

    @pytest.mark.parametrize("scale_mode", [
        "absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor",
    ])
    def test_scale_mode_passthrough(self, scale_mode):
        model = _DummyModel(d=64)
        replace_linear_layers(model, c=0.5, scale_mode=scale_mode)
        for layer in model.layers:
            assert layer.q_proj.scale_mode == scale_mode

    # ----- Replace report -----

    def test_replace_report_logs_quantized_param_count(self, caplog):
        caplog.set_level(logging.INFO)
        model = _DummyModel(d=64)
        replace_linear_layers(model, c=0.25, verbose=True)
        assert any("Replace report" in msg for msg in caplog.messages), \
            f"Expected 'Replace report' in log messages, got: {caplog.messages}"
        # Check that a percentage is reported
        report_msg = [m for m in caplog.messages if "Replace report" in m][0]
        assert "%" in report_msg
        # Rough check: model has 2k+ params in Linears, lm_head 64*1000=64000 is skipped
        assert "quantized" in report_msg.lower()

    # ----- replace_from_config -----

    def test_replace_from_config_passes_knobs(self):
        model = _DummyModel(d=32)
        cfg = QAFTConfig(
            quaternary_c=0.5,
            scale_mode="absmax_tensor",
            ste_mode="clip",
            skip_lm_head=True,
        )
        replace_from_config(model, cfg, verbose=False)
        for layer in model.layers:
            assert isinstance(layer.q_proj, QuantizedLinear)
            assert layer.q_proj.c == 0.5
            assert layer.q_proj.scale_mode == "absmax_tensor"
            assert layer.q_proj.ste_mode == "clip"
        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, QuantizedLinear)
