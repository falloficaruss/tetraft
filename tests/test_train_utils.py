"""Tests for trainer helpers and eval device utility (no full model download)."""

import torch
import torch.nn as nn

from config import QAFTConfig, SMOKE_PRESETS, apply_smoke_preset
from eval import evaluate_perplexity, model_device
from model import dump_linear_inventory, replace_from_config, set_quant_lambda
from quantize import QuantizedLinear
from train import (
    _autocast_dtype,
    _build_optimizer,
    _build_scheduler,
    _cosine_with_min_lr_lambda,
    _optimizer_schedule_steps,
)


class _TinyLM(nn.Module):
    """Minimal causal-LM-like module with a real loss for eval tests."""

    def __init__(self, vocab=32, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        h = self.fc(self.embed(input_ids))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            # Shifted CE like HF CausalLM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


class TestModelDevice:
    def test_uses_model_device_attr(self):
        m = _TinyLM()
        m.device = torch.device("cpu")
        assert model_device(m).type == "cpu"

    def test_falls_back_to_param_device(self):
        m = nn.Linear(4, 4)
        assert model_device(m) == next(m.parameters()).device


class _InvModel(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, d)
        self.q_proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, 100, bias=False)


class TestInventoryAndLambda:
    def test_dump_inventory_marks_lm_head_skipped(self):
        model = _InvModel(d=32)
        inv = dump_linear_inventory(model)
        assert inv["summary"]["n_linear"] >= 2
        lm = [r for r in inv["modules"] if r["name"] == "lm_head"][0]
        assert lm["status"] == "skipped"
        assert lm["skip_reason"] == "skip_lm_head"
        eligible = [r for r in inv["modules"] if r["status"] == "eligible"]
        assert any(r["name"] == "q_proj" for r in eligible)

    def test_set_quant_lambda(self):
        model = _InvModel(d=32)
        cfg = QAFTConfig()
        replace_from_config(model, cfg, verbose=False)
        n = set_quant_lambda(model, 0.3)
        assert n > 0
        for m in model.modules():
            if isinstance(m, QuantizedLinear):
                assert m.lambda_ == 0.3


class TestOptimizerAndAutocast:
    def test_build_optimizer_adamw_fallback(self):
        model = nn.Linear(8, 8)
        cfg = QAFTConfig(use_8bit_adam=False, learning_rate=1e-3)
        opt = _build_optimizer(model, cfg)
        assert opt is not None
        assert len(opt.param_groups) == 2

    def test_autocast_disabled_on_cpu(self):
        cfg = QAFTConfig(use_bf16=True)
        assert _autocast_dtype(cfg, torch.device("cpu")) is None


class TestEvaluatePerplexity:
    def test_finite_ppl_on_tiny_batch(self):
        model = _TinyLM()
        ids = torch.randint(0, 32, (2, 8))
        batch = {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }
        loader = [batch]
        ppl = evaluate_perplexity(model, loader, max_batches=1)
        assert ppl > 1.0
        assert ppl < 1e6


class TestSmokePresets:
    def test_preset_names(self):
        assert set(SMOKE_PRESETS) >= {
            "short", "longer", "full_smoke", "scale_25m", "scale_50m",
        }

    def test_apply_longer_preset(self):
        cfg = QAFTConfig()
        apply_smoke_preset(cfg, "longer")
        assert cfg.max_steps == 640
        assert cfg.quant_warmup_steps == 128
        assert cfg.warmup_steps == 64
        assert cfg.tokens_budget() == 640 * 1 * 512 * 8  # ≈ 2.6M

    def test_scale_25m_preset(self):
        cfg = apply_smoke_preset(QAFTConfig(), "scale_25m")
        assert cfg.max_steps == 6104
        assert cfg.tokens_budget() == 6104 * 4096  # ≈ 25.0M
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.min_lr_ratio == 0.1

    def test_unknown_preset_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown smoke preset"):
            apply_smoke_preset(QAFTConfig(), "nope")


class TestScheduler:
    def test_optimizer_schedule_steps_divides_by_accum(self):
        cfg = QAFTConfig(max_steps=1280, warmup_steps=128, gradient_accumulation_steps=8)
        warm, total = _optimizer_schedule_steps(cfg)
        assert total == 160
        assert warm == 16

    def test_cosine_ends_at_min_ratio(self):
        v0 = _cosine_with_min_lr_lambda(
            0, num_warmup_steps=10, num_training_steps=100, min_lr_ratio=0.1
        )
        assert v0 == 0.0
        vend = _cosine_with_min_lr_lambda(
            100, num_warmup_steps=10, num_training_steps=100, min_lr_ratio=0.1
        )
        assert abs(vend - 0.1) < 1e-6
        vmid = _cosine_with_min_lr_lambda(
            55, num_warmup_steps=10, num_training_steps=100, min_lr_ratio=0.1
        )
        assert 0.1 < vmid < 1.0

    def test_build_cosine_scheduler(self):
        model = nn.Linear(4, 4)
        cfg = QAFTConfig(
            use_8bit_adam=False,
            max_steps=80,
            warmup_steps=16,
            gradient_accumulation_steps=8,
            lr_scheduler_type="cosine",
            min_lr_ratio=0.1,
            learning_rate=1e-3,
        )
        opt = _build_optimizer(model, cfg)
        sched = _build_scheduler(opt, cfg)
        # 10 optimizer steps total; step through and check final lr ≈ 1e-4
        for _ in range(10):
            opt.step()
            sched.step()
        lr = sched.get_last_lr()[0]
        assert abs(lr - 1e-4) < 1e-6
