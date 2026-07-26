"""Tests for trainer helpers and eval device utility (no full model download)."""

import json
import os

import torch
import torch.nn as nn

from config import QAFTConfig, SMOKE_PRESETS, apply_smoke_preset
from eval import evaluate_perplexity, model_device
from model import dump_linear_inventory, replace_from_config, set_quant_lambda
from quantize import QuantizedLinear
from train import (
    QAFTTrainer,
    _autocast_dtype,
    _build_optimizer,
    _build_scheduler,
    _cosine_with_min_lr_lambda,
    _optimizer_schedule_steps,
    distillation_kl_loss,
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
            "short",
            "longer",
            "full_smoke",
            "full_smoke_no_gdn",
            "heal_25m",
            "heal_50m",
            "scout_kl_5m",
            "heal_kl_25m",
            "heal_kl_50m",
            "scale_25m",
            "scale_50m",
        }

    def test_scout_kl_5m_preset(self):
        cfg = apply_smoke_preset(QAFTConfig(), "scout_kl_5m")
        assert cfg.max_steps == 1280
        assert cfg.tokens_budget() == 1280 * 4096
        assert cfg.quant_warmup_steps == 256
        assert cfg.skip_linear_attn is True
        assert cfg.lr_scheduler_type == "linear"
        assert cfg.distill_alpha == 0.5
        assert cfg.distill_temperature == 2.0
        assert cfg.quant_reg_beta == 0.01
        assert cfg.quaternary_c == 0.25

    def test_heal_kl_25m_preset(self):
        cfg = apply_smoke_preset(QAFTConfig(), "heal_kl_25m")
        assert cfg.max_steps == 6104
        assert cfg.distill_alpha == 0.5
        assert cfg.quant_reg_beta == 0.01
        assert cfg.skip_linear_attn is True
        assert cfg.lr_scheduler_type == "cosine"

    def test_heal_kl_50m_preset_and_horizon(self):
        cfg = apply_smoke_preset(QAFTConfig(), "heal_kl_50m")
        assert cfg.max_steps == 12207
        assert cfg.schedule_max_steps == 12207
        assert cfg.schedule_horizon_steps() == 12207
        assert cfg.distill_alpha == 0.5
        assert cfg.skip_linear_attn is True
        # Session A early stop must not shrink cosine horizon
        cfg.max_steps = 6104
        assert cfg.schedule_horizon_steps() == 12207
        warm, total = _optimizer_schedule_steps(cfg)
        assert total == 12207 // 8  # full 50M opt steps

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
        assert cfg.skip_linear_attn is False

    def test_heal_25m_preset(self):
        cfg = apply_smoke_preset(QAFTConfig(), "heal_25m")
        assert cfg.max_steps == 6104
        assert cfg.tokens_budget() == 6104 * 4096
        assert cfg.quant_warmup_steps == 256
        assert cfg.warmup_steps == 128
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.min_lr_ratio == 0.1
        assert cfg.skip_linear_attn is True
        assert cfg.quaternary_c == 0.25
        assert cfg.save_steps == 0

    def test_full_smoke_no_gdn_preset(self):
        cfg = apply_smoke_preset(QAFTConfig(), "full_smoke_no_gdn")
        assert cfg.max_steps == 1280
        assert cfg.skip_linear_attn is True
        assert cfg.lr_scheduler_type == "linear"

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


class TestDistillLoss:
    def test_kl_finite_and_zero_when_identical(self):
        logits = torch.randn(2, 8, 16)
        labels = torch.randint(0, 16, (2, 8))
        kl = distillation_kl_loss(logits, logits.clone(), labels, temperature=2.0)
        assert torch.isfinite(kl)
        assert float(kl) < 1e-4

    def test_kl_positive_when_different(self):
        torch.manual_seed(0)
        s = torch.randn(2, 8, 16)
        t = torch.randn(2, 8, 16)
        labels = torch.randint(0, 16, (2, 8))
        kl = distillation_kl_loss(s, t, labels, temperature=2.0)
        assert float(kl) > 0.0

    def test_trainer_requires_teacher_when_alpha_lt_one(self, tmp_path):
        model = _TinyLM()
        cfg = QAFTConfig(
            use_8bit_adam=False,
            use_bf16=False,
            gradient_checkpointing=False,
            output_dir=str(tmp_path),
            distill_alpha=0.5,
            max_steps=2,
        )
        try:
            QAFTTrainer(model, tokenizer=None, config=cfg, teacher=None)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "teacher" in str(e).lower()

    def test_trainer_ce_plus_reg_one_step(self, tmp_path):
        model = _TinyLM()
        # Attach a QuantizedLinear so commitment reg is non-trivial
        model.q = QuantizedLinear(16, 16, bias=False, c=0.25)
        cfg = QAFTConfig(
            use_8bit_adam=False,
            use_bf16=False,
            gradient_checkpointing=False,
            output_dir=str(tmp_path),
            distill_alpha=1.0,
            quant_reg_beta=0.1,
            max_steps=2,
            logging_steps=1,
            eval_steps=100,
            warmup_steps=0,
            quant_warmup_steps=0,
            gradient_accumulation_steps=1,
            metrics_filename="",
        )
        trainer = QAFTTrainer(model, tokenizer=None, config=cfg)
        ids = torch.randint(0, 32, (1, 8))
        batch = {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }
        loss, ce, kl, reg = trainer._compute_loss(batch)
        assert torch.isfinite(loss)
        assert ce > 0
        assert kl == 0.0
        assert reg >= 0.0
        loss.backward()

    def test_trainer_kl_one_step(self, tmp_path):
        student = _TinyLM()
        teacher = _TinyLM()
        cfg = QAFTConfig(
            use_8bit_adam=False,
            use_bf16=False,
            gradient_checkpointing=False,
            output_dir=str(tmp_path),
            distill_alpha=0.5,
            distill_temperature=2.0,
            quant_reg_beta=0.0,
            max_steps=2,
            logging_steps=1,
            eval_steps=100,
            warmup_steps=0,
            quant_warmup_steps=0,
            gradient_accumulation_steps=1,
            metrics_filename="",
        )
        trainer = QAFTTrainer(student, tokenizer=None, config=cfg, teacher=teacher)
        ids = torch.randint(0, 32, (1, 8))
        batch = {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }
        loss, ce, kl, reg = trainer._compute_loss(batch)
        assert torch.isfinite(loss)
        assert ce > 0
        assert kl >= 0.0
        loss.backward()


class TestDiskSafeCheckpoints:
    """Weights-only defaults, optional full save, prune, metrics JSONL."""

    def _tiny_trainer(self, tmp_path, **cfg_kw):
        model = _TinyLM()
        defaults = dict(
            use_8bit_adam=False,
            use_bf16=False,
            gradient_checkpointing=False,
            output_dir=str(tmp_path),
            save_optimizer=False,
            save_steps=0,
            max_steps=4,
            logging_steps=1,
            eval_steps=100,
            warmup_steps=0,
            quant_warmup_steps=0,
            gradient_accumulation_steps=1,
            metrics_filename="metrics.jsonl",
        )
        defaults.update(cfg_kw)
        cfg = QAFTConfig(**defaults)
        return QAFTTrainer(model, tokenizer=None, config=cfg), cfg

    def test_weights_only_default_omits_optimizer(self, tmp_path):
        trainer, _ = self._tiny_trainer(tmp_path)
        trainer.global_step = 3
        path = trainer._save_checkpoint("best")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" not in ckpt
        assert "scheduler_state_dict" not in ckpt
        assert ckpt.get("weights_only") is True
        assert ckpt["step"] == 3

    def test_full_save_includes_optimizer(self, tmp_path):
        trainer, _ = self._tiny_trainer(tmp_path, save_optimizer=True)
        trainer.global_step = 2
        path = trainer._save_checkpoint("final")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert ckpt.get("weights_only") is False

    def test_full_override_kwarg(self, tmp_path):
        trainer, _ = self._tiny_trainer(tmp_path, save_optimizer=False)
        path = trainer._save_checkpoint("resume", full=True)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "optimizer_state_dict" in ckpt

    def test_load_weights_only_skips_missing_opt(self, tmp_path):
        trainer, _ = self._tiny_trainer(tmp_path)
        trainer.global_step = 5
        trainer.best_perplexity = 12.5
        path = trainer._save_checkpoint("best")

        trainer2, _ = self._tiny_trainer(tmp_path)
        trainer2.load_checkpoint(path)
        assert trainer2.global_step == 5
        assert trainer2.best_perplexity == 12.5

    def test_load_full_restores_opt(self, tmp_path):
        trainer, _ = self._tiny_trainer(tmp_path, save_optimizer=True)
        trainer.global_step = 7
        path = trainer._save_checkpoint("final")
        trainer2, _ = self._tiny_trainer(tmp_path, save_optimizer=True)
        trainer2.load_checkpoint(path)
        assert trainer2.global_step == 7

    def test_prune_keeps_newest_n(self, tmp_path):
        trainer, cfg = self._tiny_trainer(tmp_path, max_step_checkpoints=2)
        for step in (10, 20, 30):
            trainer.global_step = step
            trainer._save_checkpoint(f"step_{step}")
        trainer._prune_step_checkpoints()
        names = sorted(os.listdir(cfg.output_dir))
        step_names = [n for n in names if n.startswith("checkpoint-step_")]
        assert step_names == ["checkpoint-step_20", "checkpoint-step_30"]

    def test_prune_zero_deletes_all_step(self, tmp_path):
        trainer, cfg = self._tiny_trainer(tmp_path, max_step_checkpoints=0)
        trainer.global_step = 10
        trainer._save_checkpoint("step_10")
        trainer._prune_step_checkpoints()
        step_names = [
            n for n in os.listdir(cfg.output_dir) if n.startswith("checkpoint-step_")
        ]
        assert step_names == []

    def test_metrics_jsonl_on_log_and_eval(self, tmp_path):
        trainer, cfg = self._tiny_trainer(
            tmp_path,
            max_steps=2,
            logging_steps=1,
            eval_steps=1,
            save_steps=0,
        )
        ids = torch.randint(0, 32, (1, 8))
        batch = {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }
        loader = [batch, batch]
        trainer.train(loader, eval_dataloader=loader)

        path = os.path.join(cfg.output_dir, "metrics.jsonl")
        assert os.path.isfile(path)
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert any(r.get("event") == "log" for r in rows)
        assert any(r.get("event") == "eval" and "perplexity" in r for r in rows)
        assert any(n.startswith("checkpoint-best") for n in os.listdir(cfg.output_dir))

    def test_save_steps_zero_default_in_config(self):
        cfg = QAFTConfig()
        assert cfg.save_steps == 0
        assert cfg.save_optimizer is False
        apply_smoke_preset(cfg, "full_smoke")
        assert cfg.save_steps == 0

    def test_train_loop_no_step_ckpts_when_save_steps_zero(self, tmp_path):
        trainer, cfg = self._tiny_trainer(
            tmp_path,
            max_steps=2,
            logging_steps=1,
            eval_steps=1000,
            save_steps=0,
        )
        ids = torch.randint(0, 32, (1, 8))
        batch = {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }
        loader = [batch, batch, batch]
        trainer.train(loader, eval_dataloader=None)
        names = os.listdir(cfg.output_dir)
        assert any(n == "checkpoint-final" for n in names)
        assert not any(n.startswith("checkpoint-step_") for n in names)
        # best only on eval improvement — no eval → no best required
        assert os.path.isfile(os.path.join(cfg.output_dir, "metrics.jsonl"))
