import json
import logging
import math
import os
import re
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from eval import evaluate_perplexity, model_device
from quantize import QuantizedLinear, quant_bin_stats, quant_commitment_loss

logger = logging.getLogger(__name__)


def _shift_logits_labels(logits: torch.Tensor, labels: torch.Tensor):
    """Causal LM shift: predict token t from position t-1."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Token-mean KL(teacher ‖ student) on causal next-token positions (ignore -100)."""
    t = max(float(temperature), 1e-8)
    s_logits, s_labels = _shift_logits_labels(student_logits, labels)
    t_logits, _ = _shift_logits_labels(teacher_logits, labels)

    vocab = s_logits.size(-1)
    s_flat = s_logits.view(-1, vocab)
    t_flat = t_logits.view(-1, vocab).detach()
    lab_flat = s_labels.view(-1)
    mask = lab_flat != -100
    if not bool(mask.any()):
        return s_flat.sum() * 0.0

    s_flat = s_flat[mask]
    t_flat = t_flat[mask]
    log_p_s = torch.nn.functional.log_softmax(s_flat / t, dim=-1)
    p_t = torch.nn.functional.softmax(t_flat / t, dim=-1)
    # kl_div expects log-input; mean over tokens then scale by T^2 (Hinton)
    kl = torch.nn.functional.kl_div(log_p_s, p_t, reduction="batchmean")
    return kl * (t * t)

_STEP_CKPT_RE = re.compile(r"^checkpoint-step_(\d+)$")


def _build_optimizer(model, config):
    """AdamW, preferring bitsandbytes 8-bit when ``config.use_8bit_adam``."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    use_8bit = bool(getattr(config, "use_8bit_adam", False))
    if use_8bit:
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=config.learning_rate)
            logger.info("Using bitsandbytes AdamW8bit optimizer")
            return opt
        except Exception as e:
            logger.warning(
                "use_8bit_adam=True but bitsandbytes AdamW8bit unavailable (%s); "
                "falling back to torch AdamW",
                e,
            )

    logger.info("Using torch.optim.AdamW")
    return AdamW(optimizer_grouped_parameters, lr=config.learning_rate)


def _optimizer_schedule_steps(config):
    """Convert micro-step budgets to optimizer-step counts (scheduler.step units).

    Trainer calls ``scheduler.step()`` once per optimizer step, i.e. every
    ``gradient_accumulation_steps`` micro-steps. ``max_steps`` / ``warmup_steps``
    in config are micro-steps (same as ``global_step``).
    """
    accum = max(1, config.gradient_accumulation_steps)
    num_training_steps = max(1, config.max_steps // accum)
    num_warmup_steps = min(
        max(0, config.warmup_steps // accum),
        max(0, num_training_steps - 1),
    )
    return num_warmup_steps, num_training_steps


def _cosine_with_min_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> float:
    """Warmup → cosine decay down to ``min_lr_ratio`` (not necessarily 0)."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step >= num_training_steps:
        return float(min_lr_ratio)
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _build_scheduler(optimizer, config):
    """Linear (→0) or cosine (→ min_lr_ratio * base_lr) schedule over optimizer steps."""
    num_warmup_steps, num_training_steps = _optimizer_schedule_steps(config)
    sched_type = getattr(config, "lr_scheduler_type", "linear") or "linear"
    min_lr_ratio = float(getattr(config, "min_lr_ratio", 0.0) or 0.0)
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)

    if sched_type == "cosine":
        lr_lambda = partial(
            _cosine_with_min_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )
        sched = LambdaLR(optimizer, lr_lambda)
        logger.info(
            "LR schedule=cosine warmup_opt_steps=%d total_opt_steps=%d min_lr_ratio=%.3f",
            num_warmup_steps,
            num_training_steps,
            min_lr_ratio,
        )
        return sched

    if sched_type != "linear":
        logger.warning("Unknown lr_scheduler_type=%r; using linear", sched_type)

    sched = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info(
        "LR schedule=linear warmup_opt_steps=%d total_opt_steps=%d",
        num_warmup_steps,
        num_training_steps,
    )
    return sched


def _autocast_dtype(config, device: torch.device):
    """Return autocast dtype or ``None`` if autocast should be disabled."""
    if device.type != "cuda":
        return None
    if bool(getattr(config, "use_bf16", False)) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


class QAFTTrainer:
    def __init__(self, model, tokenizer, config, teacher=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.teacher = teacher
        self.device = model_device(model)
        # If the model is still on CPU but CUDA is free, move it (non-sharded case).
        if self.device.type == "cpu" and torch.cuda.is_available():
            if not hasattr(model, "hf_device_map") or model.hf_device_map is None:
                self.model.to("cuda")
                self.device = torch.device("cuda")

        if config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

        alpha = float(getattr(config, "distill_alpha", 1.0))
        if alpha < 1.0 and teacher is None:
            raise ValueError(
                f"distill_alpha={alpha} < 1 requires a frozen teacher model"
            )
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            if hasattr(self.teacher, "config"):
                self.teacher.config.use_cache = False
            # Keep teacher on same device when not sharded
            t_dev = model_device(self.teacher)
            if (
                t_dev.type == "cpu"
                and self.device.type == "cuda"
                and (not hasattr(self.teacher, "hf_device_map") or self.teacher.hf_device_map is None)
            ):
                self.teacher.to(self.device)

        self.optimizer = _build_optimizer(self.model, config)
        self.scheduler = _build_scheduler(self.optimizer, config)

        self.global_step = 0
        self.total_loss = 0.0
        self.total_ce = 0.0
        self.total_kl = 0.0
        self.total_reg = 0.0
        self.best_perplexity = float("inf")
        self.metrics = {
            "step": [],
            "loss": [],
            "ce": [],
            "kl": [],
            "reg": [],
            "lr": [],
            "perplexity": [],
            "lambda": [],
        }
        self._autocast_dtype = _autocast_dtype(config, self.device)
        self._metrics_path = self._resolve_metrics_path()

    def _resolve_metrics_path(self) -> Optional[str]:
        name = getattr(self.config, "metrics_filename", "metrics.jsonl") or ""
        name = str(name).strip()
        if not name:
            return None
        return os.path.join(self.config.output_dir, name)

    def _set_lambda(self, lambda_val: float) -> None:
        for module in self.model.modules():
            if isinstance(module, QuantizedLinear):
                module.lambda_ = lambda_val

    def _current_lambda(self) -> float:
        if self.config.quant_warmup_steps > 0:
            return min(1.0, self.global_step / self.config.quant_warmup_steps)
        return 1.0

    def _append_metrics_row(self, row: Dict[str, Any]) -> None:
        """Append one JSONL metrics row (loss/PPL/λ) without full model state."""
        if self._metrics_path is None:
            return
        os.makedirs(os.path.dirname(self._metrics_path) or ".", exist_ok=True)
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def train(self, train_dataloader, eval_dataloader=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(1, self.config.gradient_accumulation_steps)
        save_steps = int(getattr(self.config, "save_steps", 0) or 0)

        for _ in range(self.config.num_epochs):
            for batch in train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                batch = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }

                lambda_val = self._current_lambda()
                self._set_lambda(lambda_val)

                if self._autocast_dtype is not None:
                    cm = torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
                else:
                    from contextlib import nullcontext

                    cm = nullcontext()

                with cm:
                    loss, ce_v, kl_v, reg_v = self._compute_loss(batch)

                if accum > 1:
                    loss = loss / accum

                loss.backward()

                if (self.global_step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                self.total_loss += loss.item() * accum
                self.total_ce += ce_v
                self.total_kl += kl_v
                self.total_reg += reg_v
                self.global_step += 1

                if self.global_step % self.config.logging_steps == 0:
                    nlog = self.config.logging_steps
                    avg_loss = self.total_loss / nlog
                    avg_ce = self.total_ce / nlog
                    avg_kl = self.total_kl / nlog
                    avg_reg = self.total_reg / nlog
                    lr = self.scheduler.get_last_lr()[0]
                    self.metrics["step"].append(self.global_step)
                    self.metrics["loss"].append(avg_loss)
                    self.metrics["ce"].append(avg_ce)
                    self.metrics["kl"].append(avg_kl)
                    self.metrics["reg"].append(avg_reg)
                    self.metrics["lr"].append(lr)
                    self.metrics["lambda"].append(lambda_val)
                    logger.info(
                        "Step %d: loss=%.4f ce=%.4f kl=%.4f reg=%.4f lr=%.2e lambda=%.4f",
                        self.global_step,
                        avg_loss,
                        avg_ce,
                        avg_kl,
                        avg_reg,
                        lr,
                        lambda_val,
                    )
                    self._append_metrics_row(
                        {
                            "event": "log",
                            "step": self.global_step,
                            "loss": avg_loss,
                            "ce": avg_ce,
                            "kl": avg_kl,
                            "reg": avg_reg,
                            "lr": lr,
                            "lambda": lambda_val,
                        }
                    )
                    self.total_loss = 0.0
                    self.total_ce = 0.0
                    self.total_kl = 0.0
                    self.total_reg = 0.0

                if (
                    eval_dataloader is not None
                    and self.global_step % self.config.eval_steps == 0
                ):
                    ppl = self._evaluate(eval_dataloader)
                    if ppl < self.best_perplexity:
                        self.best_perplexity = ppl
                        self._save_checkpoint("best")
                    self._append_metrics_row(
                        {
                            "event": "eval",
                            "step": self.global_step,
                            "perplexity": ppl,
                            "best_perplexity": self.best_perplexity,
                            "lambda": self._current_lambda(),
                        }
                    )

                if save_steps > 0 and self.global_step % save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
                    self._prune_step_checkpoints()

            if self.global_step >= self.config.max_steps:
                break

        self._save_checkpoint("final")

    def _compute_loss(self, batch):
        """CE (+ optional teacher KL + quant commitment). Returns loss and scalar parts."""
        alpha = float(getattr(self.config, "distill_alpha", 1.0))
        alpha = min(max(alpha, 0.0), 1.0)
        beta = float(getattr(self.config, "quant_reg_beta", 0.0) or 0.0)
        temperature = float(getattr(self.config, "distill_temperature", 2.0) or 2.0)

        outputs = self.model(**batch)
        ce = outputs.loss
        if ce is None:
            raise RuntimeError("Student model did not return loss; batch must include labels")

        if alpha < 1.0:
            if self.teacher is None:
                raise RuntimeError("distill_alpha < 1 but trainer.teacher is None")
            t_kwargs = {"input_ids": batch["input_ids"]}
            if batch.get("attention_mask") is not None:
                t_kwargs["attention_mask"] = batch["attention_mask"]
            with torch.no_grad():
                t_out = self.teacher(**t_kwargs)
            kl = distillation_kl_loss(
                outputs.logits, t_out.logits, batch["labels"], temperature=temperature
            )
            loss = alpha * ce + (1.0 - alpha) * kl
        else:
            kl = ce * 0.0
            loss = ce

        if beta > 0.0:
            reg = quant_commitment_loss(self.model)
            loss = loss + beta * reg
        else:
            reg = ce * 0.0

        return loss, float(ce.detach()), float(kl.detach()), float(reg.detach())

    def _evaluate(self, dataloader, max_batches=5):
        ppl = evaluate_perplexity(
            self.model, dataloader, max_batches=max_batches, device=self.device
        )
        self.metrics["perplexity"].append((self.global_step, ppl))
        logger.info("Step %d: perplexity=%.2f", self.global_step, ppl)
        if self.global_step % max(self.config.eval_steps, 1) == 0:
            try:
                stats = quant_bin_stats(self.model)
                if stats.get("n", 0) > 0:
                    frac = stats.get("frac", {})
                    logger.info(
                        "Step %d: quant bins frac -1=%.3f -c=%.3f +c=%.3f +1=%.3f (n=%d)",
                        self.global_step,
                        frac.get("-1", 0.0),
                        frac.get("-c", 0.0),
                        frac.get("+c", 0.0),
                        frac.get("+1", 0.0),
                        stats["n"],
                    )
                    self._append_metrics_row(
                        {
                            "event": "bins",
                            "step": self.global_step,
                            "bin_frac": frac,
                            "bin_n": stats["n"],
                        }
                    )
            except Exception as e:
                logger.warning("quant_bin_stats failed: %s", e)
        self.model.train()
        return ppl

    def _save_checkpoint(self, tag: str, *, full: Optional[bool] = None) -> str:
        """Save checkpoint. Default is weights-only (no Adam); set full=True to resume."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"checkpoint-{tag}")
        include_opt = (
            bool(getattr(self.config, "save_optimizer", False))
            if full is None
            else bool(full)
        )
        payload: Dict[str, Any] = {
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_perplexity": self.best_perplexity,
            "config": self.config,
            "weights_only": not include_opt,
        }
        if include_opt:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, path)
        kind = "full" if include_opt else "weights-only"
        logger.info("Checkpoint saved (%s): %s", kind, path)
        return path

    def _prune_step_checkpoints(self) -> None:
        """Keep at most ``max_step_checkpoints`` newest ``checkpoint-step_*`` files."""
        keep = int(getattr(self.config, "max_step_checkpoints", 1) or 0)
        if keep < 0:
            return
        out = self.config.output_dir
        if not os.path.isdir(out):
            return
        found = []
        for name in os.listdir(out):
            m = _STEP_CKPT_RE.match(name)
            if m:
                found.append((int(m.group(1)), os.path.join(out, name)))
        found.sort(key=lambda x: x[0])
        # keep == 0 means drop all periodic step dumps after write (extreme disk mode)
        to_delete = found if keep == 0 else found[:-keep]
        for _, path in to_delete:
            try:
                os.remove(path)
                logger.info("Pruned step checkpoint: %s", path)
            except OSError as e:
                logger.warning("Failed to prune %s: %s", path, e)

    def load_checkpoint(self, path: str, *, load_optimizer: Optional[bool] = None) -> None:
        """Load weights; optimizer/scheduler only if present and requested."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.global_step = ckpt["step"]
        self.best_perplexity = ckpt.get("best_perplexity", float("inf"))

        has_opt = "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt
        want_opt = has_opt if load_optimizer is None else bool(load_optimizer)
        if want_opt and has_opt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            logger.info(
                "Checkpoint loaded (full): %s (step %d)", path, self.global_step
            )
        else:
            if load_optimizer and not has_opt:
                logger.warning(
                    "Checkpoint %s has no optimizer state; loaded weights only", path
                )
            logger.info(
                "Checkpoint loaded (weights-only): %s (step %d)",
                path,
                self.global_step,
            )
