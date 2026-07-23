import logging
import os

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from eval import evaluate_perplexity, model_device
from quantize import QuantizedLinear

logger = logging.getLogger(__name__)


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


def _autocast_dtype(config, device: torch.device):
    """Return autocast dtype or ``None`` if autocast should be disabled."""
    if device.type != "cuda":
        return None
    if bool(getattr(config, "use_bf16", False)) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


class QAFTTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
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

        self.optimizer = _build_optimizer(self.model, config)

        # Avoid zero warmup/training steps when max_steps < accum.
        accum = max(1, config.gradient_accumulation_steps)
        num_training_steps = max(1, config.max_steps // accum)
        num_warmup_steps = min(
            max(0, config.warmup_steps // accum),
            max(0, num_training_steps - 1),
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        self.global_step = 0
        self.total_loss = 0.0
        self.best_perplexity = float("inf")
        self.metrics = {"step": [], "loss": [], "lr": [], "perplexity": [], "lambda": []}
        self._autocast_dtype = _autocast_dtype(config, self.device)

    def _set_lambda(self, lambda_val: float) -> None:
        for module in self.model.modules():
            if isinstance(module, QuantizedLinear):
                module.lambda_ = lambda_val

    def _current_lambda(self) -> float:
        if self.config.quant_warmup_steps > 0:
            return min(1.0, self.global_step / self.config.quant_warmup_steps)
        return 1.0

    def train(self, train_dataloader, eval_dataloader=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accum = max(1, self.config.gradient_accumulation_steps)

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
                    outputs = self.model(**batch)
                    loss = outputs.loss

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
                self.global_step += 1

                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = self.total_loss / self.config.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    self.metrics["step"].append(self.global_step)
                    self.metrics["loss"].append(avg_loss)
                    self.metrics["lr"].append(lr)
                    self.metrics["lambda"].append(lambda_val)
                    logger.info(
                        "Step %d: loss=%.4f, lr=%.2e, lambda=%.4f",
                        self.global_step,
                        avg_loss,
                        lr,
                        lambda_val,
                    )
                    self.total_loss = 0.0

                if (
                    eval_dataloader is not None
                    and self.global_step % self.config.eval_steps == 0
                ):
                    ppl = self._evaluate(eval_dataloader)
                    if ppl < self.best_perplexity:
                        self.best_perplexity = ppl
                        self._save_checkpoint("best")

                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

            if self.global_step >= self.config.max_steps:
                break

        self._save_checkpoint("final")

    def _evaluate(self, dataloader, max_batches=5):
        ppl = evaluate_perplexity(
            self.model, dataloader, max_batches=max_batches, device=self.device
        )
        self.metrics["perplexity"].append((self.global_step, ppl))
        logger.info("Step %d: perplexity=%.2f", self.global_step, ppl)
        self.model.train()
        return ppl

    def _save_checkpoint(self, tag):
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, f"checkpoint-{tag}")
        torch.save(
            {
                "step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_perplexity": self.best_perplexity,
                "config": self.config,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["step"]
        self.best_perplexity = ckpt.get("best_perplexity", float("inf"))
        logger.info("Checkpoint loaded: %s (step %d)", path, self.global_step)
