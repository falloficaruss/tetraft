import os
import logging

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from quantize import QuantizedLinear
from eval import evaluate_perplexity

logger = logging.getLogger(__name__)


class QAFTTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        self.global_step = 0
        self.total_loss = 0.0
        self.best_perplexity = float("inf")
        self.metrics = {"step": [], "loss": [], "lr": [], "perplexity": []}

    def train(self, train_dataloader, eval_dataloader=None):
        self.model.train()
        self.optimizer.zero_grad()

        for _ in range(self.config.num_epochs):
            for batch in train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.config.quant_warmup:
                    lambda_val = min(4.0 * self.global_step / self.config.max_steps, 1.0)
                    for module in self.model.modules():
                        if isinstance(module, QuantizedLinear):
                            module.lambda_ = lambda_val

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                self.scheduler.step()

                self.total_loss += loss.item() * self.config.gradient_accumulation_steps
                self.global_step += 1

                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = self.total_loss / self.config.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    self.metrics["step"].append(self.global_step)
                    self.metrics["loss"].append(avg_loss)
                    self.metrics["lr"].append(lr)
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                    self.total_loss = 0.0

                if eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                    ppl = self._evaluate(eval_dataloader)
                    if ppl < self.best_perplexity:
                        self.best_perplexity = ppl
                        self._save_checkpoint("best")

                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

        self._save_checkpoint("final")

    def _evaluate(self, dataloader, max_batches=5):
        ppl = evaluate_perplexity(self.model, dataloader, max_batches=max_batches)
        self.metrics["perplexity"].append((self.global_step, ppl))
        logger.info(f"Step {self.global_step}: perplexity={ppl:.2f}")
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
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["step"]
        self.best_perplexity = ckpt.get("best_perplexity", float("inf"))
        logger.info(f"Checkpoint loaded: {path} (step {self.global_step})")
