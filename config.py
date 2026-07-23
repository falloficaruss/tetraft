from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class QAFTConfig:
    # Model
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    quaternary_c: float = 0.25

    # Quantization
    scale_mode: Literal["absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor"] = "absmean_channel"
    ste_mode: Literal["identity", "clip"] = "identity"
    quant_warmup_steps: int = 1000  # λ ramps from 0 → 1 over this many steps; 0 = hard quant from step 0

    # Training (Kaggle 0.8B defaults: small microbatch + accum)
    learning_rate: float = 2e-4
    batch_size: int = 1
    seq_length: int = 512
    max_steps: int = 5000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    # LR schedule: linear dies at 0 (ok for short smoke); cosine+floor for scale-up
    lr_scheduler_type: Literal["linear", "cosine"] = "linear"
    min_lr_ratio: float = 0.0  # floor = learning_rate * min_lr_ratio (cosine)

    # Logging / saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./checkpoints"

    # Memory (Phase 1 Kaggle recipe)
    gradient_checkpointing: bool = True
    use_bf16: bool = True
    use_8bit_adam: bool = True  # bitsandbytes AdamW8bit when available

    # Module policy
    skip_lm_head: bool = True
    skip_embed_tokens: bool = True
    skip_vision: bool = True
    skip_mtp: bool = True

    # Data (paths empty → resolve under ./data or /kaggle/input)
    train_data_path: str = ""
    val_data_path: str = ""
    data_text_field: str = "text"
    dataloader_num_workers: int = 0
    seed: int = 42

    # Optional HF revision pin for sample builds / docs
    data_hf_revision: Optional[str] = None

    def tokens_per_step(self) -> int:
        """Approx tokens processed per micro-step (batch × seq × accum)."""
        return (
            max(1, self.batch_size)
            * max(1, self.seq_length)
            * max(1, self.gradient_accumulation_steps)
        )

    def tokens_budget(self) -> int:
        """Approx total train tokens for ``max_steps`` micro-steps."""
        return self.max_steps * self.tokens_per_step()


# Token math @ defaults batch=1, seq=512, accum=8 → 4096 tokens / micro-step.
# max_steps is micro-steps (same counter as trainer.global_step).
SMOKE_PRESETS: Dict[str, dict] = {
    # --- Phase 1 / 1b smokes (linear LR → 0 is OK; short horizon) ---
    "short": {
        "max_steps": 200,
        "quant_warmup_steps": 100,
        "warmup_steps": 50,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 200,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
    },
    "longer": {
        "max_steps": 640,
        "quant_warmup_steps": 128,
        "warmup_steps": 64,
        "logging_steps": 20,
        "eval_steps": 128,
        "save_steps": 320,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
    },
    # 1280 × 4096 ≈ 5.24M — Phase 1b baseline (Kaggle: end PPL ~79.4)
    "full_smoke": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 640,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
    },
    # --- Phase 1c scale-up (cosine + LR floor; same quant recipe) ---
    # 6104 × 4096 ≈ 25.00M
    "scale_25m": {
        "max_steps": 6104,
        "quant_warmup_steps": 512,
        "warmup_steps": 256,
        "logging_steps": 50,
        "eval_steps": 512,
        "save_steps": 1024,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
    },
    # 12207 × 4096 ≈ 50.00M — optional if 25M still falling steeply
    "scale_50m": {
        "max_steps": 12207,
        "quant_warmup_steps": 1024,
        "warmup_steps": 512,
        "logging_steps": 50,
        "eval_steps": 1024,
        "save_steps": 2048,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
    },
}


def apply_smoke_preset(config: QAFTConfig, name: str) -> QAFTConfig:
    """Apply a named run preset onto *config* (mutates and returns it)."""
    if name not in SMOKE_PRESETS:
        raise ValueError(f"Unknown smoke preset {name!r}; choose from {sorted(SMOKE_PRESETS)}")
    for key, val in SMOKE_PRESETS[name].items():
        setattr(config, key, val)
    return config
