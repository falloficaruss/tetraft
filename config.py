from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class QAFTConfig:
    # Model
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    quaternary_c: float = 0.25

    # Quantization
    scale_mode: Literal["absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor"] = "absmean_channel"
    ste_mode: Literal["identity", "clip", "trust"] = "identity"
    # Soft trust STE (ste_mode=trust): m = clip(1 - e/(T*s), 0, 1); s→∞ → identity
    trust_softness: float = 1.0
    quant_warmup_steps: int = 1000  # λ ramps from 0 → 1 over this many steps; 0 = hard quant from step 0

    # Recovery objective (α=1, β=0 → pure CE, backward-compatible)
    # L = α·CE + (1-α)·T²·KL(teacher ‖ student) + β·||W - sg(Q(W))||²
    distill_alpha: float = 1.0  # 1.0 = CE only; <1 enables teacher KL
    distill_temperature: float = 2.0
    quant_reg_beta: float = 0.0  # commitment / grid regularizer weight

    # Training (Kaggle 0.8B defaults: small microbatch + accum)
    learning_rate: float = 2e-4
    batch_size: int = 1
    seq_length: int = 512
    max_steps: int = 5000
    # LR/λ horizon in micro-steps. None → use max_steps. Set equal to full-run
    # length when Session A stops early (e.g. max_steps=6104, schedule_max_steps=12207)
    # so cosine matches a single uninterrupted job after resume.
    schedule_max_steps: Optional[int] = None
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    # LR schedule: linear dies at 0 (ok for short smoke); cosine+floor for scale-up
    lr_scheduler_type: Literal["linear", "cosine"] = "linear"
    min_lr_ratio: float = 0.0  # floor = learning_rate * min_lr_ratio (cosine)

    # Logging / saving (disk-safe defaults for Kaggle)
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 0  # 0 = no periodic step_* dumps; best/final still saved
    save_optimizer: bool = False  # True = full model+opt+sched (resume); False = weights-only
    max_step_checkpoints: int = 1  # prune older step_* when > 0 and save_steps > 0
    metrics_filename: str = "metrics.jsonl"  # under output_dir; empty = disable file log
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
    # Phase 2 scope ablation: leave Qwen3.5 GDN (path ``linear_attn``) in FP
    skip_linear_attn: bool = False

    # Bundle adapters (R3/R4/R5) — defaults off; see scout_kl_bundle_r345_5m
    pre_rms: bool = False  # R3: RMSNorm before each QuantizedLinear (γ=1 init)
    weight_calib: Literal["none", "unit_absmean"] = "none"  # R4: one-shot W reshape at replace
    lora_rank: int = 0  # R5: 0 = off
    lora_alpha: Optional[float] = None  # None → equal to lora_rank when rank > 0

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

    def schedule_horizon_steps(self) -> int:
        """Micro-steps used for LR schedule length (full run when resuming)."""
        h = self.schedule_max_steps
        if h is None or int(h) <= 0:
            return max(1, int(self.max_steps))
        return max(1, int(h))


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
        "save_steps": 0,
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
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
    },
    # 1280 × 4096 ≈ 5.24M — Phase 1b baseline (Kaggle: end PPL ~79.4 all-Linear)
    "full_smoke": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "skip_linear_attn": False,
    },
    # Scope scout: same as full_smoke + GDN FP (Kaggle: end PPL ~60.6)
    "full_smoke_no_gdn": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "skip_linear_attn": True,
    },
    # --- Phase 2 heal scale-up (full_smoke DNA + skip GDN + cosine floor) ---
    # Do NOT use old scale_25m (λw=512, all-Linear) as mainline.
    # 6104 × 4096 ≈ 25.00M
    "heal_25m": {
        "max_steps": 6104,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 50,
        "eval_steps": 512,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
    },
    # 12207 × 4096 ≈ 50.00M — done ~43.8 PPL (after/orig ~2.48)
    "heal_50m": {
        "max_steps": 12207,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 50,
        "eval_steps": 1024,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
    },
    # --- Phase 2b: KL + quant-reg (matched λw=256 vs full_smoke_no_gdn ~60.6) ---
    # Gate: end PPL < 60.6 at ~5.24M before longer KL heals. Do NOT lengthen λ here.
    "scout_kl_5m": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.5,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
    },
    # Longer KL heal (use after scout_kl_5m beats ~60.6)
    "heal_kl_25m": {
        "max_steps": 6104,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 50,
        "eval_steps": 512,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.5,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
    },
    # Full 50M KL (same DNA as heal_50m + scout KL). Two-session resume:
    #   Session A: --max-steps 6104 --save-optimizer  (cosine still over 12207)
    #   Session B: --resume checkpoint-final --max-steps 12207 --skip-shock
    "heal_kl_50m": {
        "max_steps": 12207,
        "schedule_max_steps": 12207,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 50,
        "eval_steps": 1024,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.5,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
    },
    # Bundle smoke R3+R4+R5 @ ~5.24M — ❌ FAIL (PPL 1000+ @ λ→1; do not rerun).
    # Suspected R4 unit_absmean. Kept for replay only.
    "scout_kl_bundle_r345_5m": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.5,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
        "pre_rms": True,
        "weight_calib": "unit_absmean",
        "lora_rank": 8,
        "lora_alpha": 8.0,
    },
    # R5-only @ ~5.24M (gate < scout_kl_5m ~49.31). Fresh start. No R3/R4.
    "scout_kl_r5_5m": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.5,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
        "pre_rms": False,
        "weight_calib": "none",
        "lora_rank": 8,
        "lora_alpha": 8.0,
    },
    # Soft trust STE @ ~5.24M (RESULTS.md §5.10.1). Match scout_kl_5m DNA;
    # only change ste_mode=trust. Gate: end PPL < best 5M (scout_kl_r5_5m ~48.38).
    # No LoRA / pre_rms / calib. Abort @λ=1 PPL ≫ 500.
    "scout_kl_trust_5m": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.5,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
        "ste_mode": "trust",
        "trust_softness": 1.0,
        "pre_rms": False,
        "weight_calib": "none",
        "lora_rank": 0,
    },
    # Trust + more teacher KL (α=0.3). Confounds two knobs vs pure trust;
    # use when chasing best 5M PPL rather than clean STE attribution.
    # Gate still < 48.38. T stays 2.0 (locked).
    "scout_kl_trust_a03_5m": {
        "max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 40,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "min_lr_ratio": 0.0,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.3,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
        "ste_mode": "trust",
        "trust_softness": 1.0,
        "pre_rms": False,
        "weight_calib": "none",
        "lora_rank": 0,
    },
    # Long heal: soft trust + α=0.3 @ ~400M (16 × 6104 ≈ 25M sessions).
    # Cosine horizon locked at 97664. Session k: --max-steps k*6104;
    # k>1 resume full ckpt + --skip-shock --skip-orig; save_optimizer S1–S15.
    # DNA locked from scout_kl_trust_a03_5m PASS (~43.34 @ 5M). No LoRA.
    "heal_kl_trust_400m": {
        "max_steps": 97664,
        "schedule_max_steps": 97664,
        "quant_warmup_steps": 256,
        "warmup_steps": 128,
        "logging_steps": 50,
        "eval_steps": 1024,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.3,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
        "ste_mode": "trust",
        "trust_softness": 1.0,
        "pre_rms": False,
        "weight_calib": "none",
        "lora_rank": 0,
        "save_optimizer": True,
    },
    # Polish from heal_kl_50m B weights (~5.24M more tokens). Weights-only resume
    # rebuilds Adam+sched from 0; schedule_max_steps = polish length (not 13487).
    #   --resume B/checkpoint-final --skip-shock --skip-orig
    # stop = resumed_step(12207) + 1280 = 13487; gate PPL < 34.38
    "polish_kl_5m": {
        "max_steps": 13487,
        "schedule_max_steps": 1280,
        "quant_warmup_steps": 256,
        "warmup_steps": 0,
        "logging_steps": 50,
        "eval_steps": 256,
        "save_steps": 0,
        "learning_rate": 2e-5,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 1.0,
        "skip_linear_attn": True,
        "quaternary_c": 0.25,
        "distill_alpha": 0.5,
        "distill_temperature": 2.0,
        "quant_reg_beta": 0.01,
        "save_optimizer": False,
    },
    # --- Historical only (worse early DNA; do not default) ---
    # 6104 × 4096 ≈ 25.00M
    "scale_25m": {
        "max_steps": 6104,
        "quant_warmup_steps": 512,
        "warmup_steps": 256,
        "logging_steps": 50,
        "eval_steps": 512,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "skip_linear_attn": False,
    },
    # 12207 × 4096 ≈ 50.00M
    "scale_50m": {
        "max_steps": 12207,
        "quant_warmup_steps": 1024,
        "warmup_steps": 512,
        "logging_steps": 50,
        "eval_steps": 1024,
        "save_steps": 0,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "min_lr_ratio": 0.1,
        "skip_linear_attn": False,
    },
}


def apply_smoke_preset(config: QAFTConfig, name: str) -> QAFTConfig:
    """Apply a named run preset onto *config* (mutates and returns it)."""
    if name not in SMOKE_PRESETS:
        raise ValueError(f"Unknown smoke preset {name!r}; choose from {sorted(SMOKE_PRESETS)}")
    for key, val in SMOKE_PRESETS[name].items():
        setattr(config, key, val)
    return config
