from dataclasses import dataclass
from typing import Literal


@dataclass
class QAFTConfig:
    # Model
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    quaternary_c: float = 0.25

    # Quantization
    scale_mode: Literal["absmean_channel", "absmean_tensor", "absmax_channel", "absmax_tensor"] = "absmean_channel"
    ste_mode: Literal["identity", "clip"] = "identity"
    quant_warmup_steps: int = 1000  # λ ramps from 0 → 1 over this many steps; 0 = hard quant from step 0

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    seq_length: int = 1024
    max_steps: int = 5000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    num_epochs: int = 1

    # Logging / saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./checkpoints"

    # Memory
    gradient_checkpointing: bool = True

    # Module policy
    skip_lm_head: bool = True
    skip_embed_tokens: bool = True
    skip_vision: bool = True
    skip_mtp: bool = True

    # Data
    dataloader_num_workers: int = 0
    seed: int = 42
