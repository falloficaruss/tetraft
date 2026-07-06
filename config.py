from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QAFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    quaternary_c: float = 0.25
    learning_rate: float = 1e-4
    batch_size: int = 4
    seq_length: int = 1024
    max_steps: int = 1000
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./checkpoints"
    gradient_checkpointing: bool = True
    skip_lm_head: bool = True
    skip_embed_tokens: bool = True
    quant_warmup: bool = True
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    dataloader_num_workers: int = 0
    seed: int = 42
