# TetraFT — AGENTS.md

Instructions for coding agents working in this repository.

## Project

**TetraFT:** Quantization-Aware Fine-Tuning (QAFT) converting pretrained LLMs to **2-bit quaternary** weights \(\{-1,-c,c,1\}\), healing toward **original model quality**.

| Doc | Read for |
|-----|----------|
| `RESEARCH.md` | Math & method (must match code) |
| `PLAN.md` | Models, FineWeb-Edu data, VRAM, Kaggle |
| `RESULTS.md` | **Frozen Kaggle PPL baselines + architectural next steps** |
| `RESEARCH_PLAN.md` | Phases; **Phase 2 next** |
| `KAGGLE.md` | Kaggle ops / disk lesson |

## Current focus

**Phase 1+1b done; 1c partial (recorded).** Read **`RESULTS.md`** before changing train recipe.  
Baselines: orig ~17.7; full_smoke 5.2M→~79; scale_25m ~21M→~69 (less efficient early).  
**Next (architecture):** disk-safe ckpts → Phase 2 ablations (scope/GDN, \(c\), scale) with **full_smoke DNA** (λw≈256).  
Do not prioritize BitNet, blind scale_50m, 2B, or notebook rewrites unless asked.

## Entry pattern

```python
from config import QAFTConfig
from model import replace_from_config

config = QAFTConfig()
# model = AutoModelForCausalLM.from_pretrained(config.model_name, ...)
replace_from_config(model, config)
```

## Repository structure

Files are **flat at root** (no package dir) — required for Kaggle Dataset flattening.

| File | Purpose |
|------|---------|
| `quantize.py` | Quaternary quant + `QuantizedLinear` (STE, \(\lambda\), scale modes) |
| `model.py` | `replace_linear_layers()` + skip policy + inventory |
| `train.py` | `QAFTTrainer` (BF16, 8-bit Adam, \(\lambda\) anneal) |
| `eval.py` | `evaluate_perplexity()` |
| `config.py` | `QAFTConfig` dataclass (defaults = plan defaults) |
| `data.py` | FineWeb-Edu sample builder + packed JSONL dataloaders |
| `run_smoke.py` | Phase 1 smoke: inventory → orig/shock PPL → short QAFT |
| `tests/` | unit tests |
| `notebooks/` | Optional glue only — **not** source of truth |

## Locked design decisions

- **Grid:** `{-1, -c, c, 1}`, default **`c=0.25`**
- **Scale default:** `absmean_channel` (per `RESEARCH.md`); ablations via config
- **STE:** identity first; optional clip mode
- **\(\lambda\) anneal:** supported; trainer may ramp 0→1
- **Skip:** `lm_head`, embeddings, vision, norms
- **First model:** `Qwen/Qwen3.5-0.8B-Base` then `Qwen3.5-2B-Base`
- **Data:** FineWeb-Edu fixed sample (external Kaggle Dataset; not in repo)
- **Parity target:** original FP model — not BitNet
- **Quaternary-optimal:** do not clone BitNet schedules/APIs without reason
- **Raw PyTorch training** — no HF `Trainer`
- **Imports:** `from config import QAFTConfig` (flat modules)

## Implementation rules

1. Keep `RESEARCH.md` and code consistent; if you change math, update `RESEARCH.md` in the same change.
2. Single default for `c` and `scale_mode` across `config.py`, `QuantizedLinear`, and `replace_linear_layers`.
3. Prefer config-driven knobs over hardcoding.
4. Tests required for quant bins, scale modes, \(\lambda\in\{0,1\}\), and lm_head skip.
5. Do not add large data files to git.
6. Do not expand scope past the current phase unless the user asks.

## Commands

```bash
# tests (repo root on PYTHONPATH)
pip install pytest torch --quiet
pytest tests/ -v
```

## Architecture notes

- `replace_linear_layers()` is recursive and idempotent (`QuantizedLinear` left untouched).
- Gradient checkpointing is enabled in `QAFTTrainer` when configured.
- Checkpoints: `torch.save` dicts (step, model/opt/scheduler, metrics, config).
- Kaggle: BF16 + 8-bit Adam + checkpointing for 0.8B; see `PLAN.md`.
