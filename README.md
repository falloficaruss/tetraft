# TetraFT

**Quantization-Aware Fine-Tuning** for **2-bit quaternary** LLMs.

Convert a pretrained model to weights on \(\{-1,-c,c,1\}\), then heal so performance approaches the **original** full-precision model.

## Research docs (read in order)

| File | Contents |
|------|----------|
| [`RESEARCH.md`](RESEARCH.md) | Mathematical formulation & method (code must match) |
| [`PLAN.md`](PLAN.md) | Models, FineWeb-Edu data, VRAM, Kaggle |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | Phases, experiments, paper plan — **start at Phase 0** |
| [`AGENTS.md`](AGENTS.md) | Conventions for coding agents |

## Locked defaults

- **First model:** `Qwen/Qwen3.5-0.8B-Base` → later `Qwen3.5-2B-Base`
- **Grid:** \(c=0.25\), scale `absmean_channel`
- **Data:** fixed FineWeb-Edu sample (Kaggle Dataset; not in repo)
- **Platform:** Kaggle
- **Success:** gap to original model (not BitNet comparison)

## Current step

**Phase 1 complete.** **Next: Phase 1b** longer smoke (`--preset longer` ≈ 2.6M tokens).  
Short-smoke baseline (Kaggle): orig PPL ~17.7 → shock ≫1e6 → ~0.8M tok PPL ~472.  
See `RESEARCH_PLAN.md` §3 and **[`KAGGLE.md`](KAGGLE.md)**.

```python
from config import QAFTConfig
from model import replace_from_config

config = QAFTConfig()
replace_from_config(model, config)
```

### Kaggle

```bash
# Phase 1b longer smoke
python run_smoke.py --preset longer \
  --train-data /kaggle/input/.../train.jsonl \
  --val-data /kaggle/input/.../val.jsonl \
  --output-dir /kaggle/working/checkpoints

# Presets: short (~0.8M) | longer (~2.6M) | full_smoke (~5.2M)
# tokens/step = batch × seq × accum (default 1×512×8 = 4096)
```

## Repo layout

Flat modules at repository root (Kaggle-friendly):

- `quantize.py` — quaternary quant + `QuantizedLinear`
- `model.py` — linear replace + skips + inventory
- `train.py` — `QAFTTrainer` (BF16, 8-bit Adam optional)
- `eval.py` — perplexity
- `config.py` — `QAFTConfig`
- `data.py` — FineWeb-Edu sample builder + packed dataloaders
- `run_smoke.py` — Phase 1 entry (inventory → orig/shock PPL → short QAFT)
- `KAGGLE.md` — upload / build / smoke checklist
- `notebooks/` — Kaggle glue only (not source of truth)
- `tests/` — unit tests

## Local checks

```bash
pip install pytest torch
pytest tests/ -v
```

## What it does (pipeline)

1. Load pretrained LLM (start: Qwen3.5-0.8B-Base)
2. Replace eligible `nn.Linear` with `QuantizedLinear` (quaternary forward)
3. QAFT with STE + optional \(\lambda\) anneal on FineWeb-Edu CPT sample
4. Evaluate recovery vs the **original** model (held-out PPL, later downstream)
