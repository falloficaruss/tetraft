# TetraFT

**Quantization-Aware Fine-Tuning** for **2-bit quaternary** LLMs.

Convert a pretrained model to weights on \(\{-1,-c,c,1\}\), then heal so performance approaches the **original** full-precision model.

## Research docs (read in order)

| File | Contents |
|------|----------|
| [`RESEARCH.md`](RESEARCH.md) | Mathematical formulation & method (code must match) |
| [`RESULTS.md`](RESULTS.md) | **Kaggle baselines + what to do next architecturally** |
| [`PLAN.md`](PLAN.md) | Models, FineWeb-Edu data, VRAM, Kaggle |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | Phases, experiments, paper plan |
| [`KAGGLE.md`](KAGGLE.md) | Kaggle checklist / disk lesson |
| [`AGENTS.md`](AGENTS.md) | Conventions for coding agents |

## Locked defaults

- **First model:** `Qwen/Qwen3.5-0.8B-Base` → later `Qwen3.5-2B-Base`
- **Grid:** \(c=0.25\), scale `absmean_channel`
- **Data:** fixed FineWeb-Edu sample (Kaggle Dataset; not in repo)
- **Platform:** Kaggle
- **Success:** gap to original model (not BitNet comparison)

## Current step

**Read [`RESULTS.md`](RESULTS.md).**  

| Milestone | Val PPL |
|-----------|--------:|
| Original | ~17.7 |
| full_smoke + skip GDN ~5.2M | ~60.6 |
| **scout_kl_5m** ~5.2M | **~49.3** |
| heal_25m / heal_50m CE | ~48.2 / **~43.8** |

**Next:** `heal_kl_50m` two-session resume (A: 25M full ckpt → B: resume to 50M). See `KAGGLE.md`.

```python
from config import QAFTConfig
from model import replace_from_config

config = QAFTConfig()
replace_from_config(model, config)
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
