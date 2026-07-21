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

**Phase 0** — align quantizer / replace / config / tests with `RESEARCH.md`.  
See `RESEARCH_PLAN.md` §3 Phase 0 and §8.

## Repo layout

Flat modules at repository root (Kaggle-friendly):

- `quantize.py` — quaternary quant + `QuantizedLinear`
- `model.py` — linear replace + skips
- `train.py` — `QAFTTrainer`
- `eval.py` — perplexity
- `config.py` — `QAFTConfig`
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
