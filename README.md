# TetraFT

Quantization-Aware Fine-Tuning for 2-bit Quaternary LLMs.

## Quickstart

```bash
pip install -e .
```

Then open and run the notebook:

```bash
jupyter notebook notebooks/qaft_demo.ipynb
```

Or open it directly in [Google Colab](https://colab.research.google.com/) — just
clone the repo first:

```
!git clone https://github.com/YOUR_USER/tetraft.git
%cd tetraft
!pip install -e .
```

### Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## What it does

1. Loads a pre-trained LLM (default: Qwen2.5-0.5B)
2. Replaces all `nn.Linear` with `QuantizedLinear` using a 4-state quaternary
   grid `{-1, -c, c, 1}` (c = 0.25 or 0.5)
3. Fine-tunes using Straight-Through Estimator with gradient clipping
4. Evaluates perplexity recovery

## Project

- `PLAN.md` — VRAM budgeting & hardware planning
- `RESEARCH.md` — Mathematical formulation
- `AGENTS.md` — Agent instructions for OpenCode
