# TetraFT

Quantization-Aware Fine-Tuning for 2-bit Quaternary LLMs.

## Quickstart (Google Colab)

1. Push this repo to `YOUR_USERNAME/tetraft` on GitHub
2. Open `notebooks/qaft_demo.ipynb` in [Google Colab](https://colab.research.google.com)
3. In cell 1, replace `YOUR_USERNAME` with your GitHub username
4. Select **Runtime → Change runtime type → T4 GPU**
5. **Runtime → Run all**

See `COLAB_GUIDE.md` for detailed instructions.

### Local Development

```bash
pip install pytest
pytest tests/ -v
jupyter notebook notebooks/qaft_demo.ipynb
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
- `COLAB_GUIDE.md` — Running on Google Colab
