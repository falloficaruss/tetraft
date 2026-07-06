# TetraFT — AGENTS.md

## Project

Quantization-Aware Fine-Tuning (QAFT) for 2-bit quaternary LLMs. See `PLAN.md` (VRAM/hardware planning) and `RESEARCH.md` (mathematical formulation).

## Repository Structure

Files are **flat at root** (no package directory) — required by Kaggle Dataset flattening.

| File | Purpose |
|---|---|
| `quantize.py` | `QuantizeFunction` (autograd STE) + `QuantizedLinear` (nn.Module) |
| `model.py` | `replace_linear_layers()` — walks modules, swaps `nn.Linear` → `QuantizedLinear` |
| `train.py` | `QAFTTrainer` — training loop, checkpointing, AMP, cosine LR |
| `eval.py` | `evaluate_perplexity()` — no-grad PPL on a DataLoader |
| `config.py` | `QAFTConfig` dataclass |
| `notebooks/qaft_demo.ipynb` | Self-contained Colab notebook entrypoint |
| `tests/` | `test_quantize.py`, `test_model.py` |

## Key Design Decisions

- **Notebook-first interface.** The library modules are flat (.py files at root). Import directly: `from config import QAFTConfig`.
- **Raw PyTorch training.** No HF `Trainer`. `QAFTTrainer` uses `AdamW` + linear warmup schedule + `GradScaler`.
- **Quaternary grid** `{-1, -c, c, 1}`, default `c=0.5`. Adjustable per layer via `QuantizedLinear(c=...)`.
- **STE backward** clips gradients for `|W/γ| > 1.0` (saturated weights frozen).
- **`lm_head` skipped** by default (`QuantizedLinear` never replaces it). Controlled by `skip_lm_head` flag.
- **Model loaded in FP32** on CPU, layers replaced, then moved to GPU. Ensures master weights are FP32.

## Commands

```bash
# run tests (root must be on Python path)
pip install pytest
pytest tests/ -v

# launch notebook
jupyter notebook notebooks/qaft_demo.ipynb
```

On Colab: upload repo folder to Drive → open notebook → run all.

## Architecture Notes

- `replace_linear_layers()` is recursive and idempotent. Already-replaced `QuantizedLinear` layers are left untouched.
- Gradient checkpointing enabled via `model.gradient_checkpointing_enable()` inside `QAFTTrainer.__init__`.
- Checkpoints are plain `torch.save` dicts (step, model/opt/scheduler states, best PPL, config).
- The 0.5B model fits on T4 (15 GB) with FP32 AdamW at batch_size=2, seq_len=512, grad_accum=4.
