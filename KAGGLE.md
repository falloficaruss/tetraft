# TetraFT — Kaggle checklist (Phase 1 / 1b)

## Datasets

| Kaggle Dataset | Contents | How |
|----------------|----------|-----|
| **`tetraft-code`** | Flat root `.py` (+ optional `notebooks/`) | Upload zip; refresh when code changes |
| **`tetraft-fineweb-edu-50m`** | `train.jsonl`, `val.jsonl`, `sample_meta.json` | Build notebook output (below) |

Do **not** put FineWeb text in git.

### `tetraft-code` upload

```
config.py
data.py
eval.py
model.py
quantize.py
run_smoke.py
train.py
notebooks/build_fineweb_sample.ipynb
notebooks/run_smoke.ipynb
```

Zip from repo root (no `.venv`, no `data/`, no checkpoints).

---

## A. Build FineWeb-Edu 50M sample ✅

1. Notebook → **CPU**, **Internet ON**, attach **`tetraft-code`**.
2. Run `notebooks/build_fineweb_sample.ipynb` (~1–3 h).
3. Save output → Dataset **`tetraft-fineweb-edu-50m`**.

---

## B. Phase 1 short smoke ✅ (baseline recorded)

Engineering exit **met** on Kaggle (Qwen3.5-0.8B-Base, FineWeb-Edu 50M val):

| Metric | Approx. value |
|--------|----------------|
| Original val PPL | **~17.7** |
| Shock PPL (λ=1, 0 FT) | **≫ 1e6** (treat as broken, not calibrated) |
| ~200 steps (~0.8M tok), λ→1 | eval PPL **~472**, loss **finite** |
| Stack | BF16 + AdamW8bit + grad ckpt, seq 512, batch 1, accum 8 |

```bash
python run_smoke.py --preset short \
  --train-data .../train.jsonl --val-data .../val.jsonl \
  --output-dir /kaggle/working/checkpoints
```

**Installs:** recent `transformers` (needs `qwen3_5`), `accelerate`, `bitsandbytes`.  
If `KeyError: qwen3_5`:  
`%pip install -U "git+https://github.com/huggingface/transformers.git"`

---

## C. Phase 1b — longer smoke (next)

Same datasets, **GPU**. Goal: recovery curve in the **1–5M token** band; gap still vs **original ~17.7**.

Token math: `batch × seq × accum = 1 × 512 × 8 = 4096` tokens/step.

| Preset | `max_steps` | ≈ tokens | λ warmup | LR warmup |
|--------|------------:|---------:|---------:|----------:|
| `longer` | 640 | ~2.6M | 128 | 64 |
| `full_smoke` | 1280 | ~5.2M | 256 | 128 |

```bash
# Recommended next run
python run_smoke.py --preset longer \
  --train-data /kaggle/input/.../train.jsonl \
  --val-data /kaggle/input/.../val.jsonl \
  --output-dir /kaggle/working/checkpoints

# Or upper smoke band
python run_smoke.py --preset full_smoke ...
```

Notebook: set `PRESET = "longer"` in `notebooks/run_smoke.ipynb`.

**Watch in `smoke_results.json`:**
- `ppl_original`, `ppl_shock`, `ppl_after_smoke`
- `tokens_seen`, `steps_ran`, `loss_finite`
- `train_metrics.perplexity` vs steps (should trend down after λ=1)

**Success (1b):** finite loss; post-smoke PPL **&lt; ~472** and still falling; no requirement to hit 17.7 yet.

**Wall time (rough):** short was ~8–15 min train; `longer` ~3× (~30–45 min); `full_smoke` ~6× (~1–1.5 h). Depends on GPU.

---

## Internet policy

| Step | Internet |
|------|----------|
| Build FineWeb sample | ON |
| First model download / transformers upgrade | ON |
| Reproducible train after cache | OFF preferred |

---

## Phase status

| Phase | Status |
|-------|--------|
| **1** short smoke + data + memory recipe | **COMPLETE** |
| **1b** longer smoke 1–5M tokens | **NEXT** |
| **2** recipe ablations on fixed 50M | after 1b curve |
