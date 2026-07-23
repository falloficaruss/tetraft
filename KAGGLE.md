# TetraFT — Kaggle checklist

## Datasets

| Kaggle Dataset | Contents |
|----------------|----------|
| **`tetraft-code`** | Flat root `.py` + `notebooks/` |
| **`tetraft-fineweb-edu-50m`** | `train.jsonl`, `val.jsonl`, `sample_meta.json` |

Zip code without `.venv` / `data/` / checkpoints. Refresh `tetraft-code` when modules change.

---

## A. FineWeb-Edu 50M sample ✅

CPU, Internet ON → `notebooks/build_fineweb_sample.ipynb` → Dataset `tetraft-fineweb-edu-50m`.

---

## B. Phase 1 / 1b smokes ✅

| Run | Tokens | Val PPL (end, λ=1) |
|-----|--------|-------------------:|
| Original | — | **~17.67** |
| Shock (0 FT) | 0 | **≫ 1e6** |
| `short` | ~0.82M | ~472 |
| `full_smoke` | **~5.24M** | **~79.4** (gap ≈ 4.5× orig) |

Stack: BF16 + AdamW8bit + grad ckpt, seq 512, batch 1, accum 8.

If `KeyError: qwen3_5`:  
`%pip install -U "git+https://github.com/huggingface/transformers.git"`

---

## C. Phase 1c — scale-up (NEXT)

Same recipe (`c=0.25`, `absmean_channel`, identity STE). **Cosine LR** with **min_lr_ratio=0.1** (no die-to-zero).

Token math: `1 × 512 × 8 = 4096` tokens/micro-step.

| Preset | Steps | ≈ tokens | λ warmup | LR warmup | Wall (rough) |
|--------|------:|---------:|---------:|----------:|--------------|
| **`scale_25m`** | 6104 | **~25.0M** | 512 | 256 | ~4–5× full_smoke (~3–4 h if full_smoke ~50 min) |
| `scale_50m` | 12207 | ~50.0M | 1024 | 512 | ~2× scale_25m |

```bash
python run_smoke.py --preset scale_25m \
  --train-data /kaggle/input/.../train.jsonl \
  --val-data /kaggle/input/.../val.jsonl \
  --output-dir /kaggle/working/checkpoints
```

Notebook: `PRESET = "scale_25m"` in `notebooks/run_smoke.ipynb`.

**Watch**
- `ppl_after_smoke` vs **79.4**
- `train_metrics.perplexity` after λ=1 (step ≥ 512)
- end LR ≈ `0.1 * 2e-4` (not 0)
- `after/orig` (target → 1.0)

**Success (1c):** PPL **&lt; 79**, finite loss.  
**Gate → Phase 2:** still falling at 25M → optional `scale_50m` **or** start ablations; plateau ≫ 17.7 → ablations sooner.

---

## D. Phase 2 — recipe search (after 1c)

Fixed data order; one factor at a time vs control (= 1c recipe).  
Scout ~10M tok; confirm winners longer. Factors: `c`, scale, λ schedule, STE, module scope (GDN).  
Harness TBD after `scale_25m` results.

---

## Internet policy

| Step | Internet |
|------|----------|
| Build FineWeb / first model download | ON |
| Train after cache | OFF preferred |

---

## Phase status

| Phase | Status |
|-------|--------|
| 1 short smoke + data + memory | **COMPLETE** |
| 1b 1–5M smoke | **COMPLETE** (5.2M → PPL ~79) |
| **1c scale_25m** | **NEXT** |
| 2 ablations | after 1c |
| 3 main 100–200M | after recipe freeze |
