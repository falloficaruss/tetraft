# TetraFT — Kaggle checklist

**Empirical baselines & next architecture:** [`RESULTS.md`](RESULTS.md) (source of truth for numbers).

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

## B. Recorded runs (do not re-litigate)

| Run | ≈ tokens | Val PPL | Notes |
|-----|---------:|--------:|-------|
| Original | — | **~17.67** | FP baseline |
| Shock (all Linear) | 0 | ≫1e6 | λ=1, zero FT |
| Shock (skip GDN) | 0 | ~1.78e4 | milder |
| `short` | ~0.8M | ~472 | pipeline |
| `full_smoke` | 5.2M | ~79.4 | all Linear |
| `full_smoke` + skip GDN | 5.2M | ~60.6 | scope win |
| **`heal_25m`** | **25.0M** | **~48.2** | **best so far; after/orig ~2.73** |
| `scale_25m` (partial) | ~21M | ~68.6 | old DNA; not default |

Stack: BF16 + AdamW8bit + grad ckpt, seq 512, batch 1, accum 8 → **4096 tok/step**.

If `KeyError: qwen3_5`:  
`%pip install -U "git+https://github.com/huggingface/transformers.git"`

**Sanity before train:** with skip GDN, inventory should be ~**96 eligible / 91 skipped**, ~**41%** quantized — not 186/1 and 66%.

---

## C. Disk lesson (implemented)

`scale_25m` filled Kaggle working disk via repeated **full model + optimizer** checkpoints.

**Defaults now (in code):**
- Weights-only `best` / `final` (`save_optimizer=False`)
- `save_steps=0` — no periodic `step_*`; if enabled, prune via `max_step_checkpoints`
- Loss/PPL logged to `metrics.jsonl` under `output_dir` (not full state)
- Resume: pass `--save-optimizer` only when you need full state

**Ops:** clear output dir before long jobs.

---

## D. What to run next — `heal_50m`

**Locked recipe (c stays 0.25):**
- `skip_linear_attn=True` (GDN FP)
- λ_warmup=**256**, peak lr 2e-4
- Long run: **cosine + min_lr_ratio=0.1**
- Disk-safe checkpoints (weights-only)

| Preset | ≈ tokens | Status |
|--------|---------:|--------|
| `full_smoke_no_gdn` | 5.2M | done ~60.6 |
| `heal_25m` | 25M | **done ~48.2** |
| **`heal_50m`** | **50M** | **next — fresh start** (no resume from weights-only 25M) |

```bash
python run_smoke.py --preset heal_50m \
  --train-data /kaggle/input/.../train.jsonl \
  --val-data /kaggle/input/.../val.jsonl \
  --output-dir /kaggle/working/checkpoints_heal_50m
```

Notebook: set `PRESET = "heal_50m"`.

**Not recommended:** resume heal_25m weights-only as primary 50M; old `scale_25m`/`scale_50m`.

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
| 1 + 1b smoke | **COMPLETE** (5.2M → ~79 all-Linear) |
| 1c scale_25m | **PARTIAL** (obsolete DNA) |
| 2 scope | **COMPLETE** (skip GDN → ~60.6 @ 5.2M) |
| 2 length `heal_25m` | **COMPLETE** (**~48.2 @ 25M**, after/orig ~2.73) |
| **Next** | **`heal_50m` from scratch** |
| 3 main 100–200M | after 50M signal / recipe freeze |
