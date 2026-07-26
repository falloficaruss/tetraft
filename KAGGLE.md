# TetraFT — Kaggle checklist

**Empirical baselines & next architecture:** [`RESULTS.md`](RESULTS.md) (source of truth for numbers).

## Datasets

| Kaggle Dataset | Contents |
|----------------|----------|
| **`tetraft-code`** | Flat root `.py` + `notebooks/` |
| **`tetraft-fineweb-edu-50m`** | `train.jsonl`, `val.jsonl`, `sample_meta.json` |
| **Session B** | Session A `checkpoint-final` (**full** opt+sched) |

Zip code without `.venv` / `data/` / checkpoints. Refresh `tetraft-code` when modules change.

---

## A. FineWeb-Edu 50M sample ✅

CPU, Internet ON → `notebooks/build_fineweb_sample.ipynb` → Dataset `tetraft-fineweb-edu-50m`.

---

## B. Recorded runs (do not re-litigate)

| Run | ≈ tokens | Val PPL | Notes |
|-----|---------:|--------:|-------|
| Original | — | **~17.67** | FP baseline |
| Shock (skip GDN) | 0 | ~1.78e4 | milder |
| `full_smoke` + skip GDN | 5.2M | ~60.6 | CE 5M gate |
| `heal_25m` | 25.0M | ~48.2 | CE heal |
| **`heal_50m`** | **50.0M** | **~43.77** | **best CE** |
| **`scout_kl_5m`** | **5.2M** | **~49.31** | **KL PASS** vs 60.6 |

Stack: BF16 + AdamW8bit + grad ckpt, seq 512, batch 1, accum 8 → **4096 tok/step**.

**Sanity:** skip GDN → ~**96 eligible / 91 skipped**, ~**41%** quantized.

---

## C. Disk lesson

Defaults: weights-only best/final; `save_steps=0`.  
**Resume sessions:** `save_optimizer=True` → **one** full `checkpoint-final` only (several GB). No `step_*` spam.

---

## D. What to run next — `heal_kl_50m` (2 sessions)

Logical **one 50M KL run** (cosine over **12207** steps). Split for Kaggle wall time.

### Session A (~25M)

```bash
python run_smoke.py --preset heal_kl_50m \
  --max-steps 6104 \
  --save-optimizer \
  --train-data .../train.jsonl --val-data .../val.jsonl \
  --output-dir /kaggle/working/checkpoints_heal_kl_50m_A
```

- Fresh; measure orig + shock  
- End PPL go/no-go: ≲ **50–52** and falling → upload Dataset  
- Upload **`checkpoint-final`** (must be **full**, not weights-only)

### Session B (~25M → 50M)

```bash
python run_smoke.py --preset heal_kl_50m \
  --max-steps 12207 \
  --resume /kaggle/input/.../checkpoint-final \
  --skip-shock --skip-orig \
  --train-data .../train.jsonl --val-data .../val.jsonl \
  --output-dir /kaggle/working/checkpoints_heal_kl_50m_B
```

- Same preset (horizon stays 12207)  
- Final bar: CE **~43.77**; strong ≲ 35  

Notebook: `SESSION = "A"` or `"B"` in `notebooks/run_smoke.ipynb`.

| Preset | Status |
|--------|--------|
| scout_kl_5m | ✅ ~49.3 |
| heal_kl_50m A→B | **next** |

---

## Internet

| Step | Internet |
|------|----------|
| Build FineWeb / first model download | ON |
| Train after cache | OFF preferred |

---

## Phase status

| Phase | Status |
|-------|--------|
| CE heal 25/50M | ✅ ~48.2 / **~43.77** |
| KL scout 5M | ✅ **~49.3** PASS |
| **Next** | **heal_kl_50m Session A** → Dataset → **Session B** |
