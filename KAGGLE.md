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
| **`scout_kl_5m`** | **5.2M** | **~49.31** | KL PASS vs 60.6 |
| **`heal_kl_50m` A** | **25M** | **~48.65** | full ckpt; go/no-go PASS |
| **`heal_kl_50m` A+B** | **50M** | **~34.38** | **best; after/orig ~1.95** |
| **`polish_kl_5m`** | +5.2M | ≥34.38 | ❌ FAIL — stop |

Stack: BF16 + AdamW8bit + grad ckpt, seq 512, batch 1, accum 8 → **4096 tok/step**.

**Sanity:** skip GDN → ~**96 eligible / 91 skipped**, ~**41%** quantized.

---

## C. Disk lesson

Defaults: weights-only best/final; `save_steps=0`.  
**Resume sessions:** `save_optimizer=True` → **one** full `checkpoint-final` only (several GB). No `step_*` spam.  
**heal_kl_50m A→B resume worked** (resumed_step=6104 → 12207).

---

## D. Completed — `heal_kl_50m` (2 sessions) ✅

| Session | PPL | Notes |
|---------|----:|-------|
| A @ 25M | **48.65** | full final ckpt |
| B @ 50M | **34.38** | beat CE 43.77; after/orig **1.95** |

Replay (if needed): notebook `SESSION=A|B`; see `RESULTS.md` §2.6.

**Next science:** `RESULTS.md` §5.9 **R5-only** LoRA scout.  
α/T null; polish FAIL; bundle R345 FAIL; D0 done (flat).

| Option | What | Gate | Status |
|--------|------|------|--------|
| **D0 layer map** | `run_layer_map.py` on B | role table | ✅ done |
| **Bundle R345** | R3+R4+R5 | &lt; 49.31 | ❌ FAIL — stop |
| **R5-only** | `scout_kl_r5_5m` LoRA r=8 | &lt; **49.31** | **next** |
| **D1 KL 100M** | fresh same DNA | &lt; **30** | length floor |
| α/T scout | static α/T @ 5M | &lt; 49.31 | ⚪ null |
| polish | `polish_kl_5m` | &lt; 34.38 | **FAIL** |

### D.0b R5-only scout ← **run this**

Attach **code + FineWeb only**. Refresh `tetraft-code`. Notebook `SESSION="R5"`.

```bash
python run_smoke.py --preset scout_kl_r5_5m \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_scout_kl_r5_5m
```

Sanity: log `pre_rms=False weight_calib=none lora_rank=8`; @λ=1 PPL not ≫500. Gate end &lt; **49.31**.

### D.1 D0 layer map ← **run this**

Attach **code + FineWeb + B `checkpoint-final`**. Refresh `tetraft-code`.

```bash
# Full D0 (PPL + FP-mask top-8)
python run_layer_map.py \
  --checkpoint /kaggle/input/.../checkpoint-final \
  --preset heal_kl_50m \
  --val-data /kaggle/input/tetraft-fineweb-edu-50m/val.jsonl \
  --max-eval-batches 20 \
  --fp-mask-topk 8 \
  --output-dir /kaggle/working/layer_map_b

# Weight-only if VRAM tight
python run_layer_map.py \
  --checkpoint /kaggle/input/.../checkpoint-final \
  --preset heal_kl_50m --skip-ppl \
  --output-dir /kaggle/working/layer_map_b
```

Download `layer_map_summary.json` + paste role table / `suggest=` into `RESULTS.md`.

| Artifact | Contents |
|----------|----------|
| `layer_map.csv` / `.json` | per-module rel_l2, mae/γ, bins |
| `layer_map_summary.json` | by_role, top_modules, ppl, suggest |

### D.1b α/T scout (historical; not priority)

Attach **code + FineWeb only** (no ckpt). Static α/T null — do not re-run unless needed.

```bash
python run_smoke.py --preset scout_kl_5m \
  --distill-alpha 0.3 --distill-temperature 2.0 \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_scout_kl_a03_t2
```

| Cell | α | T | Tag |
|------|---|---|-----|
| **first** | **0.3** | 2.0 | `a03_t2` |
| next | 0.5 | 1.0 | `a05_t1` |
| only if both help | 0.3 | 1.0 | `a03_t1` |

**Sanity:** fresh start; shock ~1.78e4; ~96 eligible; teacher on; end PPL (20 batch) vs **49.31**.  
**If PASS:** lock α/T → fresh long KL (not polish-only on old B).

### D.2 Polish (`polish_kl_5m`) — FAIL

Did not beat **34.38**. Do **not** extend polish. Replay only if debugging.

### D.3 Future Muon 5M

Documented in `RESULTS.md` §5.7. Still AdamW8bit in code. No Kaggle job until implemented.

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
| KL 50M A+B | ✅ **~34.38** (after/orig ~1.95) |
| polish_kl_5m | ❌ FAIL (≥34.38) — stop |
| **Next** | **α/T scout** `a03_t2` (`RESULTS.md` §5.4) |
