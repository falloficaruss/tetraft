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

**Next science:** `RESULTS.md` §5.4–5.5:

| Option | What | Gate |
|--------|------|------|
| **1 α/T scout** | Fresh `scout_kl_5m` + `--distill-alpha` / `--distill-temperature` | &lt; **49.31** |
| **2 polish** | preset `polish_kl_5m`, notebook `SESSION=P` | &lt; **34.38** |

### D.1 Polish (`polish_kl_5m`) — ready

Attach B `checkpoint-final` (weights-only OK) + refresh `tetraft-code`.

```bash
python run_smoke.py --preset polish_kl_5m \
  --resume /kaggle/input/.../checkpoint-final \
  --skip-shock --skip-orig \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_polish_kl_5m
```

| Knob | Value |
|------|--------|
| stop | **13487** (= 12207 + 1280) |
| lr | **2e-5** constant (`cosine`, `min_lr_ratio=1.0`, `warmup=0`) |
| sched horizon | **1280** (fresh Adam; not 12207) |
| KL | α=0.5 T=2 β=0.01 |

**Sanity:** `resumed_step=12207`, lr≈2e-5 flat, λ=1, ~96 eligible. **`max_steps` &gt; 12207** or train no-ops.

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
| **Next** | paper hygiene / α–T / optional longer KL (`RESULTS.md` §5.3) |
