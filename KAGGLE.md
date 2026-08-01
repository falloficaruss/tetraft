# TetraFT — Kaggle checklist

**Empirical baselines & next architecture:** [`RESULTS.md`](RESULTS.md) (source of truth for numbers).

## Datasets

| Kaggle Dataset | Contents |
|----------------|----------|
| **`tetraft-code`** | Flat root `.py` + `notebooks/` + `scripts/` |
| **`tetraft-fineweb-edu-400m`** | Marathon train/val (build with `SAMPLE=400m`) |
| **`tetraft-fineweb-edu-50m`** | Legacy 50M sample |
| **`tetraft-heal-kl-trust-400m`** | Growing run pack: ledger + `sessions/Sxx` + ckpt |

Zip code without `.venv` / `data/` / checkpoints. Refresh `tetraft-code` when modules change.

---

## A. FineWeb samples

| Sample | Notebook `SAMPLE=` | Dataset name |
|--------|--------------------|--------------|
| 50M (legacy) | `50m` | `tetraft-fineweb-edu-50m` ✅ |
| **400M (marathon)** | **`400m`** | **`tetraft-fineweb-edu-400m`** ← build first |

CPU, Internet ON → `notebooks/build_fineweb_sample.ipynb`.

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
**Token pack cache:** train loader tokenizes into a flat int32 memmap (~1.6 GB per 400M) under  
`/kaggle/working/tetraft_pack_cache` (override: `--pack-cache-dir` / `TETRAFT_PACK_CACHE`).  
Keep the dir attached between hops to skip rebuild; a rebuild is batched-tokenizer only (no 400M in RAM).

---

## D. Completed — `heal_kl_50m` (2 sessions) ✅

| Session | PPL | Notes |
|---------|----:|-------|
| A @ 25M | **48.65** | full final ckpt |
| B @ 50M | **34.38** | beat CE 43.77; after/orig **1.95** |

Replay (if needed): notebook `SESSION=A|B`; see `RESULTS.md` §2.6.

**Mainline:** **`heal_kl_trust_400m`** (trust+α=0.3, 16×25M). Scout ✅ **~43.34** @ 5M.

| Option | What | Gate | Status |
|--------|------|------|--------|
| **Trust+α0.3 scout** | `scout_kl_trust_a03_5m` | &lt; 48.38 | ✅ **~43.34** |
| **R5-only** | `scout_kl_r5_5m` | &lt; 49.31 | ✅ **48.38** |
| **400M marathon** | `heal_kl_trust_400m` S1–S16 | S1&lt;48.65 · S2&lt;34.38 | **next** |
| Bundle R345 / polish | — | — | FAIL |

### D.0 Marathon — `heal_kl_trust_400m` ← **run this**

**DNA:** trust s=1.0, α=0.3, T=2, β=0.01, skip GDN, cosine horizon **97664**, no LoRA.

| Attach | When |
|--------|------|
| code + `tetraft-fineweb-edu-400m` | every hop |
| + `tetraft-heal-kl-trust-400m` | S2–S16 (prior pack) |

Notebook: `SESSION = 1` … `16` only. After each hop publish  
`/kaggle/working/heal_kl_trust_400m/` → Dataset **`tetraft-heal-kl-trust-400m`**.

| Session | stop steps | ≈ tok | save_optimizer | Gate |
|--------:|----------:|------:|:--------------:|------|
| 1 | 6104 | 25M | full | &lt; 48.65 |
| 2 | 12208 | 50M | full | &lt; 34.38 |
| 4 | 24416 | 100M | full | &lt; 30 |
| 16 | 97664 | 400M | weights OK | stretch ≲ 23 |

Pack layout: `ledger.jsonl`, `LATEST.json`, `sessions/Sxx/{session_summary,metrics,smoke_results,checkpoint-final}`.

```bash
# CLI equivalent for session k (example k=1)
python run_smoke.py --preset heal_kl_trust_400m \
  --max-steps 6104 --save-optimizer \
  --run-session 1 --run-pack-dir /kaggle/working/heal_kl_trust_400m \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/train_heal_kl_trust_400m_S01
```

Paper figures (local after S16):

```bash
python scripts/merge_run_pack.py path/to/heal_kl_trust_400m
python scripts/plot_heal_kl_trust_400m.py path/to/heal_kl_trust_400m
```

### D.0b Soft trust scout ✅ done (~43.34)

`scout_kl_trust_a03_5m`: post-smoke **43.34** (gate 48.38). DNA locked into 400M.

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
