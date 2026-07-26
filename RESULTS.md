# TetraFT — Empirical results & architectural next steps

**Model:** `Qwen/Qwen3.5-0.8B-Base`  
**Val:** held-out FineWeb-Edu (`tetraft-fineweb-edu-50m`)  
**Stack:** BF16, AdamW8bit, grad checkpointing, seq 512, batch 1, accum 8  
**→ 4096 tokens / micro-step**  
**Quant defaults (unless noted):** \(c=0.25\), `absmean_channel`, STE `identity`, λ anneal on, skip lm_head / embeds / vision / mtp  

Primary metric: **val PPL vs original FP** (not BitNet).  
Math/method: `RESEARCH.md`. Phases: `RESEARCH_PLAN.md`. Kaggle ops: `KAGGLE.md`.

---

## 1. Frozen baselines

| Run | Schedule (high level) | ≈ tokens | Val PPL | after/orig |
|-----|------------------------|---------:|--------:|-----------:|
| Original FP | — | — | **17.67** | **1.0** |
| Shock (all eligible Linear) | λ=1, zero FT | 0 | **≫ 1e6** | — (broken; not calibrated) |
| Shock (`skip_linear_attn`) | λ=1, zero FT; GDN FP | 0 | **~1.78e4** | ~1009× |
| `short` | short QAFT smoke | ~0.82M | ~**472** | — |
| **`full_smoke`** | all Linear; λw=**256**, linear→0, lr 2e-4 | **5.24M** | **~79.4** | **~4.5×** |
| **`full_smoke` + `skip_linear_attn`** | GDN FP; same short schedule | **5.24M** | **~60.6** | **~3.43×** |
| **`heal_25m`** | GDN FP; λw=**256**, **cosine+0.1**, lr 2e-4 | **25.0M** | **~48.2** | **~2.73×** |
| **`heal_50m`** | same heal DNA | **50.0M** | **~43.77** | **~2.48** |
| **`scout_kl_5m`** | skip GDN; λw=256; **KL α=0.5 T=2 β=0.01**; linear | **5.24M** | **~49.31** | **~2.79** |
| **`scale_25m`** (partial) | all Linear; λw=**512**, cosine+0.1 | **~21.0M** (disk stop) | **~68.6** | **~3.9×** |

**Inventory — all eligible:** ~187 Linear; ~186 eligible; ~**66%** quantized (~498M / ~752M).

**Inventory — heal DNA (`skip_linear_attn`):** ~**96** eligible / **91** skipped; ~**41%** quantized (~308M / ~752M).

Do **not** treat all-Linear shock PPL as a continuous quality score. Skip-GDN shock is finite but still far from original.

---

## 2. Recovery curves

### 2.1 `full_smoke` (~5.24M) — all eligible Linear

| Step | ≈ tokens | PPL |
|-----:|---------:|----:|
| 256 (λ→1) | ~1.0M | ~388 |
| 512 | ~2.1M | ~160 |
| 768 | ~3.1M | ~133 |
| 1024 | ~4.2M | ~100 |
| 1280 final | ~5.2M | **~79.4** |

### 2.2 `full_smoke` + `skip_linear_attn` (~5.24M) ✅ scope win

| Step | ≈ tokens | PPL |
|-----:|---------:|----:|
| 256 (λ→1) | ~1.0M | ~169 |
| 512 | ~2.1M | ~102 |
| 768 | ~3.1M | ~92 |
| 1024 | ~4.2M | ~73 |
| 1280 mid | ~5.2M | ~66.9 |
| **final** | **5.24M** | **~60.6** |

after/orig **~3.43**. vs all-Linear same budget: 79.4 → 60.6.

### 2.3 `heal_25m` (~25.0M) ✅ length scale win

Kaggle 2026-07-24. Preset `heal_25m`: skip GDN, λw=256, lr_warmup=128, cosine min_lr_ratio=0.1, c=0.25, disk-safe weights-only.  
Completed full 6104 steps (no disk OOM).

| Step | ≈ tokens | PPL |
|-----:|---------:|----:|
| 512 | ~2.1M | ~117 |
| 1024 | ~4.2M | ~124 |
| 1536 | ~6.3M | ~111 |
| 2048 | ~8.4M | ~97 |
| 2560 | ~10.5M | ~86 |
| 3072 | ~12.6M | ~76 |
| 3584 | ~14.7M | ~68 |
| 4096 | ~16.8M | ~65 |
| 4608 | ~18.9M | ~61 |
| 5120 | ~21.0M | ~56 |
| 5632 | ~23.1M | ~55 |
| **6104 final** | **25.0M** | **~48.2** |

Loss after λ=1 ~5.0 → ~3.8. LR ended on floor (~2e-5). after/orig **~2.73**.  
vs scout skip-GDN @ 5.2M: **60.6 → 48.2** (~20% relative).  
vs old `scale_25m` ~69 @ ~21M (all-Linear, λw=512): **clearly better** at similar/higher budget.

**Note:** mid-run eval at ~4M tok (~124) was briefly worse than ~2M (~117); curve recovered and kept falling through 25M. Still improving late (56→55→**48**).

**Resume:** final ckpt is **weights-only** (no Adam). Do **not** treat as seamless resume into 50M — run longer jobs **from scratch**.

### 2.4 `heal_50m` (~50.0M) ✅ new CE best; diminishing returns

Kaggle 2026-07-25. Preset `heal_50m`: same heal DNA, fresh start (12207 steps).  
Shock **17839** (ratio ~1009) matches skip-GDN baseline.

| Step | ≈ tokens | PPL |
|-----:|---------:|----:|
| 1024 | ~4.2M | 117.4 |
| 2048 | ~8.4M | 108.1 |
| 3072 | ~12.6M | 90.2 |
| 4096 | ~16.8M | 80.8 |
| 5120 | ~21.0M | 74.9 |
| 6144 | ~25.2M | 68.8 |
| 7168 | ~29.4M | 63.8 |
| 8192 | ~33.6M | 58.7 |
| 9216 | ~37.7M | 54.3 |
| 10240 | ~41.9M | 52.1 |
| 11264 | ~46.1M | 51.0 |
| **12207 final** | **50.0M** | **43.77** |

Loss after λ=1 ~5.06 → ~3.47. after/orig **~2.48**.  
vs `heal_25m` final: **48.2 → 43.77** (~9% relative) at **2×** tokens — length still helps, returns shrinking.  
**Schedule note:** at ~25M tokens mid-run PPL was **~68.8**, not ~48 — longer cosine keeps LR higher at matched step; compare **end-of-run**, not matched step alone.  
Late drop 51→44 is steep vs prior slope; treat **43.77** as logged final (optional cold re-eval of `checkpoint-final`).

**CE-only length is no longer the primary lever** toward parity (~1.3×).

### 2.5 `scout_kl_5m` (~5.24M) ✅ KL gate PASS

Kaggle 2026-07-26. Matched CE skip-GDN schedule (λw=256, linear→0) + α=0.5 CE/KL, T=2, β=0.01.  
Shock 17839; inventory 96 eligible / 41%.

| Step | ≈ tokens | PPL (in-train ~5 batch) |
|-----:|---------:|------------------------:|
| 256 (λ→1) | ~1.0M | 123.4 |
| 512 | ~2.1M | 85.8 |
| 768 | ~3.1M | 73.6 |
| 1024 | ~4.2M | 61.7 |
| 1280 mid | ~5.2M | 56.8 |
| **final (20 batch)** | **5.24M** | **49.31** |

after/orig **~2.79**. vs CE skip-GDN ~60.6: **PASS** (~18.6% relative).  
≈ CE `heal_25m` (~48.2) at **~5× fewer tokens**. Bins healthy (~30% ±1, ~20% ±c).  
`reg≈0` in logs (commitment tiny vs CE/KL). Still falling at end → scale KL.

### 2.6 `scale_25m` (partial) — historical; wrong DNA

Interrupted ~21M for disk (full opt dumps). End ~68.6. Obsolete vs heal DNA.

---

## 3. Locked takeaways

1. **Method works.** Shock → finite recovery; loss stays finite.
2. **Best CE:** **`heal_50m` → ~43.77 @ 50M** (after/orig **~2.48**).
3. **Best KL short:** **`scout_kl_5m` → ~49.31 @ 5.2M** — beats CE 5M gate hard.
4. **Scope:** skip GDN locked (~41% quantized).
5. **CE length diminishing;** KL is the recovery lever now.
6. **Keep \(c=0.25\)**, λw=**256** on KL scale-up.
7. **Next:** **`heal_kl_50m` two-session resume** (cosine horizon 12207).

---

## 4. What we did *not* settle

| Question | Status |
|----------|--------|
| **`heal_kl_50m` end PPL @ 50M** | **Next** (Session A then B) |
| Mid-25M KL PPL (Session A gate) | TBD |
| Longer λ schedule | Deferred |
| FP CPT control | Recommended paper control |

---

## 5. Current recipe

### 5.1 CE heal DNA (frozen baseline)

| Knob | Value |
|------|--------|
| Best CE | **`heal_50m` → ~43.77 @ 50M** |

### 5.2 KL recovery DNA (scout proven)

| Knob | Value |
|------|--------|
| α / T / β | **0.5** / **2.0** / **0.01** |
| λ_warmup | **256** |
| Scope | skip GDN, c=0.25 |
| Scout | **`scout_kl_5m` → ~49.31** ✅ |
| Scale-up | **`heal_kl_50m`** (schedule_max_steps=**12207**) |

### 5.3 Next runs — two-session 50M KL

**Logical one job:** cosine over 12207 steps. Split for Kaggle time.

**Session A** (~25M): stop early, **full** ckpt  
```bash
python run_smoke.py --preset heal_kl_50m \
  --max-steps 6104 --save-optimizer \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_heal_kl_50m_A
```
Go/no-go: mid PPL ≲ **50–52** → upload `checkpoint-final` (full) as Dataset.

**Session B** (→50M): resume  
```bash
python run_smoke.py --preset heal_kl_50m \
  --max-steps 12207 \
  --resume /kaggle/input/.../checkpoint-final \
  --skip-shock --skip-orig \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_heal_kl_50m_B
```
Final bar: **&lt; 43.77** (CE heal_50m); strong ≲ 35.

Notebook: `SESSION = "A"` | `"B"`.  
**VRAM:** student + teacher. **Disk:** one full ckpt only for A→B.

---

## 6. Success criteria

| Gate | Criterion | Status |
|------|-----------|--------|
| Eng | Multi-hour without disk death | ✅ |
| CE scale | PPL **&lt; 48.2** at 50M | ✅ **~43.77** |
| KL scout | PPL **&lt; 60.6** at 5.24M | ✅ **~49.31** |
| KL 50M | PPL **&lt; 43.77** | TBD heal_kl_50m A+B |
| Parity path | after/orig → 1.0 | open |

---

## 7. How to cite

- Numbers: **this file**  
- Phases: `RESEARCH_PLAN.md`  
- Math: `RESEARCH.md`  
- Kaggle: `KAGGLE.md`  

Update when a new controlled run finishes.
