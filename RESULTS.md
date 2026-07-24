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

**Resume:** final ckpt is **weights-only** (no Adam). Do **not** treat as seamless resume into 50M — run `heal_50m` **from scratch**.

### 2.4 `scale_25m` (partial) — historical; wrong DNA

Interrupted ~21M for disk (full opt dumps). End ~68.6. Obsolete vs heal DNA.

---

## 3. Locked takeaways

1. **Method works.** Shock → finite recovery; loss stays finite.
2. **Parity still open.** Best end PPL **~48.2** vs orig **~17.7** (~2.7×) after 25M heal DNA.
3. **Scope:** skip GDN is locked for heal runs (scout + scale both use it).
4. **Length helps under heal DNA.** 5.2M→25M cut PPL 60.6→48.2; late curve still falling → **`heal_50m` next**.
5. **Heal DNA ≫ old scale_25m.** Same ~25M class budget, better recipe (skip GDN + λw=256).
6. **Disk-safe ckpts work** for multi-hour 25M jobs.
7. **Keep \(c=0.25\)** (c=0.5 deferred).
8. **No true resume** from weights-only 25M into 50M without new opt/sched — fresh `heal_50m`.

---

## 4. What we did *not* settle

| Question | Status |
|----------|--------|
| **`heal_50m` end PPL / after/orig** | **Next** |
| Plateau vs still falling at 50M? | Unknown |
| Better \(c\), scale_mode, STE? | Deferred |
| KL / 100–200M main? | Later if 50M plateaus far from orig |

---

## 5. Current recipe (locked)

| Knob | Value |
|------|--------|
| Model | Qwen3.5-0.8B-Base |
| \(c\) | **0.25** |
| scale_mode | absmean_channel |
| STE | identity |
| Scope | **`skip_linear_attn=True`** |
| λ_warmup | **256** |
| Peak lr | 2e-4 |
| Long-run LR | **cosine → min_lr_ratio=0.1** |
| Best so far | **`heal_25m` → ~48.2 @ 25M** (after/orig ~2.73) |
| Next preset | **`heal_50m`** (~50M, fresh start) |

### 5.1 Next runs

1. **`heal_50m` from scratch** (same DNA; do not resume 25M weights-only as primary)  
2. If still falling hard near 50M → consider longer main / more data  
3. Defer c / scale_mode until length signal clearer  
4. Deprioritize: old scale_*, BitNet, 2B, SFT, c=0.5

---

## 6. Success criteria

| Gate | Criterion |
|------|-----------|
| Eng | Multi-hour run without disk death | ✅ heal_25m |
| Scale win | PPL **&lt; 48.2** at 50M; after/orig down | TBD heal_50m |
| Parity path | after/orig → 1.0 — not claimed early |
| Recipe freeze | Before 100–200M main |

---

## 7. How to cite

- Numbers: **this file**  
- Phases: `RESEARCH_PLAN.md`  
- Math: `RESEARCH.md`  
- Kaggle: `KAGGLE.md`  

Update when a new controlled run finishes.
