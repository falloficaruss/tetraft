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

### 2.5 `scale_25m` (partial) — historical; wrong DNA

Interrupted ~21M for disk (full opt dumps). End ~68.6. Obsolete vs heal DNA.

---

## 3. Locked takeaways

1. **Method works.** Shock → finite recovery; loss stays finite.
2. **Best CE result:** **`heal_50m` → ~43.77 @ 50M** (after/orig **~2.48**). Parity still open.
3. **Scope:** skip GDN locked for heal runs (~41% params quantized).
4. **Length helps with diminishing returns:** 5.2M→25M→50M: 60.6→48.2→43.8.
5. **Heal DNA ≫ old scale_25m.**
6. **Disk-safe ckpts work** for multi-hour jobs.
7. **Keep \(c=0.25\)** (c=0.5 deferred).
8. **Next science:** KL + quant-reg scout (`scout_kl_5m`), **not** another plain CE 100M first.
9. **Do not** put longer λ-warmup in the 5M gate (confounds soft vs hard quant).

---

## 4. What we did *not* settle

| Question | Status |
|----------|--------|
| **`scout_kl_5m` end PPL vs ~60.6** | **Next** |
| KL 25–50M after scout pass | Pending gate |
| Longer λ schedule | Deferred (test @ ≥25M, matched time@λ=1) |
| Better \(c\), scale_mode, STE? | Deferred |
| FP CPT control (same tokens, no quant) | Recommended paper control |

---

## 5. Current recipe

### 5.1 CE heal DNA (frozen baseline)

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
| Best CE | **`heal_50m` → ~43.77 @ 50M** (after/orig ~2.48) |

### 5.2 KL recovery DNA (in code; unproven until scout)

| Knob | Value |
|------|--------|
| Preset | **`scout_kl_5m`** |
| Schedule | match `full_smoke_no_gdn`: 1280 steps, λw=**256**, linear→0 |
| `distill_alpha` | **0.5** |
| `distill_temperature` | **2.0** |
| `quant_reg_beta` | **0.01** |
| Teacher | frozen original FP (second load) |
| Gate | end PPL **&lt; 60.6** @ ~5.24M |

Longer KL: `heal_kl_25m` (after scout pass only).

### 5.3 Next runs

1. **`scout_kl_5m`** on Kaggle — go if PPL **&lt; 60.6**  
2. If pass → **`heal_kl_25m`** / 50M with same α,β,T (λw stays 256)  
3. Longer λ only after KL recipe is established (@ 25M+)  
4. Deprioritize: CE-only 100M, old scale_*, BitNet, 2B, SFT, c=0.5  

```bash
python run_smoke.py --preset scout_kl_5m \
  --train-data /kaggle/input/.../train.jsonl \
  --val-data /kaggle/input/.../val.jsonl \
  --output-dir /kaggle/working/checkpoints_scout_kl_5m
```

**VRAM:** student + frozen teacher (~2× 0.8B). If OOM: reduce eval batches; last resort lower batch/seq (breaks token math — avoid).

---

## 6. Success criteria

| Gate | Criterion | Status |
|------|-----------|--------|
| Eng | Multi-hour run without disk death | ✅ heal_25m / heal_50m |
| CE scale | PPL **&lt; 48.2** at 50M | ✅ **~43.77** |
| KL scout | PPL **&lt; 60.6** at 5.24M | TBD `scout_kl_5m` |
| Parity path | after/orig → 1.0 — not claimed early | open (~2.48 best) |
| Recipe freeze | Before 100–200M main | after KL scout |

---

## 7. How to cite

- Numbers: **this file**  
- Phases: `RESEARCH_PLAN.md`  
- Math: `RESEARCH.md`  
- Kaggle: `KAGGLE.md`  

Update when a new controlled run finishes.
