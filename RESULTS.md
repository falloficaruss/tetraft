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
| **`full_smoke`** | all Linear; λw=**256**, lr_warmup=128, **linear→0**, lr 2e-4 | **5.24M** | **~79.4** | **~4.5×** |
| **`full_smoke` + `skip_linear_attn`** | GDN FP; same schedule as full_smoke | **5.24M** | **~60.6** | **~3.43×** |
| **`scale_25m`** (partial) | all Linear; λw=**512**, cosine+0.1 floor | **~21.0M** (disk stop) | **~68.6** | **~3.9×** |

**Inventory — all eligible (control):** ~187 Linear; ~186 eligible; lm_head skipped; ~**66%** params quantized (~498M / ~752M).

**Inventory — `skip_linear_attn`:** ~**96** eligible / **91** skipped; ~**41%** quantized (~308M / ~752M). GDN path `linear_attn.*` left FP.

Do **not** treat all-Linear shock PPL as a continuous quality score — use it only as “zero-FT full quant is unusable.” Skip-GDN shock is finite but still far from original.

---

## 2. Recovery curves (after λ ≈ 1)

### 2.1 `full_smoke` (~5.24M tok) — all eligible Linear

| Step (micro) | ≈ tokens | PPL |
|-------------:|---------:|----:|
| 256 (λ→1) | ~1.0M | ~388 |
| 512 | ~2.1M | ~160 |
| 768 | ~3.1M | ~133 |
| 1024 | ~4.2M | ~100 |
| 1280 end | ~5.2M | **~79–90** mid-eval → **~79.4** final |

Loss after λ=1 drifted ~6 → ~4.2. Finite throughout.

### 2.2 `full_smoke` + `skip_linear_attn` (~5.24M tok) ✅ Phase 2 scope win

Kaggle 2026-07-24. Disk-safe code (weights-only best/final, `save_steps=0`).

| Step (micro) | ≈ tokens | PPL |
|-------------:|---------:|----:|
| 256 (λ→1) | ~1.0M | **~169** |
| 512 | ~2.1M | **~102** |
| 768 | ~3.1M | **~92** |
| 1024 | ~4.2M | **~73** |
| 1280 mid-eval | ~5.2M | **~66.9** |
| **final val** | **5.24M** | **~60.6** |

Loss after λ=1 ~4.9 → ~4.0. after/orig **~3.43**.  
vs control at same budget: **79.4 → 60.6** (~24% relative PPL drop). Shock far milder (~1.8e4 vs ≫1e6).

### 2.3 `scale_25m` (interrupted ~84% for disk) — historical; wrong DNA

| Step | ≈ tokens | PPL |
|-----:|---------:|----:|
| 512 (λ→1) | ~2.1M | ~246 |
| 1024 | ~4.2M | ~161 |
| 2048 | ~8.4M | ~111 |
| 3072 | ~12.6M | ~94 |
| 4096 | ~16.8M | ~80 |
| 4608 | ~18.9M | ~75 |
| **5120 (stop)** | **~21.0M** | **~68.6** |

**Stop reason:** full model+optimizer checkpoints filled disk. **Do not** use as default control (λw=512 + no GDN skip).

---

## 3. Locked takeaways

1. **Method works.** Hard quant destroys the model; QAFT recovers large ground quickly; loss stays finite.
2. **Parity is still open.** Best end PPL ~**60.6** vs orig **~18** (~3.4×) at 5.2M with GDN skipped — not closed.
3. **Scope matters (P0 settled at scout budget).** Leaving Qwen3.5 GDN (`linear_attn`) in FP is a clear win at fixed 5.2M tokens and milder shock.
4. **Sample efficiency:** old `scale_25m` (all Linear, λw=512) only hit ~69 @ ~21M — worse early path than short full_smoke. New scale runs must use **full_smoke DNA + skip_linear_attn**, not scale_25m knobs.
5. **Diminishing returns after λ=1** still apply; extra tokens should help grind toward orig, especially with healthier LR on long horizons.
6. **Engineering:** weights-only + `save_steps=0` + `metrics.jsonl` required for long Kaggle jobs.
7. **Keep \(c=0.25\)** for the next scale-up (c=0.5 deferred).

---

## 4. What we did *not* settle

| Question | Status |
|----------|--------|
| How close with **heal DNA × 25M / 50M** tokens? | **Next** (`heal_25m` / `heal_50m` presets) |
| Is full-horizon **linear→0** right for long runs? | **Doubtful** — heal presets use **cosine + floor 0.1** |
| Better \(c\), scale_mode, STE? | Deferred (\(c\) stays 0.25 for now) |
| KL to original, 100–200M main? | Later if heal plateaus |

---

## 5. Current recipe (locked for scale-up)

| Knob | Value |
|------|--------|
| Model | Qwen3.5-0.8B-Base |
| \(c\) | **0.25** |
| scale_mode | absmean_channel |
| STE | identity |
| Scope | **`skip_linear_attn=True`** (GDN FP) |
| λ_warmup | **256** micro-steps |
| Peak lr | 2e-4 |
| Long-run LR | **cosine → min_lr_ratio=0.1** (not linear→0) |
| Scout | `full_smoke` + skip → **~60.6 @ 5.2M** |
| Scale presets | `heal_25m` (~25M), `heal_50m` (~50M) |

### 5.1 Engineering ✅

Weights-only best/final; `save_steps=0`; `metrics.jsonl`; clear working checkpoints before long jobs.

### 5.2 Next runs (priority)

1. **`heal_25m`** on Kaggle — primary length test toward original  
2. If still falling steeply near end → **`heal_50m`**  
3. Defer \(c\)/scale_mode ablations until after length signal  
4. Deprioritize: blind old `scale_25m`/`scale_50m`, BitNet, 2B, SFT

---

## 6. Success criteria going forward

| Gate | Criterion |
|------|-----------|
| Eng | Multi-hour run without filling Kaggle disk |
| Scale win | End PPL **&lt; 60.6** at higher budget; after/orig trending down |
| Parity path | after/orig toward 1.0 — not claimed early |
| Recipe freeze | Winning length + knobs written here before 100–200M main |

---

## 7. How to cite in other docs

- Numbers and takeaways: **this file**  
- Phase checklist: `RESEARCH_PLAN.md`  
- Math: `RESEARCH.md`  
- Kaggle how-to: `KAGGLE.md`  

Update this file when a new controlled run finishes (date, preset, tokens, end PPL, after/orig, notes).
