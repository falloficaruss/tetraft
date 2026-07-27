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
| **`heal_kl_50m` A** | KL DNA; cosine horizon 12207; stop 6104 | **25.0M** | **~48.65** | **~2.75** |
| **`heal_kl_50m` A+B** | 2-session resume; full 12207 | **50.0M** | **~34.38** | **~1.95** |
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

### 2.6 `heal_kl_50m` (~50.0M, 2-session) ✅ **new SOTA**

Kaggle 2026-07-26. Preset `heal_kl_50m`: skip GDN, λw=256, cosine **horizon 12207**, α=0.5 / T=2 / β=0.01.  
Two-session resume (opt+sched continuous); data reshuffled at session boundary.

| Session | Steps | ≈ tokens | Val PPL | after/orig | Ckpt |
|---------|------:|---------:|--------:|-----------:|------|
| **A** | 0→6104 | **25.0M** | **48.65** | **2.75** | full `checkpoint-final` |
| **B** | 6104→12207 (`resumed_step=6104`) | **50.0M** total | **34.38** | **1.95** | weights-only final |

Shock A: **17839** (matches skip-GDN). Inventory: 96 eligible / **41%** quantized.

**vs controls (same ~50M class budget where noted):**

| Compare | PPL | Δ |
|---------|----:|---|
| CE `heal_50m` @ 50M | 43.77 | **KL −9.4** (~21% relative better) |
| CE `heal_25m` @ 25M | ~48.2 | Session A ≈ tied (~48.65) |
| `scout_kl_5m` @ 5.2M | 49.31 | 50M KL continues falling hard |
| Orig | 17.67 | still **~1.95×** (not parity) |

**Read:** Session A mid PPL looks like CE-25M because cosine is only halfway; **second half (B) is where KL pulls away** from CE-50M. Two-session resume **worked** (full A → B).

**Paper-relevant:** first after/orig **&lt; 2.0**; hybrid model (~41% quaternary). Optional cold re-eval of B `checkpoint-final` / best.

### 2.7 `scale_25m` (partial) — historical; wrong DNA

Interrupted ~21M for disk (full opt dumps). End ~68.6. Obsolete vs heal DNA.

---

## 3. Locked takeaways

1. **Method works.** Shock → finite recovery.
2. **Best overall:** **`heal_kl_50m` → ~34.38 @ 50M** (after/orig **~1.95**).
3. **Best CE:** `heal_50m` → ~43.77 (KL clearly beats CE at matched budget).
4. **KL is the recovery lever;** CE length alone plateaus higher.
5. **Scope:** skip GDN locked (~41% quantized) — always report with PPL.
6. **Keep \(c=0.25\)**, λw=**256**, α=0.5 / T=2 until ablated.
7. **Two-session resume** (full opt ckpt + `schedule_max_steps`) is production path for long KL.
8. **Parity still open** (target ≲1.3× / ~23 PPL).

---

## 4. What we did *not* settle

| Question | Status |
|----------|--------|
| Longer KL (100M+) / α–T ablations | Open — see §5.3 |
| FP CPT control (same tokens, no quant) | Recommended paper control |
| Downstream vs orig | Deferred until closer PPL |
| Longer λ / c=0.5 / all-Linear + KL | Deferred |
| Bit-shift kernels | Systems track (parallel) |

---

## 5. Current recipe (frozen KL DNA)

| Knob | Value |
|------|--------|
| Best | **`heal_kl_50m` → ~34.38 @ 50M** (after/orig ~1.95) |
| α / T / β | **0.5** / **2.0** / **0.01** |
| λ_warmup | **256** |
| LR | 2e-4, cosine → 0.1, horizon = run length |
| Scope | `skip_linear_attn=True`, c=0.25, ~41% quantized |
| Teacher | frozen original FP |

### 5.3 Science options next

**A. Paper hygiene (cheap, high value)**  
1. **FP continued-pretrain control** — same FineWeb, ~25–50M tokens, **no quant**.  
2. **Cold re-eval** of KL-50M final/best on full val protocol.  
3. Freeze tables: Original | Shock | CE-50M | KL-5M | KL-50M | (% quantized).

**B. Two concrete quality runs (documented protocols below)**  
Independent; either order. Do **not** mix into one job.

**C. Later / parallel**  
`heal_kl_100m`; GDN+KL; post-quant RMSNorm; 2-bit pack + bit-shift kernels.

**D. Deprioritize**  
CE-only longer, c=0.5, BitNet, chat SFT, 2B — until gap ≲1.5×.

---

### 5.4 Option 1 — Fresh α / T scout @ ~5.2M

**Goal:** Beat locked scout **`scout_kl_5m` ~49.31** by changing only distill knobs.  
**Start:** **from scratch** (no resume from heal_kl_50m).  
**DNA (match scout):** skip GDN, λw=**256**, linear→0, lr **2e-4**, 1280 steps (~5.24M), β=0.01, c=0.25.

| Run ID (suggested) | α | T | Notes |
|--------------------|---|---|--------|
| `scout_kl_5m` (done) | 0.5 | 2.0 | **baseline gate 49.31** |
| `scout_kl_a03_t2` | **0.3** | 2.0 | more teacher |
| `scout_kl_a05_t1` | 0.5 | **1.0** | harder KL |
| `scout_kl_a03_t1` | **0.3** | **1.0** | only if both above help |

```bash
# Example: more teacher, same 5M schedule
python run_smoke.py --preset scout_kl_5m \
  --distill-alpha 0.3 --distill-temperature 2.0 \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_scout_kl_a03_t2
```

| Gate | Action |
|------|--------|
| End PPL **&lt; 49.31** | Winner; use α/T on next long KL / polish |
| All ≥ 49.31 | Keep α=0.5 T=2.0; try Option 2 or hygiene |

**Do not** lengthen λ on these scouts. One cell at a time if VRAM/time tight (prefer α=0.3/T=2 first).

---

### 5.5 Option 2 — Polish from heal_kl_50m B @ lr 2e-5 (~5M)

**Goal:** Shave PPL below **~34.38** with small LR + same (or scout-winning) KL.  
**Start:** **Session B weights** (`checkpoint-final` or best from `heal_kl_50m` B).  
**Not** a seamless cosine continue — B final is **weights-only** → new Adam; that is intended.

| Knob | Value |
|------|--------|
| Init | `--resume` B `checkpoint-final` (weights OK) |
| Extra tokens | **~5.24M** → **1280** micro-steps |
| `max_steps` | **resumed_step + 1280** = **12207 + 1280 = 13487** |
| lr | **2e-5** |
| Schedule | **constant 2e-5** — cosine + `min_lr_ratio=1.0`, `warmup_steps=0`, `schedule_max_steps=1280` |
| λ | stays 1 (`global_step` already ≫ 256) |
| α / T / β | **0.5 / 2.0 / 0.01** unless Option 1 found a winner |
| Scope | skip GDN (same replace policy) |
| Flags | `--skip-shock --skip-orig` |

```bash
# ~5M polish after KL-50M (weights-only resume → new optimizer)
# Preset locks: max_steps=13487, lr=2e-5 constant, schedule_max_steps=1280
python run_smoke.py --preset polish_kl_5m \
  --resume /kaggle/input/.../checkpoint-final \
  --skip-shock --skip-orig \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_polish_kl_5m
```

Notebook: `SESSION = "P"`. Weights-only rebuilds Adam+sched from 0 → `schedule_max_steps=1280` is polish length, not 13487.

**Critical:** `max_steps` must be **&gt; resumed_step**. If you leave `max_steps=12207` after loading step 12207, train **no-ops**.

| Gate | Action |
|------|--------|
| End PPL **&lt; 34.38** | New SOTA; log here; consider more polish or long run with winning α/T |
| ≥ 34.38 | Stop polish; try Option 1 or FP CPT hygiene |

**Optional:** 10M polish = `max_steps 12207+2560` (=14767), same lr 2e-5.

---

### 5.6 How the two options relate

| | Option 1 α/T scout | Option 2 polish |
|--|--------------------|-----------------|
| Init | Fresh pretrained | KL-50M B weights |
| Tokens | ~5.2M | ~5.2M (or 10M) |
| Gate | &lt; **49.31** | &lt; **34.38** |
| Changes | α, T only | lr 2e-5; continue weights |
| Independent? | **Yes** | **Yes** |

**Suggested order:** Option 1 first (no big ckpt needed) → Option 2 with baseline α/T or scout winner.  
**Recommended default path:** hygiene when drafting paper; run **§5.4** and/or **§5.5** for quality; 100M KL only if still far from 1.5×.

---

## 6. Success criteria

| Gate | Criterion | Status |
|------|-----------|--------|
| Eng | Multi-hour / multi-session resume | ✅ |
| CE scale | PPL **&lt; 48.2** at 50M | ✅ **~43.77** |
| KL scout | PPL **&lt; 60.6** at 5.24M | ✅ **~49.31** |
| KL 50M vs CE | PPL **&lt; 43.77** | ✅ **~34.38** |
| KL strong | PPL **≲ 35** | ✅ **~34.38** |
| Parity path | after/orig ≲ **1.3** (~23) | open (~1.95) |

---

## 7. How to cite

- Numbers: **this file**  
- Phases: `RESEARCH_PLAN.md`  
- Math: `RESEARCH.md`  
- Kaggle: `KAGGLE.md`  

Update when a new controlled run finishes.
