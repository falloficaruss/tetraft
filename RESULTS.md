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
| Shock | λ=1, zero FT | 0 | **≫ 1e6** | — (broken; not a calibrated ratio) |
| `short` | short QAFT smoke | ~0.82M | ~**472** | — |
| **`full_smoke`** | λ_warmup=**256**, lr_warmup=128, **linear→0**, peak lr 2e-4 | **5.24M** | **~79.4** | **~4.5×** |
| **`scale_25m`** (partial) | λ_warmup=**512**, lr_warmup=256, **cosine + min_lr_ratio=0.1** | **~21.0M** (stop ~step 5120/6104) | **~68.6** | **~3.9×** |

**Inventory (stable across runs):** ~187 Linear; ~186 eligible; lm_head skipped; ~**66%** params in quantized Linears (~498M of ~752M counted).

Do **not** treat shock PPL as a continuous quality score — use it only as “zero-FT quant is unusable.”

---

## 2. Recovery curves (after λ ≈ 1)

### 2.1 `full_smoke` (~5.24M tok)

Eval PPL (approximate, from Kaggle logs):

| Step (micro) | ≈ tokens | PPL |
|-------------:|---------:|----:|
| 256 (λ→1) | ~1.0M | ~388 |
| 512 | ~2.1M | ~160 |
| 768 | ~3.1M | ~133 |
| 1024 | ~4.2M | ~100 |
| 1280 end | ~5.2M | **~79–90** mid-eval → **~79.4** final |

Loss after λ=1 drifted ~6 → ~4.2. Finite throughout.

### 2.2 `scale_25m` (interrupted ~84% for disk)

| Step | ≈ tokens | PPL |
|-----:|---------:|----:|
| 512 (λ→1) | ~2.1M | ~246 |
| 1024 | ~4.2M | ~161 |
| 2048 | ~8.4M | ~111 |
| 3072 | ~12.6M | ~94 |
| 4096 | ~16.8M | ~80 |
| 4608 | ~18.9M | ~75 |
| **5120 (stop)** | **~21.0M** | **~68.6** |

Loss after λ=1 ~5.1 → ~3.8. LR still above floor at stop (~3e-5 vs floor 2e-5).  
**Stop reason:** Kaggle working disk full from many full **model + optimizer** checkpoints — not training divergence.  
**No resume** planned for this run as mainline.

---

## 3. Locked takeaways

1. **Method works.** Hard quant destroys the model; QAFT recovers large ground quickly; loss stays finite.
2. **Parity is far.** Best end PPL ~**69** vs orig **~18** (~3.9×). Thesis not closed.
3. **Sample efficiency matters more than raw length so far.**  
   - `full_smoke` reached **~79 @ 5.2M**.  
   - `scale_25m` only reached **~80 @ ~17M**, then **~69 @ ~21M**.  
   Early gap is explained by a **bundled** change (longer λ warmup + cosine schedule), not by “25M is worse than 5M” in absolute best PPL.
4. **Diminishing returns.** Fast drop right after λ=1; then slow grind. Extra tokens still help, but slowly.
5. **Engineering is a first-class constraint.** Unbounded step/best checkpoints with optimizer state will kill Kaggle runs before science finishes.
6. **Do not default to re-running `scale_25m` knobs.** Prefer a deliberate control schedule + one-factor ablations (below).

---

## 4. What we did *not* settle

| Question | Status |
|----------|--------|
| Would **full_smoke knobs × 25M tokens** beat ~69 cheaper than scale_25m? | **Unknown** (never run) |
| Is full-horizon **linear→0** right for long runs? | **Doubtful** — good short finisher; mid-run LR may be too small on long horizons |
| Is residual gap **GDN / hybrid layers**? | **Untested** (high priority) |
| Better \(c\), scale_mode, STE, λ? | **Untested** systematically |
| KL to original, more unique data, 100–200M main? | Later (Phase 3+) |

---

## 5. Architectural next steps (priority order)

These are **design decisions for the next implementation pass**, not a request to blindly scale tokens.

### 5.1 Engineering (before any long Kaggle run)

| Item | Intent |
|------|--------|
| Weights-only `best` / `final` by default | Avoid multi-GB Adam dumps |
| No (or rare) periodic `step_*` saves; prune if enabled | Disk |
| Log PPL/loss to JSON/CSV without full state every eval | Observability without disk death |
| Clear `/kaggle/working/checkpoints` before long jobs | Ops |

### 5.2 Training-schedule architecture

| Keep as **control DNA** | Rethink for **long** runs |
|-------------------------|---------------------------|
| λ_warmup ≈ **256** (full_smoke) | Full-horizon **linear→0** over 25M+ |
| Peak lr **2e-4** until ablated | Prefer **mid-run useful LR** + **late anneal** |
| Same data/model/quant defaults for fair compares | Don’t bundle λw=512 with cosine again |

**Recommended long-run shape (to implement later):**  
`λ_warmup=256` + (cosine → floor ~0.05–0.1 **or** constant then last-20% decay).  
Treat schedule as an **ablation factor**, not a silent default flip.

### 5.3 Method architecture (Phase 2)

**Control:** full_smoke-like quant + λ recipe @ fixed scout budget (**~5–10M tok**), disk-safe.

| Priority | Factor | Why |
|----------|--------|-----|
| **P0** | **Module scope** (all Linear vs exclude GDN / linear-attn) | Qwen3.5 hybrid; GDN may dominate residual shock |
| **P0** | **\(c\)** ∈ {0.25, 0.5} | Grid geometry |
| **P0** | **scale_mode** | absmean_channel vs tensor / absmax |
| **P1** | λ schedule, STE, peak/late LR | Training dynamics |
| **P2** | KL to frozen original | If CE stalls |
| **P2** | 100–200M main (Phase 3) | Only after recipe freeze |
| **P3** | 2B scale-up | After 0.8B recipe stable |

One factor at a time; confirm 1–2 winners at longer budget; then Phase 3.

### 5.4 Explicitly deprioritize

- Blind `scale_50m` with scale_25m knobs  
- BitNet / ternary competitor baselines  
- Resuming the interrupted cosine run for “a bit more”  
- Notebook-as-source-of-truth, 2B, instruct SFT before PPL story  

---

## 6. Success criteria going forward

| Gate | Criterion |
|------|-----------|
| Eng | Multi-hour run without filling Kaggle disk |
| Ablation win | Clearly better val PPL than control at **same token budget** |
| Recipe freeze | Written control + winning factors in this file |
| Parity path | after/orig trending toward 1.0 on main budget — not claimed early |

---

## 7. How to cite in other docs

- Numbers and takeaways: **this file**  
- Phase checklist: `RESEARCH_PLAN.md`  
- Math: `RESEARCH.md`  
- Kaggle how-to: `KAGGLE.md`  

Update this file when a new controlled run finishes (date, preset, tokens, end PPL, after/orig, notes).
