# TetraFT — Research Plan (Final)

**Thesis.** Can quantization-aware fine-tuning onto a 2-bit quaternary grid \(\{-1,-c,c,1\}\) bring a modern pretrained LLM back to **near original quality**, without training a low-bit model from scratch?

**Primary success metric:** gap to the **original** full-precision model (PPL, then downstream)—not comparison to BitNet finetunes.

| Doc | Role |
|-----|------|
| `RESEARCH.md` | Math + method (implementation must match) |
| `RESULTS.md` | **Frozen Kaggle baselines + architectural next steps** |
| `PLAN.md` | Models, data, VRAM, Kaggle |
| `RESEARCH_PLAN.md` | This file — phases, experiments, paper |
| `KAGGLE.md` | Kaggle ops |
| `AGENTS.md` | Coding agent conventions |

---

## 0. Locked decisions

| Decision | Choice |
|----------|--------|
| Method class | Conversion QAFT (pretrained → quaternary forward + heal) |
| Grid | \(\{-1,-c,c,1\}\), default **\(c=0.25\)** |
| Scale default | **per-channel absmean** (`absmean_channel`) |
| STE | `identity` first; ablate `clip` |
| \(\lambda\) anneal | On by default |
| First model | **`Qwen/Qwen3.5-0.8B-Base`** |
| Scale-up model | `Qwen/Qwen3.5-2B-Base` |
| Train data | **FineWeb-Edu fixed sample** (custom Kaggle Dataset) |
| Val data | Fixed held-out (never in train) |
| Platform | **Kaggle** |
| Results primary | Original vs TetraFT (+ zero-FT quant, later W4 PTQ) |
| BitNet | Related work only — **no BitNet baseline runs** |
| Design rule | **Quaternary-optimal**, not BitNet-clone |
| Notebook | Not source of truth; logic in flat `.py` modules |
| Multimodal | Skip vision; language Linear only |
| SFT chat data | After PPL recovery only |

---

## 1. Research questions

1. **Shock** — How far from original right after \(Q(W)\) with no FT?  
2. **Recovery** — How close after 50M / 100–200M tokens on FineWeb-Edu?  
3. **Recipe** — Which \(c\), scale, \(\lambda\), STE, module scope minimize gap to original?  
4. **Necessity** — Does zero-FT quant (and later W4 PTQ) fail where TetraFT recovers?  
5. **Scale** — Does the 0.8B recipe transfer to 2B?  
6. **Downstream** (after PPL) — Relative accuracy vs original on a fixed suite.

---

## 2. Method summary (see RESEARCH.md)

- Latent FP weights \(W\); forward uses \(\widetilde{W} = W + \lambda\,\mathrm{stopgrad}(Q(W)-W)\).  
- \(Q\): channel absmean \(\gamma\), threshold \(t=(1+c)/2\), map to \(\mathcal{G}_c\).  
- Skip lm_head, embeds, vision, norms.  
- Objective: CE on CPT data; optional KL to frozen original later.  
- Optimize for **quaternary conversion of pretrained hybrid Qwen3.5**, not ternary from-scratch schedules.

---

## 3. Phased execution

### Phase 0 — Make the method real  ✅ **COMPLETE**

**Goal:** Code matches `RESEARCH.md`; tests pass; ready for 0.8B smoke.

| # | Task | Status |
|---|------|--------|
| 0.1 | Rewrite `quantize.py`: `scale_mode`, channel/tensor scales, unified `c`, \(\lambda\), `ste_mode` | ✅ |
| 0.2 | Harden `model.py`: exclude patterns, replace report, `replace_from_config` | ✅ |
| 0.3 | Align `config.py` defaults (0.8B-Base, \(c=0.25\), absmean_channel) | ✅ |
| 0.4 | Expand `tests/` | ✅ |
| 0.5 | Clip STE forward fix; λ logging in trainer | ✅ |

**Out of scope for Phase 0:** FineWeb download, full 0.8B train, downstream eval, packing kernels, distillation.

---

### Phase 1 — Data + 0.8B smoke (Kaggle)  ✅ **COMPLETE**

| # | Task | Status |
|---|------|--------|
| 1.1 | Build FineWeb-Edu sample (start **50M**, val held-out); upload Kaggle Dataset | ✅ |
| 1.2 | Load `Qwen3.5-0.8B-Base`, language-only policy, dump Linear inventory | ✅ |
| 1.3 | Measure **original** val PPL | ✅ ~**17.7** |
| 1.4 | Hard quant \(\lambda=1\), **zero FT** → shock PPL | ✅ ≫ **1e6** (broken; not calibrated) |
| 1.5 | Short QAFT smoke: loss finite, recovery vs shock | ✅ ~200 steps / ~0.8M tok → eval PPL ~**472** |
| 1.6 | Memory recipe: BF16 + 8-bit Adam + grad checkpointing | ✅ |

**Exit:** Stable train/eval loop; shock and partial recovery observed. **Met.**

**Kaggle short-smoke baseline (record):** orig ≈ 17.7 → shock ≫ 1e6 → after ~0.8M tok ≈ 472; loss finite; 66% params quantized (lm_head skipped).

### Phase 1b — Longer smoke (1–5M tokens)  ✅ **COMPLETE**

| # | Task | Status |
|---|------|--------|
| 1b.1–1b.2 | `full_smoke` ~5.24M tok | ✅ end PPL **~79.4** (after/orig ≈ 4.5) |
| 1b.3 | PPL vs steps after λ=1 | ✅ 388 → 160 → 133 → 100 → 90 → **79** |

**Most sample-efficient controlled run so far.** Schedule DNA: λ_warmup=**256**, lr_warmup=128, linear→0, lr=2e-4.

### Phase 1c — Longer scale-up  ✅ **PARTIAL (recorded)**

| Run | Result |
|-----|--------|
| `scale_25m` (λw=512, cosine+0.1 floor) | Interrupted ~21M tok (disk) → PPL **~68.6** (after/orig ≈ 3.9) |
| Efficiency vs full_smoke | Worse early: ~79 only around **~17M** tok, not 5M |

**Do not** treat `scale_25m` knobs as the default control.  
Full numbers, curves, and lessons: **`RESULTS.md`**.

---

### Phase 2 — Recipe + schedule architecture  ← **IN PROGRESS**

Fixed data order; optimize **gap to original**.  
**Heal DNA:** λ_warmup≈**256**, peak lr 2e-4, **\(c=0.25\)**, absmean_channel, **`skip_linear_attn=True`**, disk-safe ckpts.  
Long runs: **cosine + min_lr_ratio=0.1** (presets `heal_25m` / `heal_50m`).

| Priority | Factor | Status |
|----------|--------|--------|
| P0 | **scope** (all Linear vs exclude GDN) | ✅ scout @ 5.2M: skip GDN **~60.6** vs all-Linear **~79.4** |
| P0 | **length** with heal DNA | ✅ `heal_25m` ~48.2; ✅ **`heal_50m` ~43.77** (after/orig ~2.48) |
| P0 | **KL + quant-reg** | ✅ scout ~49.3; ✅ **`heal_kl_50m` ~34.38 @ 50M** (after/orig ~1.95) |
| P0 | **\(c\)** 0.25 vs 0.5 | Deferred (keep **0.25**) |
| P0 | **scale_mode** | Deferred |
| P1 | λ / STE / LR shape | cosine+floor in heal; longer λ only after KL @ ≥25M |
| tokens | CE 50M done; KL scout next; main after freeze |

Details: **`RESULTS.md`**.

---

### Phase 3 — 0.8B main recovery + supporting baselines

- Best recipe at **100–200M** tokens  
- Tables: Original | Zero-FT quant | TetraFT  
- Later: W4 PTQ on same checkpoint  
- Optional: same-data FP16 CPT control  
- Efficiency: size estimate, % quantized, train cost  

---

### Phase 4 — Scale to Qwen3.5-2B-Base

- Reuse recipe; adjust batch/seq for VRAM  
- Shock + recovery curves  
- Downstream suite with Original column  

---

### Phase 5 — Paper polish

- Ablation figures, failure analysis, limitations  
- Optional: instruct checkpoint, learnable \(c\), second architecture  

---

## 4. Results protocol (for later; do not block Phase 0)

**Always include Original.**

| Block | Content |
|-------|---------|
| A | Parity: Original vs TetraFT (PPL ± downstream) |
| B | Recovery curves vs tokens |
| C | Zero-FT quant (+ later W4 PTQ) |
| D | Method ablations |
| E | Footprint / train cost |

**Parity definitions**

- \(\mathrm{PPL}_Q / \mathrm{PPL}_{\mathrm{orig}}\)  
- Mean relative task accuracy when enabled  
- Fraction of shock closed after budget \(B\)

**Do not run:** BitNet / ternary competitor finetunes.

---

## 5. Engineering roadmap (priority)

1. **Phase 0** quant + model + config + tests  
2. Data sampler script + Kaggle dataset docs  
3. Trainer: BF16, 8-bit Adam, logging (loss, PPL, \(\lambda\), bin hist)  
4. Eval script: original vs checkpoint PPL on fixed val  
5. Experiment configs (YAML or dataclass presets)  
6. Later: 2-bit pack export, lm-eval wrapper  

---

## 6. Risks

| Risk | Mitigation |
|------|------------|
| Hybrid GDN brittle under quant | Module-scope ablation; FFN-first |
| Kaggle VRAM | BF16 + 8-bit Adam + short seq + accum |
| Recipe overfit to 0.8B | Freeze recipe before 2B; limited retune |
| Only PPL recovers | Downstream in Phase 4; don’t overclaim early |
| Code ≠ math | Phase 0 + tests; `RESEARCH.md` is law |

---

## 7. Paper outline (target)

1. Intro — convert existing LLMs to 2-bit quaternary; retain quality  
2. Background — QAT, PTQ limits; BitNet as related motivation  
3. Method — TetraFT (grid, scale, STE, \(\lambda\), module policy)  
4. Setup — Qwen3.5, FineWeb-Edu sample, budgets, parity metrics  
5. Results — shock, recovery, parity, ablations, efficiency  
6. Analysis — bins, hybrid layers, remaining gaps  
7. Limitations & conclusion  

---

## 8. Immediate next step

### → **Post–`heal_kl_50m` science** (see `RESULTS.md` §5.4+)

1. **Locked SOTA:** KL-50M **~34.38** PPL (after/orig **~1.95**)  
2. **Now:** fresh α/T scout @ 5.2M (gate &lt;49.31) — `SESSION=S` / `a03_t2` first  
3. ~~Polish B @ 2e-5~~ — **FAIL**; stop small-LR continue on B  
4. **Later:** Muon 5M ablation design in §5.7 (**not implemented**); hygiene; longer KL  
5. Deprioritize: more polish, CE-only longer, c=0.5, BitNet, 2B  

Details: `RESULTS.md` §5.4 / §5.5 / §5.7.
