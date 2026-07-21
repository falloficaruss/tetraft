# TetraFT — Research Plan (Final)

**Thesis.** Can quantization-aware fine-tuning onto a 2-bit quaternary grid \(\{-1,-c,c,1\}\) bring a modern pretrained LLM back to **near original quality**, without training a low-bit model from scratch?

**Primary success metric:** gap to the **original** full-precision model (PPL, then downstream)—not comparison to BitNet finetunes.

| Doc | Role |
|-----|------|
| `RESEARCH.md` | Math + method (implementation must match) |
| `PLAN.md` | Models, data, VRAM, Kaggle |
| `RESEARCH_PLAN.md` | This file — phases, experiments, paper |
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

### Phase 0 — Make the method real  ✅ **START HERE**

**Goal:** Code matches `RESEARCH.md`; tests pass; ready for 0.8B smoke.

| # | Task | Done when |
|---|------|-----------|
| 0.1 | Rewrite `quantize.py`: `scale_mode`, channel/tensor scales, unified `c`, \(\lambda\), `ste_mode` | Unit tests for bins + STE |
| 0.2 | Harden `model.py`: exclude patterns, replace report, default `c` from config | lm_head/vision safe; inventory print |
| 0.3 | Align `config.py` defaults with this plan (0.8B model, \(c=0.25\), absmean_channel) | Single source of truth |
| 0.4 | Expand `tests/` | `pytest tests/ -v` green |
| 0.5 | Minimal train/eval path works offline on CPU toy or tiny tensors | No notebook dependency |

**Out of scope for Phase 0:** FineWeb download, full 0.8B train, downstream eval, packing kernels, distillation.

**Next after Phase 0:** Phase 1 (data + 0.8B smoke on Kaggle).

---

### Phase 1 — Data + 0.8B smoke (Kaggle)

| # | Task |
|---|------|
| 1.1 | Build FineWeb-Edu sample (start **50M**, val held-out); upload Kaggle Dataset |
| 1.2 | Load `Qwen3.5-0.8B-Base`, language-only policy, dump Linear inventory |
| 1.3 | Measure **original** val PPL |
| 1.4 | Hard quant \(\lambda=1\), **zero FT** → shock PPL |
| 1.5 | QAFT **1–5M tokens** smoke: loss finite, some recovery movement |
| 1.6 | Memory recipe: BF16 + 8-bit Adam + grad checkpointing |

**Exit:** Stable train/eval loop; shock and partial recovery observed.

---

### Phase 2 — 0.8B recipe search

Fixed data order; optimize **gap to original**.

| Factor | Levels |
|--------|--------|
| \(c\) | 0.25, 0.5 |
| scale | absmean_channel, absmean_tensor, absmax_* |
| \(\lambda\) | hard / linear / delayed |
| ste | identity / clip |
| scope | all eligible Linear / FFN-heavy / exclude GDN |
| tokens | ~50M per ablation; 100–200M for best recipe |

Optional: CE + KL(original) if CE alone stalls.

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

### → Implement **Phase 0** in code

Order of work:

1. `quantize.py` (absmean channel, \(c\), \(\lambda\), ste modes)  
2. `model.py` (skips + report)  
3. `config.py` (defaults: 0.8B-Base, \(c=0.25\), `scale_mode=absmean_channel`)  
4. `tests/` update  
5. Smoke import path on CPU  

When Phase 0 is done, proceed to Phase 1 on Kaggle (data sample + 0.8B smoke).
