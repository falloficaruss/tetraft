# TetraFT — Paper draft (internal)

**Status:** living internal outline. Not submitted; venue unset.  
**Math / derivations:** [`PAPER_MATH.md`](PAPER_MATH.md) (split equation sheet + theory appendix).  
**Implementation law:** [`RESEARCH.md`](RESEARCH.md).  
**Numbers & gates:** [`RESULTS.md`](RESULTS.md).  
**Mainline run:** `heal_kl_trust_400m` (16×25M; paper pack via `run_pack.py`).

Update this file as sessions finish. Do not invent BitNet competitor numbers. Soften abstract strength until 400M (and any parity target) lands.

---

## Working titles

1. *TetraFT: Quantization-Aware Conversion of Pretrained LLMs to 2-bit Quaternary Weights*
2. *Healing Pretrained LLMs onto a Quaternary Grid*
3. *Quaternary Weight Conversion for Efficient LLMs via Quantization-Aware Fine-Tuning*

---

## Claim checklist

### Must defend (core)

| # | Claim | Evidence source |
|---|--------|-----------------|
| M1 | Pretrained LLM linears can be mapped to \(\mathcal{G}_c=\{-1,-c,c,1\}\) and trained with STE/QAFT | Method + code |
| M2 | Zero-FT hard quant is catastrophic or near-catastrophic; QAFT recovers finite, improving PPL | Shock + curves |
| M3 | Recipe levers matter: module scope (skip GDN), token budget, teacher KL, soft-trust STE | Ablation tables |
| M4 | Main results are **hybrid**: ~41% params quaternary under heal DNA; embed/lm_head/GDN FP | Inventory |
| M5 | Grid is **2-bit-indexable**; inference footprint is BitNet-*class in principle* (codes + scales) | `PAPER_MATH.md` §10 |

### Soft / aspirational (word carefully)

| # | Claim | Condition |
|---|--------|-----------|
| S1 | Quality “comparable” to original | Prefer after/orig \(\lesssim 1.3\); else say **substantial recovery** + residual gap |
| S2 | Inference “much more efficient like BitNet” | **Representational** efficiency now; latency only if kernels/pack land |
| S3 | Trust+α mainline beats prior KL-50M SOTA | Needs `heal_kl_trust_400m` (or intermediate sessions) |

### Forbidden / avoid

| # | Do not |
|---|--------|
| F1 | BitNet / b1.58 **baseline bake-off** tables (related work only) |
| F2 | Claim full-model 2-bit when ~41–66% quantized |
| F3 | Claim custom kernel speedups without measurements |
| F4 | Equate train VRAM (latent FP + teacher) with deploy cost |
| F5 | Present LoRA runs as pure weight-only quaternary |
| F6 | Hide polish / bundle failures if discussing recipe search |

---

## Narrative spine

1. **Problem.** Extreme low-bit weights are attractive for memory/compute; from-scratch low-bit pretraining is expensive; PTQ at 2-bit on pretrained LLMs fails.
2. **Proposal.** **TetraFT** — conversion QAFT onto a **quaternary** grid \(\{-1,-c,c,1\}\), heal toward the **original** checkpoint.
3. **Method.** Channel scales, λ-anneal, STE (identity → soft trust), optional KL to frozen FP teacher, hybrid module policy on Qwen3.5.
4. **Efficiency angle.** 2-bit codes + scales enable BitNet-like *representational* efficiency; v1 proves **capability recovery**, not kernel SOTA.
5. **Evidence.** Shock → heal on Qwen3.5-0.8B-Base / FineWeb-Edu; ablations; long marathon TBD.

---

## Contributions (draft bullets)

1. **Quaternary conversion QAFT** for modern pretrained LLMs, with a fixed grid and conversion-oriented scale/STE design (not a BitNet training clone).
2. **Heal recipe** for hybrid Qwen3.5: skip brittle GDN path, λ warmup, CE+KL recovery, soft-trust STE.
3. **Empirical recovery study** on 0.8B: frozen Original column, shock, length/scope/KL/trust ablations, negative results (polish, bundle).
4. **Efficiency model** for packed quaternary weights + FP residuals (theory; systems track optional).
5. **Analysis tools / discussion:** layer-wise Q-error, STE trust, MSE-scale discussion, gradient alignment definition.

---

# Section outline

## Abstract *(draft bullets — rewrite after 400M)*

- Convert pretrained LLMs to **2-bit quaternary** weights \(\{-1,-c,c,1\}\) via QAFT (**TetraFT**).
- Goal: approach **original** quality while enabling extreme-bit **storage / structured matmul** potential (BitNet-class efficiency narrative).
- Method: absmean-channel scale, λ-anneal, soft-trust STE, teacher KL, hybrid scope on Qwen3.5.
- Results: [Orig PPL 17.67; best locked long KL ~34.4 @ 50M; trust scout ~43.3 @ 5M; **400M TBD**].
- Limitation one-liner: residual gap; kernels not in v1; hybrid % quantized.

## 1. Introduction

- LLM deploy cost → low-bit weights.
- PTQ limits at extreme bits; from-scratch ternary/binary needs huge compute.
- **Conversion** underexplored for **quaternary** grids on hybrid architectures.
- Thesis + contributions + non-goals (no W+A v1, no chat-SFT-first, no BitNet bake-off).

## 2. Related work

| Thread | Notes |
|--------|--------|
| STE / QAT / BNN | Classic STE; LSQ-style learned quant (contrast fixed grid) |
| Extreme low-bit LLMs | BitNet / b1.58 — **motivation** for discrete weights + systems |
| PTQ / GPTQ-style | Why zero-FT fails at 2-bit here |
| KD under compression | Teacher = original FP |
| QuEST (ICML 2025) | Trust / grad alignment **ideas only** |
| Hybrid attention (Qwen3.5 GDN) | Module-scope necessity |

## 3. Method

Point to [`PAPER_MATH.md`](PAPER_MATH.md) for full equations. Body should include:

| § | Content | Math ref |
|---|---------|----------|
| 3.1 | Setup: latent \(W\), `QuantizedLinear` | §0–2 |
| 3.2 | Grid \(\mathcal{G}_c\), no zero, \(c=0.25\) | §1 |
| 3.3 | \(Q(W)\): γ, \(t\), segment | §2 |
| 3.4 | λ-forward + anneal | §3 |
| 3.5 | STE: identity, clip, **soft trust** | §4 |
| 3.6 | Loss: CE + KL + commitment | §5 |
| 3.7 | Module policy / hybrid GDN skip | §6, §11 |
| 3.8 | Inference representation (codes, γ, FP rest) | §10 |

**Figures (method):** F2 schedule schematic; F3 grid+threshold; F4 trust mask \(m(e)\).

## 4. Experimental setup

| Item | Value |
|------|--------|
| Model | `Qwen/Qwen3.5-0.8B-Base` |
| Train data | FineWeb-Edu fixed samples (`tetraft-fineweb-edu-*`) |
| Val | Held-out FineWeb-Edu (never in train) |
| Stack | BF16, AdamW8bit, grad checkpointing |
| Seq / tokens per step | 512; batch 1 × accum 8 → **4096 tok/step** |
| Primary metric | Val PPL; after/orig |
| Always report | % params quantized; Original column |
| Platform | Kaggle |
| Mainline preset | `heal_kl_trust_400m` — trust \(s=1.0\), \(\alpha=0.3\), \(T=2\), \(\beta=0.01\), skip GDN, no LoRA |

Downstream (lm-eval): **deferred** until PPL closer to original.

## 5. Results

### 5.1 Main parity table (frozen + TBD)

| Run | ≈ tokens | Val PPL | after/orig | Notes |
|-----|---------:|--------:|-----------:|-------|
| Original FP | — | **17.67** | **1.0** | |
| Shock (all Linear) | 0 | ≫1e6 | — | broken |
| Shock (skip GDN) | 0 | ~1.78e4 | ~1009× | |
| CE `heal_50m` | 50M | ~43.77 | ~2.48 | |
| KL `heal_kl_50m` | 50M | **~34.38** | **~1.95** | best locked long (legacy DNA) |
| Trust scout `scout_kl_trust_a03_5m` | 5.24M | **~43.34** | ~2.45 | best 5M pure Q |
| **`heal_kl_trust_400m`** | **400M** | **TBD** | **TBD** | mainline |

Inventory heal DNA: ~**41%** quantized (~308M / ~752M).  
All-eligible: ~**66%**.

### 5.2 Matched ~5.2M ablations

| Run | PPL | Highlight |
|-----|----:|-----------|
| CE skip-GDN `full_smoke` | ~60.6 | scope baseline |
| CE all-Linear | ~79.4 | scope loss |
| `scout_kl_5m` | ~49.31 | KL gate PASS |
| `scout_kl_r5_5m` (LoRA r=8) | ~48.38 | hybrid + adapters |
| `scout_kl_trust_a03_5m` | **~43.34** | trust+α=0.3 PASS |

### 5.3 Recovery curves

- Historical: `full_smoke`, `heal_25m`/`50m`, `heal_kl_50m` A/B — see `RESULTS.md` §2.
- Mainline: `scripts/plot_heal_kl_trust_400m.py` on run pack → **F1**.

### 5.4 Negative results (keep)

| Result | Lesson |
|--------|--------|
| Polish-on-B @ 2e-5 | FAIL — not a substitute for length / new DNA |
| Bundle R345 (pre_rms + unit_absmean + LoRA) | FAIL — do not default |
| Static α/T-only @ 5M | null — lock recipe; stop as main lever |
| `unit_absmean` calib | destroys magnitudes |

### 5.5 Efficiency table (representational)

| Quantity | Heal DNA (skip GDN) | All eligible |
|----------|--------------------:|-------------:|
| % params in quaternary mats | ~41% | ~66% |
| Ideal index bits / Q-weight | 2 | 2 |
| Scales | per-out-channel γ | same |
| FP residuals | embed, lm_head, GDN, norms, bias | embed, lm_head, … |
| Kernel latency | **not measured (v1)** | same |

Full model: `PAPER_MATH.md` §10.

## 6. Analysis

- Bin mass healthy on successful runs (~30% ±1, ~20% ±c typical).
- Layer map / role-wise Q-error (`run_layer_map.py`) — paste when available.
- Why skip GDN (shock + recovery).
- Soft trust vs identity (5M); long-run confirmation TBD.
- Residual gap buckets (`RESULTS.md` §5.8): under-train vs irreducible Q-error vs STE.

## 7. Limitations

- Single family/size (0.8B) until 2B recipe transfer.
- PPL-first; downstream open.
- Activations not quantized; no custom pack kernel in v1.
- Hybrid model (not fully 2-bit).
- KL needs frozen teacher at train time.
- Marathon multi-session resume / data reshuffle caveats.

## 8. Conclusion

- Restate conversion + quaternary grid + heal.
- Efficiency potential vs capability evidence.
- Next: pack/kernels, scale, downstream, close gap.

## Appendix A — Theory

**→ Full content in [`PAPER_MATH.md`](PAPER_MATH.md)** (grid, \(Q\), λ, STE, loss, MSE-γ, \(\Xi\), footprint, hybrid scope).

## Appendix B — Hyperparameters

Lock from presets: `heal_kl_50m`, `scout_kl_trust_a03_5m`, `heal_kl_trust_400m` in `config.py`. Include λ_warmup=256, lr=2e-4, cosine floor 0.1 (long), seq 512, etc.

## Appendix C — Extra curves / dumps

Session ledger, layer_map JSON summaries, optional LoRA ablation detail.

---

# Figure & table inventory

| ID | Content | Fill from |
|----|---------|-----------|
| T1 | Main parity | §5.1 + 400M pack |
| T2 | 5.2M ablations | §5.2 / `RESULTS.md` |
| T3 | Hyperparams | `config.py` presets |
| T4 | Footprint | inventory + `PAPER_MATH` §10 |
| F1 | PPL vs tokens | `plot_heal_kl_trust_400m.py` |
| F2 | λ / LR schematic | method |
| F3 | Grid + threshold on \(\mathbb{R}\) | method |
| F4 | Soft-trust \(m(e)\) | method |
| F5 | Layer-map / roles | `run_layer_map.py` |
| F6 | Bin mass over train | metrics logs |

---

# Writing schedule (while 400M runs)

| Phase | Work | Status |
|-------|------|--------|
| **P0** | Outline (`PAPER.md`) + math split (`PAPER_MATH.md`) + frozen tables | **done** |
| **P1** | Prose: intro, method body from math, related work, setup, negatives | next |
| **P2** | Paste curves; update abstract/conclusion to match final ratio | after sessions |
| **P3** | Harden efficiency claim only if pack/kernel or strong parity | optional |

---

# Consistency rules

1. Math changes: update `RESEARCH.md` + code, then `PAPER_MATH.md`, then any copied equations in prose.
2. Defaults in text must match mainline DNA unless labeled ablation.
3. Every PPL table row: scope (% Q) and Original reference.
4. BitNet = related motivation, never silent baseline.

---

# Doc map

| File | Role |
|------|------|
| [`PAPER.md`](PAPER.md) | This outline, claims, tables, writing plan |
| [`PAPER_MATH.md`](PAPER_MATH.md) | Equations + theory appendix |
| [`RESEARCH.md`](RESEARCH.md) | Code-facing method law |
| [`RESULTS.md`](RESULTS.md) | Empirical log + decision tree |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | Phases |
| [`run_pack.py`](run_pack.py) / [`scripts/plot_heal_kl_trust_400m.py`](scripts/plot_heal_kl_trust_400m.py) | Paper figures from marathon |
