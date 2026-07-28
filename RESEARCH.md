# TetraFT — Method & Mathematical Formulation

**Status:** source of truth for the quantizer design. Implementation must match this file.

## Goal

Convert a **pretrained** full-precision LLM into **2-bit quaternary** linear weights via **quantization-aware fine-tuning (QAFT)**, then **heal** so the model approaches the **original checkpoint** on language modeling and downstream tasks.

This is **conversion QAFT**, not from-scratch pretraining. Design choices optimize the **quaternary grid** \(\{-1,-c,c,1\}\) and recovery to the original—not a copy of BitNet ternary training.

BitNet / b1.58 is **related work only** (motivation for extreme discrete weights + STE). It is not an experimental competitor.

---

## 1. Quaternary grid

Weights on each quantized linear layer use a symmetric 4-level set:

\[
\mathcal{G}_c = \{-1,\,-c,\,c,\,1\}
\]

with fixed design parameter \(c \in (0,1)\).

| Default | Notes |
|---------|--------|
| **\(c = 0.25\)** | Primary default (power-of-two friendly) |
| \(c = 0.5\) | Ablation (more even spacing under some scales) |
| Learnable \(c\) | Optional later experiment |

There is **no explicit zero**. Mid-magnitude capacity is carried by \(\pm c\).

---

## 2. Forward quantization \(Q(W)\)

Let \(W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}\) be the **latent** (trainable) full-precision weights, initialized from the pretrained layer.

### 2.1 Scale \(\gamma\) (default)

**Default:** per-output-channel absmean (recommended for conversion):

\[
\gamma_i = \frac{1}{d_{\mathrm{in}}} \sum_{j=1}^{d_{\mathrm{in}}} |W_{i,j}| + \varepsilon, \quad i = 1,\ldots,d_{\mathrm{out}}
\]

**Ablations (config-selectable):**

| `scale_mode` | Definition |
|--------------|------------|
| `absmean_channel` | above (default) |
| `absmean_tensor` | \(\gamma = \mathrm{mean}(|W|) + \varepsilon\) (scalar) |
| `absmax_channel` | \(\gamma_i = \max_j |W_{i,j}| + \varepsilon\) |
| `absmax_tensor` | \(\gamma = \max |W| + \varepsilon\) |

\(\varepsilon = 10^{-5}\) (clamp). Compute \(\gamma\) in FP32; do not backprop through \(\gamma\) (detach).

**MSE-optimal scale / grid (discussion — not default).** QuEST-style fitting chooses scale (and optionally grid) to minimize \(\|W-Q(W)\|^2\) under a model of the weight distribution, rather than absmean alone. Open design choices for TetraFT (fixed quaternary \(\mathcal{G}_c\)): optimize only \(\gamma\); jointly \(\gamma\) and \(c\); per-channel vs tensor; online each step vs periodic. **Do not** use `weight_calib=unit_absmean` as a substitute (bundle FAIL). Full discussion: `RESULTS.md` §5.10.2.

### 2.2 Segmentation

Normalize \(x = W / \gamma\) (broadcast \(\gamma\)). Threshold:

\[
t = \frac{1+c}{2}
\]

\[
\mathrm{sign\_segment}(x) =
\begin{cases}
-1 & x < -t \\
-c & -t \le x < 0 \\
\;c & 0 \le x < t \\
\;1 & x \ge t
\end{cases}
\]

\[
Q(W) = \gamma \cdot \mathrm{sign\_segment}(W / \gamma)
\]

---

## 3. Straight-through estimator (STE) and \(\lambda\)-anneal

### 3.1 Soft quantization strength \(\lambda \in [0,1]\)

Forward (training):

\[
\widetilde{W} = W + \lambda \cdot \mathrm{stopgrad}\big(Q(W) - W\big)
\]

Then \(\mathrm{Linear}(x) = x\,\widetilde{W}^\top + b\) (bias stays full precision).

| \(\lambda\) | Behavior |
|-------------|----------|
| \(0\) | Pure latent FP forward (no quant) |
| \(1\) | Full discrete forward |

**Default schedule:** linear ramp over the first fraction of steps, e.g.

\[
\lambda(s) = \min\!\left(1,\; \frac{s}{s_{\mathrm{quant\_warmup}}}\right)
\]

with `quant_warmup` enabled by default. Ablate hard quant from step 0 and delayed ramps.

### 3.2 STE modes (`ste_mode`)

Gradients flow to latent \(W\) through the STE path of \(\widetilde{W}\).

| Mode | Backward approx | Status |
|------|-----------------|--------|
| `identity` (default) | \(\partial\mathcal{L}/\partial W \approx \partial\mathcal{L}/\partial\widetilde{W}\) | ✅ implemented |
| `clip` | \(\partial\mathcal{L}/\partial W \approx (\partial\mathcal{L}/\partial\widetilde{W}) \odot \mathbb{I}(|W/\gamma| \le 1)\) | ✅ implemented |
| `trust` / soft trust | gate by **quantization error** \(|W-Q(W)|\) (not magnitude) | 📋 planned — `RESULTS.md` §5.10.1 |

**Soft trust (planned).** Let \(e = |W - Q(W)|\) (or normalized \(e/\gamma\)). With threshold \(T\) (e.g. half bin width in the normalized domain) and softness \(s>0\):

\[
m = \mathrm{clip}\!\left(1 - \frac{e}{T\cdot s},\, 0,\, 1\right),
\quad
\frac{\partial\mathcal{L}}{\partial W}
\approx
m \odot \frac{\partial\mathcal{L}}{\partial\widetilde{W}}.
\]

Hard trust (\(s\to 1\), binary \(m=\mathbb{I}(e\le T)\)) can starve high-error entries of gradient under conversion QAFT — **prefer soft**. Inspired by QuEST trust estimation (ICML 2025); we do **not** adopt their full W+A / from-scratch stack. Choose by recovery on 0.8B.

---

## 4. What gets quantized

Replace eligible `nn.Linear` with `QuantizedLinear`.

**Always skip (v1):**

- `lm_head` (and tied output if separate)
- token embeddings (`embed_tokens` / equivalent)
- vision tower / multimodal encoder
- MTP or auxiliary heads if present
- norms (not Linear)

**Language body (Qwen3.5 hybrid):** FFN (`mlp.*`), full gated attention (`self_attn.{q,k,v,o}_proj`), Gated DeltaNet (`linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`) — **module-scope is an ablation**. Default: all eligible language Linears. Phase 2 arm: `skip_linear_attn=True` leaves path component `linear_attn` in FP. Optional later: FFN-first (also skip `self_attn`).

Report **% parameters quantized** every run.

---

## 5. Training objective (recovery)

**Primary:** causal LM cross-entropy on continual pretraining data (FineWeb-Edu sample).

**Combined recovery loss** (config-driven; defaults keep pure CE):

\[
\mathcal{L}
=
\alpha\,\mathrm{CE}(y, p_s)
+
(1-\alpha)\,T^{2}\,\mathrm{KL}\big(p_t \,\|\, p_s\big)
+
\beta\cdot
\frac{1}{|\mathcal{Q}|}
\sum_{W\in\mathcal{Q}}
\big\|W - \mathrm{sg}\big(Q(W)\big)\big\|_{2}^{2}
\]

| Symbol | Config | Default | Meaning |
|--------|--------|---------|---------|
| \(\alpha\) | `distill_alpha` | `1.0` | CE weight; \(\alpha<1\) enables teacher KL |
| \(T\) | `distill_temperature` | `2.0` | Softmax temperature; KL scaled by \(T^{2}\) |
| \(\beta\) | `quant_reg_beta` | `0.0` | Grid commitment on all `QuantizedLinear` |
| \(p_t\) | frozen original FP | — | Teacher; no grad; never quantized |
| \(p_s\) | student @ current \(\lambda\) | — | TetraFT forward |
| \(\mathcal{Q}\) | quantized layers | — | Commitment average over layers |

KL is next-token, ignore_index-aware (same shift as causal LM). Scout preset: `scout_kl_5m` (\(\alpha=0.5\), \(T=2\), \(\beta=0.01\), matched \(\lambda_w=256\) vs CE skip-GDN ~60.6).

Do **not** start with chat-only SFT for the main recovery claim.

### 5.1 Optional bundle adapters (scout only; off by default)

Config knobs on `QuantizedLinear` / replace (defaults preserve baseline):

| Knob | Default | Effect |
|------|---------|--------|
| `pre_rms` | `False` | RMSNorm on activations before matmul; γ init **1** |
| `weight_calib` | `none` | At replace: `unit_absmean` divides each out-channel by its absmean |
| `lora_rank` / `lora_alpha` | `0` / `None` | Residual LoRA: \(y \mathrel{+}= (\alpha/r)\,x A^\top B^\top\); **B=0** init |

Preset `scout_kl_bundle_r345_5m` enables all three (**FAIL** in practice — do not default).  
Prefer single-knob `scout_kl_r5_5m` (LoRA only). Report adapter param count; not pure weight-only quaternary when on.

---

## 6. Evaluation (parity with original)

Always evaluate under the **same** protocol as the frozen original:

1. **Held-out PPL** (fixed val split)
2. **Shock:** PPL at \(\lambda=1\), zero training steps vs original
3. **Recovery curves:** PPL vs tokens
4. Later: downstream suite (lm-eval); always include an **Original** column
5. **Optional diagnostics:** layer-wise Q-error map (`run_layer_map.py`); gradient alignment (planned — `RESULTS.md` §5.10.3)

Parity metrics:

- \(\mathrm{PPL}_Q / \mathrm{PPL}_{\mathrm{orig}}\)
- Relative task scores \(\mathrm{acc}_Q / \mathrm{acc}_{\mathrm{orig}}\) when downstream is enabled
- Gradient alignment \(\Xi\) (cosine of STE grads vs FP-forward grads) when enabled — **not** a parity claim, diagnostic only

---

## 7. Design principles (quaternary-optimal)

1. Optimize **gap to original**, not similarity to ternary methods.
2. Prefer scales/thresholds that fit **pretrained weight histograms**.
3. Use \(\lambda\)-anneal and module scope as first-class conversion tools.
4. Activation quantization / custom kernels are **out of scope for v1 capability runs**.
5. Keep equations, `config.py` defaults, and tests aligned.

---

## 8. Code status (Phase 0 complete)

All Phase-0 code debt resolved (see `RESEARCH_PLAN.md`):

| Spec (this file) | Code status |
|------------------|-------------|
| Default `absmean_channel` | ✅ `compute_scale()` dispatches 4 modes; default `absmean_channel` |
| Default `c=0.25` everywhere | ✅ Unified default `c=0.25` in `config.py`, `quantize.py`, `model.py` |
| STE `identity` + optional `clip` | ✅ `ste_mode` flag in `QuantizedLinear.__init__`; both modes implemented |
| Selective module policy | ✅ Vision/MTP skip patterns, replace report with `%` quantized |
| `config.py` model target | ✅ Default `Qwen/Qwen3.5-0.8B-Base` with `scale_mode`, `ste_mode`, `quant_warmup_steps` |
| λ anneal | ✅ Config-driven `quant_warmup_steps`; formula `min(1, step/warmup)` matches §3.1 |

---

## 9. Empirical baselines

Kaggle PPL numbers, schedule lessons, and **architectural next steps** live in **`RESULTS.md`** (not duplicated here). Keep this file as math/method law; update `RESULTS.md` when runs finish.
