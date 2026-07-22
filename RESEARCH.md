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

| Mode | Backward approx |
|------|-----------------|
| `identity` (default to implement first) | \(\partial\mathcal{L}/\partial W \approx \partial\mathcal{L}/\partial\widetilde{W}\) |
| `clip` | \(\partial\mathcal{L}/\partial W \approx (\partial\mathcal{L}/\partial\widetilde{W}) \odot \mathbb{I}(|W/\gamma| \le 1)\) |

Choose by recovery stability on 0.8B, not by external papers.

---

## 4. What gets quantized

Replace eligible `nn.Linear` with `QuantizedLinear`.

**Always skip (v1):**

- `lm_head` (and tied output if separate)
- token embeddings (`embed_tokens` / equivalent)
- vision tower / multimodal encoder
- MTP or auxiliary heads if present
- norms (not Linear)

**Language body (Qwen3.5 hybrid):** FFN, full gated attention projections, Gated DeltaNet projections — **module-scope is an ablation**. Start with all eligible language Linears; if unstable, FFN-first.

Report **% parameters quantized** every run.

---

## 5. Training objective (recovery)

**Primary:** causal LM cross-entropy on continual pretraining data (FineWeb-Edu sample).

**Optional (if CE plateaus):** KL distillation toward a **frozen original** teacher on the same tokens.

Do **not** start with chat-only SFT for the main recovery claim.

---

## 6. Evaluation (parity with original)

Always evaluate under the **same** protocol as the frozen original:

1. **Held-out PPL** (fixed val split)
2. **Shock:** PPL at \(\lambda=1\), zero training steps vs original
3. **Recovery curves:** PPL vs tokens
4. Later: downstream suite (lm-eval); always include an **Original** column

Parity metrics:

- \(\mathrm{PPL}_Q / \mathrm{PPL}_{\mathrm{orig}}\)
- Relative task scores \(\mathrm{acc}_Q / \mathrm{acc}_{\mathrm{orig}}\) when downstream is enabled

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
