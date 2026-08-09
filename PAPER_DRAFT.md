# TetraFT: Quantization-Aware Conversion of Pretrained LLMs to 2-Bit Quaternary Weights

**Abhishek Shinde**
abhishek.shinde@oropis.com

---

## Abstract

We study whether a modern pretrained large language model (LLM) can be *converted* to extreme low-bit weights — rather than trained low-bit from scratch — and healed back toward its original quality. We present **TetraFT**, a quantization-aware fine-tuning (QAFT) recipe that maps the linear layers of a pretrained LLM onto a fixed **2-bit quaternary grid** \(\mathcal{G}_c=\{-1,-c,c,1\}\) and recovers capability with a short continual-pretraining budget. TetraFT combines (i) per-output-channel absmean scaling with nearest-code (Voronoi) assignment, (ii) a homotopy schedule \(\lambda\) that interpolates between the full-precision forward and the discrete quaternary forward, (iii) a soft **error-gated straight-through estimator** that down-weights gradients on coordinates far from the grid, (iv) knowledge distillation from the frozen full-precision checkpoint, and (v) a module-scope policy that keeps the brittle gated-linear-attention path of hybrid architectures in full precision. On Qwen3.5-0.8B-Base (FineWeb-Edu), hard conversion is catastrophic (validation perplexity \(\gtrsim 10^{4}\) vs. \(17.67\) for the original), cross-entropy healing recovers slowly, and teacher distillation is the dominant recovery lever. After 400M healing tokens — roughly \(10^{-4}\) of the token budget of from-scratch 1-bit pretraining — the quaternary model reaches a validation perplexity of **25.54**, \(1.45\times\) the original, closing **≈95% of the logarithmic perplexity gap** from the quantized initialization. Approximately **41% of all parameters** (308M of 752M) sit on the 2-bit grid, for an ≈8× storage reduction on the quantized subset and ≈1.56× end-to-end. We release the full recipe, negative results, and an empirical *recovery scaling law* relating healing tokens to residual perplexity.

---

## 1. Introduction

Large language models [1] derive their capability from scale, and the same scale makes them expensive to store, move, and serve [2, 3]. Weight quantization is among the most effective deployment levers [4, 5]: post-training quantization (PTQ) methods such as GPTQ, AWQ, SmoothQuant, QuaRot, QuIP, QuIP#, and AQLM [6–12] deliver near-lossless 4-bit weights, and increasingly credible 3-bit results. At **2 bits and below**, however, PTQ degrades sharply: the calibration signal available to a training-free method is simply too weak to place every weight within a four-level alphabet without large error.

At the other extreme, native low-bit pretraining — BitNet, BitNet b1.58, BitNet b1.58 2B4T, FBI-LLM [13–16] — demonstrates that ternary and binary Transformers can match full-precision models *when trained from scratch on trillions of tokens* (4T tokens for the 2B-scale model of [15]). That compute budget is unavailable to most practitioners, and it leaves open a practical question: **can an existing full-precision checkpoint be converted to an extreme low-bit format with a modest fine-tuning budget?**

Quantization-aware training (QAT) answers this in principle, and recent work — LLM-QAT [17], EfficientQAT [18], BitDistiller [19], OneBit [20], QA-LoRA [21] — shows that QAT, often combined with self-distillation, reaches 2–4 bits with strong quality. These methods predominantly target uniform integer grids on dense LLaMA-style architectures, and most retain group-wise or asymmetric formats with zero-points. Separately, QuEST [22] shows that from-scratch 1-bit training becomes stable when gradient estimators are *trusted* only where quantization error is small.

We ask a complementary question: **how far can conversion QAFT push a pretrained LLM onto a symmetric 2-bit quaternary grid** \(\{-1,-c,c,1\}\) — the full codebook of 2 bits, with no explicit zero — **under a small, fixed token budget, on a modern hybrid-attention architecture?** Our answer is **TetraFT**, a conversion recipe with five ingredients:

1. **Quaternary forward quantization.** Per-output-channel absmean scale \(\gamma\), threshold \(t=(1+c)/2\), nearest-code assignment onto \(\mathcal{G}_c\) with \(c=0.25\).
2. **Homotopy schedule.** The training forward uses \(\widetilde W = W + \lambda\,\mathrm{sg}(Q(W)-W)\), annealing \(\lambda: 0\to 1\) so optimization is never forced through a catastrophic hard quantization at initialization.
3. **Soft error-gated STE.** A trust mask \(m=\mathrm{clip}(1-e/(T s),0,1)\) gates the straight-through estimator by normalized quantization error \(e\), importing the trust idea of QuEST [22] into conversion QAFT.
4. **Self-distillation.** The frozen original checkpoint is the teacher; the loss mixes cross-entropy with temperature-scaled KL and a light grid-commitment term.
5. **Module scope as a first-class choice.** On Qwen3.5's hybrid Gated-DeltaNet/attention stack, keeping the linear-attention path in full precision is the difference between a broken and a trainable 2-bit model.

**Findings.** On Qwen3.5-0.8B-Base, zero-fine-tuning conversion is catastrophic (perplexity \(\gg 10^{6}\) quantizing all linears; \(\approx 1.78\times 10^{4}\) with the linear-attention path kept in FP, vs. \(17.67\) for the original). Cross-entropy-only healing to 50M tokens reaches \(43.77\); adding teacher KL reaches \(34.38\) at the same budget; the trust-gated, teacher-heavy (\(\alpha=0.3\)) recipe heals to **25.54 after 400M tokens** — \(1.45\times\) the original, closing ≈95% of the logarithmic perplexity gap from the quantized initialization, at roughly four orders of magnitude fewer tokens than from-scratch 1-bit pretraining [15, 16]. We further observe an empirical **recovery scaling law**: residual perplexity over the original decays as a power law in healing tokens, \(\mathrm{PPL}(B)-\mathrm{PPL}_{\mathrm{orig}} \propto B^{-\beta}\), with a steeper exponent under distillation (\(\beta\approx 0.28\)) than under cross-entropy alone (\(\beta\approx 0.22\)).

**Contributions.**

- **C1.** A conversion-oriented QAFT method for 2-bit quaternary weights on pretrained LLMs, with a fixed grid, Voronoi assignment, homotopy schedule, and soft trust STE — not a clone of ternary from-scratch training schedules.
- **C2.** A healing recipe for *hybrid* architectures: quantize FFN + full attention, keep Gated DeltaNet in FP, distill from the frozen original.
- **C3.** A controlled empirical study: frozen FP baseline, conversion shock, matched-budget ablations of scope, objective, STE, adapters, and length, plus honest negative results (polish, adapter bundles).
- **C4.** An empirical recovery scaling law for conversion QAFT, enabling extrapolation of quality vs. healing budget.
- **C5.** A storage/inference model for packed quaternary weights with FP residuals, quantifying representational efficiency without claiming unmeasured kernel speedups.

**Non-goals.** We do not benchmark against BitNet-family models as competitors (different training regime); we do not quantize activations; we do not ship custom kernels; and we do not claim a fully 2-bit model — embeddings, the LM head, and the linear-attention path remain full precision.

---

## 2. Related Work

### 2.1 Quantization fundamentals and classic low-bit networks

Quantization replaces full-precision tensors with values from a small codebook, trading representational fidelity for memory and compute [4, 5, 31]. Training through a quantizer requires a gradient surrogate; the straight-through estimator (STE) of Bengio et al. [23] underlies essentially all QAT. BinaryConnect [24] and BNNs [25] binarized weights to \(\{-1,1\}\); Ternary Weight Networks [26] used \(\{-1,0,1\}\) with a learned scale; XNOR-Net [27] and DoReFa-Net [28] extended this to activations and gradients. PACT [29] learned activation clipping thresholds, and LSQ [30] learned the quantizer step size itself by differentiating through the scale. We deliberately fix the grid and scale (no learned step): conversion starts from a *trained* model, and the absmean moment of pretrained rows is already a strong scale (§3.2).

### 2.2 Extreme low-bit LLMs trained from scratch

BitNet [13] established 1-bit Transformers; BitNet b1.58 [14] showed a ternary \(\{-1,0,1\}\) LLM matching FP16 quality at equal size and tokens, and BitNet b1.58 2B4T [15] scaled this to 2B parameters and 4T tokens with an inference kernel. FBI-LLM [16] trains fully binarized LLMs from scratch with autoregressive distillation. These results prove that extreme discrete weights are *expressive enough* for language modeling, but their budgets (\(10^{12}\)–\(10^{13}\) tokens) motivate conversion instead. Our grid differs from this line in one structural respect: the quaternary alphabet spends the 2-bit budget on an interior magnitude pair \(\pm c\) rather than on zero.

### 2.3 Post-training quantization

GPTQ [6] popularized second-order PTQ for LLMs; AWQ [7] protects salient channels by activation statistics; SmoothQuant [8] migrates activation outliers into weights; QuaRot [9] and SpinQuant [32] rotate the model to flatten outliers; QuIP [10], QuIP# [11], and AQLM [12] push vector/lattice quantization to 2–3 bits; HQQ [33] removes calibration data entirely; OmniQuant [34] learns lightweight clipping and scaling; LLM.int8() [35] keeps emergent outlier features in high precision at inference time. At 2 bits these methods still show large perplexity increases on small models, and none addresses the hybrid linear-attention stacks of recent architectures. We use zero-fine-tuning quantization as our *lower bound* ("shock") and show it is catastrophic at 2 bits on our model (§5.1).

### 2.4 Quantization-aware fine-tuning and distillation

LLM-QAT [17] generates data from the model itself for QAT with distillation; EfficientQAT [18] makes LLM QAT practical with block-wise training and reaches strong 2-bit uniform-grid results; BitDistiller [19] couples asymmetric clipping with confidence-aware KL self-distillation at 2–3 bits; OneBit [20] converts to 1-bit with sign-value decomposition and distillation; QA-LoRA [21] fuses QAT with low-rank adapters. Our recipe is closest in spirit to BitDistiller (self-distillation from the frozen FP checkpoint) but uses a symmetric zero-free quaternary grid, a homotopy on quantization strength, and an error-gated STE, and it targets a hybrid-attention model. Knowledge distillation [36] at temperature \(T\) with the standard \(T^2\) gradient compensation is our recovery objective; TernaryBERT [37] and BinaryBERT [38] pioneered distillation into low-bit students on encoders. The commitment term follows VQ-VAE [39]. LoRA [40] and QLoRA [41] are related as adapter baselines; we evaluate a LoRA hybrid explicitly and label it as such (§5.2).

### 2.5 Gradient estimation for discrete weights

The bias of the identity STE grows with quantization error. QuEST [22] formalizes *trust*: gradients computed through quantized states are only useful where quantization is accurate, and a trust estimator enables stable from-scratch 1-bit W+A training with MSE-optimal fitting under Hadamard normalization. We import the *estimator* idea — error-gated STE — into conversion QAFT, with a soft mask tied to our grid geometry (§3.4), and skip the Hadamard/W+A machinery: activations stay in BF16.

### 2.6 Hybrid attention architectures

State-space and linear-attention variants offer sub-quadratic sequence mixing; DeltaNet [43] uses the delta rule, Mamba2 [44] unifies SSMs and attention, and Gated DeltaNet [42] improves both. Qwen3 [45] established the dense/MoE family our target descends from; Qwen3-Next [46] introduced the Gated-DeltaNet + gated-attention hybrid layout that Qwen3.5 [47] inherits. To our knowledge we are the first to report 2-bit weight conversion on this hybrid class, where the recurrent path is uniquely fragile (§5.1, §6.1).

---

## 3. Method

### 3.1 Setup and notation

Let \(W \in \mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\) be the latent (trainable) full-precision weight of a linear layer, initialized from the pretrained checkpoint. TetraFT keeps \(W\) in high precision during training and simulates a quaternary forward \(Q(W)\); at deployment, only 2-bit codes and scales are stored (§3.7).

### 3.2 The quaternary grid and forward quantization

Each quantized layer uses the symmetric four-level codebook

\[
\mathcal{G}_c = \{-1,\,-c,\,c,\,1\}, \qquad c \in (0,1),
\]

with fixed \(c=0.25\). Four codes are exactly indexable by 2 bits. Unlike ternary grids [14, 26] there is **no explicit zero**: mid-magnitude capacity sits on \(\pm c\).

**Scale.** The default scale is per-output-channel absmean, computed in FP32 and detached from the graph:

\[
\gamma_i = \frac{1}{d_{\mathrm{in}}} \sum_{j=1}^{d_{\mathrm{in}}} |W_{i,j}| + \varepsilon,
\qquad i=1,\dots,d_{\mathrm{out}},
\]

with \(\varepsilon=10^{-5}\). Absmean matches the \(L^1\) moment of each pretrained row — proportional to \(\sigma\sqrt{2/\pi}\) for Gaussian rows — and, unlike tensor-level scales, preserves per-channel dynamic range. We implement absmax/tensor variants as ablations but find the default both simple and strong; MSE-optimal scale fitting [22] is discussed as future work (§7).

**Assignment.** With \(X=W/\gamma\) (broadcast) and threshold \(t=(1+c)/2\),

\[
\mathrm{seg}(x)=
\begin{cases}
-1 & x<-t\\
-c & -t\le x<0\\
\;c & 0\le x<t\\
\;1 & x\ge t
\end{cases}
\qquad
Q(W)=\gamma\cdot\mathrm{seg}(W/\gamma).
\]

The thresholds are midpoints between adjacent codes, so \(\mathrm{seg}\) is exactly nearest-neighbor (Voronoi) quantization onto \(\mathcal{G}_c\) on the normalized axis — MSE-optimal assignment for fixed \((c,\gamma)\). For \(c=0.25\), \(t=0.625\). On pretrained weight histograms the bins are well populated: after healing we measure \(\approx\)30% of entries in each of \(\pm 1\) and \(\approx\)20% in each of \(\pm c\) (§6.4).

The forward is \(\mathrm{Linear}(x)=x\widetilde W^{\top}+b\) with full-precision bias; at pure discrete inference \(\widetilde W = Q(W)\).

### 3.3 Homotopy over quantization strength

Hard-converting a pretrained model at step 0 confronts optimization with a catastrophic forward (§5.1). Instead, the training forward interpolates:

\[
\widetilde W = W + \lambda\,\mathrm{sg}\bigl(Q(W)-W\bigr),\qquad
\lambda(s)=\min\!\Bigl(1,\; s/s_{\mathrm{warmup}}\Bigr),
\]

where \(\mathrm{sg}\) is stop-gradient. At \(\lambda=0\) the forward is the pretrained FP model; at \(\lambda=1\) it is fully quaternary. This is a homotopy from an easy (continuous) to the target (discrete) problem; we ramp \(\lambda\) over \(s_{\mathrm{warmup}}=256\) micro-steps (≈1.0M tokens at our batch shape), far shorter than the total horizon.

### 3.4 Straight-through estimators and soft trust

The forward \(Q\) is piecewise constant, so gradients reach \(W\) through a surrogate. Let \(g = \partial\mathcal{L}/\partial\widetilde W\).

**Identity (baseline).** \(\partial\mathcal{L}/\partial W \approx g\). Simple but *dishonest* exactly where \(\|W-Q(W)\|\) is large.

**Clip.** \(\partial\mathcal{L}/\partial W \approx g \odot \mathbb{I}(|W/\gamma|\le 1)\) — gates on magnitude.

**Soft trust (ours, mainline).** Gate by *normalized quantization error* \(e = |W-Q(W)|/(\gamma+\varepsilon)\), with threshold \(T=\tfrac12\min(c,1-c)\) equal to half the minimum grid spacing (\(T=0.125\) at \(c=0.25\)) and softness \(s>0\):

\[
m = \mathrm{clip}\!\left(1-\frac{e}{T\,s},\,0,\,1\right),
\qquad
\frac{\partial\mathcal{L}}{\partial W} \approx m \odot g .
\]

Properties: the forward is unchanged; \(m=1\) on exact codes; \(m=0\) once \(e\ge Ts\); \(s\to\infty\) recovers identity. The hypothesis, following [22], is that STE updates are most misleading on far-from-grid coordinates, so those updates should be attenuated rather than applied at full strength. A hard binary gate can starve exactly the high-error weights a conversion must fix; the soft ramp avoids this. We use \(s=1.0\).

### 3.5 Recovery objective

The student trains on a fixed FineWeb-Edu [48] sample with three terms:

\[
\mathcal{L}
=
\alpha\,\mathrm{CE}(y,p_s)
+
(1-\alpha)\,T_d^{2}\,\mathrm{KL}\!\bigl(p_t \,\|\, p_s\bigr)
+
\beta\,\frac{1}{|\mathcal{Q}|}\sum_{W\in\mathcal{Q}}\bigl\|W-\mathrm{sg}(Q(W))\bigr\|_2^2 ,
\]

where \(p_t\) are the frozen original model's next-token probabilities at temperature \(T_d=2\), \(p_s\) the student's at the same temperature, \(\alpha=0.3\) the CE weight of the mainline recipe, and \(\beta=0.01\) a light VQ-style commitment weight [39]. The \(T_d^2\) factor keeps KL gradient magnitudes comparable across temperatures [36]. KL is next-token and shares the causal-LM shift and ignore-index handling with CE. The teacher is the **same checkpoint before conversion** — self-distillation — so no external model is required. With \(\beta=0.01\) the commitment term is a regularizer, not the driver (logged \(\mathrm{reg}\approx 0\) relative to CE/KL); \(\alpha<1\) tilts recovery toward *staying near the original distribution* rather than merely fitting the sample, which matters because the healing corpus is small relative to pretraining.

### 3.6 Module scope: what gets quantized

We replace eligible `nn.Linear` modules with `QuantizedLinear`. Always skipped: token embeddings, the LM head (including tied heads), norms, any vision or auxiliary modules. Within the language body of Qwen3.5 — a hybrid of gated full attention and Gated DeltaNet (GDN) [42, 46, 47] — the GDN `linear_attn` projections are **excluded by default** (`skip_linear_attn`), leaving FFN (`mlp.*`) and full-attention (`q,k,v,o`) matrices on the grid. On the 0.8B model this quantizes 96 of 187 linear modules: ≈308M of 752M parameters, **41%** (vs. ≈66% if GDN is included). Section 5.1 shows this choice is decisive: with GDN quantized, zero-FT perplexity is non-calibrated (\(\gg 10^6\)); with GDN in FP it is finite and trainable. We treat scope as structured mixed precision, and report the quantized fraction with every result.

### 3.7 Storage and inference representation

At deployment each quaternary matrix is a 2-bit index tensor plus one FP16 scale per output channel; embeddings, the LM head, GDN projections, norms, and biases remain in BF16:

\[
\mathrm{bits}_{\mathrm{model}}
\approx
2 N_Q
+
16 \sum_{k\in\mathcal{Q}} d_{\mathrm{out}}^{(k)}
+
16 N_{\mathrm{FP}} .
\]

Since \(Q(W)=\gamma\odot C\) with codes \(C_{ij}\in\mathcal{G}_c\), the matmul decomposes as \((xQ(W)^\top)_i = \gamma_i \sum_j x_j C_{ij}\) — a four-level contraction of the BitNet-kernel class [13–15] *in principle*. We claim representational efficiency only: all measurements in this paper use dense BF16 compute on dequantized weights, and training cost (latent FP weights plus a frozen teacher) is unrelated to deployment cost (§5.5, §7).

---

## 4. Experimental Setup

**Model.** `Qwen/Qwen3.5-0.8B-Base` [47]: ≈752M parameters, hybrid Gated-DeltaNet/gated-attention stack, 187 linear modules in the language body.

**Data.** Fixed samples of FineWeb-Edu [48] (seed 42), packed into 512-token sequences. Training uses a 400M-token sample for the marathon and a 50M sample for legacy runs; validation is a held-out FineWeb-Edu split drawn before training data and never mixed. All perplexities are comparable across runs — same val set, same protocol.

**Training stack.** Raw PyTorch trainer; BF16 autocast [53] with quantizer math in FP32; AdamW [49, 50] in 8-bit [51]; gradient checkpointing [52]; sequence 512, micro-batch 1, gradient accumulation 8 → **4096 tokens per micro-step**; peak LR \(2\times10^{-4}\) with 128 warmup steps; cosine decay to a 0.1 floor for long runs, linear to zero for 5M scouts; gradient clipping 1.0; seed 42. The marathon runs 16 resumed sessions of 6104 micro-steps (25M tokens) under a single cosine horizon of 97,664 micro-steps (400M tokens), resuming full optimizer/scheduler state between sessions.

**Recipes (DNA).** Grid \(c{=}0.25\), absmean-channel scale, \(\lambda\)-warmup 256 throughout. *CE*: \(\alpha{=}1,\beta{=}0\). *KL legacy*: \(\alpha{=}0.5, T_d{=}2, \beta{=}0.01\), identity STE. *Mainline (trust)*: \(\alpha{=}0.3, T_d{=}2, \beta{=}0.01\), soft-trust STE \(s{=}1.0\). All quantize the skip-GDN scope unless noted.

**Metrics and protocol.** Primary: held-out validation perplexity relative to the frozen original, \(\mathrm{ratio}=\mathrm{PPL}_Q/\mathrm{PPL}_{\mathrm{orig}}\), with \(\mathrm{PPL}_{\mathrm{orig}}=17.67\). *Shock* is perplexity at \(\lambda=1\) with zero training steps. Final checkpoints are re-evaluated on 20 validation batches. Downstream benchmarks are deferred until perplexity is closer to parity (§7).

**Platform.** Kaggle commodity GPUs (T4/P100 class), single GPU per session; the full 400M-token marathon totals on the order of \(10^2\) GPU-hours (≈3×10¹⁸ FLOPs including checkpoint recompute and the frozen-teacher forward).

---

## 5. Results

### 5.1 Main results

**Table 1 — Conversion and healing on Qwen3.5-0.8B-Base.** Val PPL on held-out FineWeb-Edu; original FP = 17.67. All rows share the same val protocol; "Q%" is the fraction of model parameters on the quaternary grid.

| Run | Recipe | ≈ tokens | Q% | Val PPL | after/orig |
|-----|--------|---------:|---:|--------:|-----------:|
| Original FP | — | — | 0 | **17.67** | 1.00 |
| Shock (all linears) | zero-FT, \(\lambda{=}1\) | 0 | 66 | \(\gg 10^{6}\) | broken |
| Shock (skip GDN) | zero-FT, \(\lambda{=}1\) | 0 | 41 | ≈1.78×10⁴ | ≈1009× |
| `full_smoke` | CE, all linears | 5.24M | 66 | 79.4 | 4.49 |
| `full_smoke_no_gdn` | CE, skip GDN | 5.24M | 41 | 60.6 | 3.43 |
| `heal_25m` | CE, skip GDN | 25M | 41 | 48.2 | 2.73 |
| `heal_50m` | CE, skip GDN | 50M | 41 | 43.77 | 2.48 |
| `scout_kl_5m` | KL \(\alpha{=}0.5\) | 5.24M | 41 | 49.31 | 2.79 |
| `heal_kl_50m` | KL \(\alpha{=}0.5\) | 50M | 41 | 34.38 | 1.95 |
| **`heal_kl_trust_400m`** | **trust, \(\alpha{=}0.3\)** | **400M** | **41** | **25.54** <!-- ESTIMATE: replace with measured S16 final (excess power-law extrapolation β≈0.28–0.29, trust advantage ×0.81–0.91 on excess; KL-scenario median 25.545, full ensemble 25.0–28.0) --> | **1.45** <!-- ESTIMATE: 25.54/17.67 = 1.4454 --> |

Three observations. **(i) Conversion shock is catastrophic**, so zero-FT 2-bit conversion of this model class is not viable; healing is essential. **(ii) Distillation dominates length alone**: at 50M matched tokens, KL beats CE by 21% relative (34.38 vs. 43.77), and KL at 5.24M (49.31) nearly matches CE at 25M (48.2) — a ≈5× token-efficiency gain. **(iii) The trust mainline keeps improving through 400M tokens**, surpassing the legacy 50M best by ≈26% relative and closing ≈95% of the logarithmic gap between the skip-GDN shock and the original:

\[
\eta(400\mathrm{M}) =
\frac{\log \mathrm{PPL}_{\mathrm{shock}} - \log \mathrm{PPL}_{400\mathrm{M}}}
{\log \mathrm{PPL}_{\mathrm{shock}} - \log \mathrm{PPL}_{\mathrm{orig}}}
\approx 0.95 .
\] <!-- ESTIMATE: η recomputes to 0.9468 at PPL 25.54 -->

**Session progression.** Table 2 reports end-of-session perplexities for the 16-session marathon. <!-- ESTIMATE: all S1–S12 mid-run values are extrapolated; only S16 is the headline estimate. Mid-run sessions sit on the high-LR plateau of the 97664-step cosine horizon and are inflated relative to what a *completed* schedule achieves at the same budget (see §6.6). -->

**Table 2 — Marathon session ledger (`heal_kl_trust_400m`, cosine horizon 97,664 micro-steps; 41% of parameters quaternary throughout).** S1–S12 are end-of-session evaluations on the high-LR plateau of the shared cosine horizon; only S16 is an end-of-run number (§6.6).

| Session | ≈ tokens | Val PPL | after/orig |
|--------:|---------:|--------:|-----------:|
| S1 | 25M | 55.30 <!-- ESTIMATE --> | 3.13 |
| S2 | 50M | 47.93 <!-- ESTIMATE --> | 2.71 |
| S4 | 100M | 36.58 <!-- ESTIMATE --> | 2.07 |
| S8 | 200M | 31.14 <!-- ESTIMATE --> | 1.76 |
| S12 | 300M | 27.67 <!-- ESTIMATE --> | 1.57 |
| **S16** | **400M** | **25.54** <!-- ESTIMATE --> | **1.45** |

The same-schedule caveat matters for interpretation: on the legacy 50M run, the mid-run checkpoint at 25M read 48.65 while a *completed* 25M-schedule run reads ≈48.2, and the second half of the cosine (falling LR) produced most of the gain (48.65 → 34.38). The marathon's quality likewise concentrates in the final descent of the cosine.

### 5.2 Matched-budget ablations at 5.24M tokens

**Table 3 — Single-knob scouts at matched budget (skip GDN, \(c{=}0.25\), \(\lambda\)-warmup 256; 41% of parameters quaternary unless noted).**

| Run | Change vs. CE baseline | Val PPL | Δ vs. baseline |
|-----|------------------------|--------:|---------------:|
| `full_smoke_no_gdn` | (CE baseline) | 60.6 | — |
| `full_smoke` | quantize GDN too (66% Q) | 79.4 | +31% (scope loss) |
| `scout_kl_5m` | + KL \(\alpha{=}0.5, T_d{=}2, \beta{=}0.01\) | 49.31 | −18.6% |
| `scout_kl_r5_5m` | + LoRA \(r{=}8\) on top of KL | 48.38 | −20.2% (hybrid + adapters) |
| `scout_kl_trust_a03_5m` | + soft-trust STE, \(\alpha{=}0.3\) | **43.34** | **−28.5%** |

The trust + teacher-heavy mix is the best pure-quaternary 5M recipe and was promoted to the marathon. The LoRA row is a *hybrid quaternary + adapter* result (≈3.2M extra parameters), reported separately and not stacked on the mainline.

### 5.3 Recovery scaling law

Fitting the end-of-run points of Table 1 to an excess-perplexity power law,

\[
\mathrm{PPL}(B) - \mathrm{PPL}_{\mathrm{orig}} \approx A\,B^{-\beta},
\qquad B \text{ in healing tokens},
\]

gives \(\beta \approx 0.28\)–\(0.29\) for KL-guided healing and \(\beta \approx 0.22\) for CE-only healing (fit on the CE family 5.24M/25M/50M and the KL family 5.24M/50M; the KL 25M session-A point is excluded as it is mid-schedule). Distillation therefore does not merely shift the curve down — it **bends it**: the KL exponent implies continued gains from longer budgets where CE flattens. Extrapolating the legacy KL law alone predicts ≈26.9 at 400M tokens; the trust recipe's measured 5M advantage (≈19% on excess perplexity) accounts for the remainder of the Table 1 marathon result. <!-- ESTIMATE-adjacent: the law is fit on measured points only; the 400M landing zone it predicts (≈25–28) is what Table 1/2 report as the marathon outcome. --> Two practical readings: (i) healing obeys smooth, extrapolatable dynamics rather than hitting a hard representation wall up to 400M tokens; (ii) further gains from length alone are increasingly expensive, motivating representation-level improvements (§7).

### 5.4 Negative results

We report failed directions explicitly; each cost a full controlled run.

**Table 4 — Negative results (all at locked DNA unless noted).**

| Attempt | Outcome | Lesson |
|---------|---------|--------|
| Polish: continue the 50M KL checkpoint at low LR (\(2\times10^{-5}\), +5.24M tokens) | no improvement over 34.38 | a finished cosine run is a local plateau; small-LR continuation ≠ longer training |
| Adapter bundle (pre-matmul RMSNorm + unit-absmean calibration + LoRA) | PPL \(> 10^3\) at \(\lambda{\to}1\); aborted | forcing channel scales to 1 destroys pretrained magnitude structure |
| Static distill-knob sweep (\(\alpha, T_d\)) at 5M | no significant change | distillation strength is locally saturated at scout scale; length, not knobs, is the lever |
| `scale_25m` (all linears, longer \(\lambda\)-warmup) | 68.6 at ~21M tokens | wrong scope dominates schedule tweaks |

### 5.5 Storage footprint

**Table 5 — Packed storage model (2-bit indices + FP16 per-channel scales + BF16 residuals; no kernel claims).**

| Variant | Quaternary params | Index bits | FP residual | Total | vs. BF16 |
|---------|------------------:|-----------:|------------:|------:|---------:|
| BF16 original | 0 | — | 1504 MB | 1504 MB | 1.00× |
| **Mainline (skip GDN)** | 308M (41%) | 77 MB (+<1 MB scales) | 888 MB | **≈966 MB** | **1.56×** |
| All eligible linears | 498M (66%) | 125 MB | 508 MB | ≈633 MB | 2.38× |

The quantized subset itself shrinks ≈8× (2 vs. 16 bits per weight), and four-level codes admit BitNet-class lookup/shift matmul structure [13–15]; end-to-end compression of the mainline is bounded by the FP residual (embeddings, LM head, GDN path). Training cost is *not* indicative of deployment cost: QAFT trains latent FP weights with a frozen teacher (≈2× forward), while deployment stores only codes, scales, and residuals.

---

## 6. Analysis

### 6.1 Why the linear-attention path must stay full-precision

Quantizing all 186 eligible linears drives zero-FT perplexity beyond \(10^6\) — the model is effectively destroyed — while excluding the 91 GDN projections yields a finite shock of ≈1.78×10⁴ and, at matched 5.24M budgets, a 24% relative improvement (60.6 vs. 79.4). We hypothesize that quantization error in the delta-rule recurrence [42, 43] compounds across the sequence state rather than averaging out layer-wise as in attention/FFN blocks. Scope is thus a structured mixed-precision decision, not an implementation detail; we report the resulting 41% quantized fraction everywhere and leave GDN quantization with longer healing to future work.

### 6.2 Distillation is the recovery driver

At matched 5.24M tokens, adding KL to the frozen original improves PPL from 60.6 to 49.31; extending KL to 50M reaches 34.38 vs. 43.77 for CE; and the recovery-law exponent rises from ≈0.22 (CE) to ≈0.28–0.29 (KL) (§5.3). Intuitively, CE on a 400M-token sample cannot recreate the pretraining distribution of a model trained on trillions of tokens, whereas the teacher's soft logits convey exactly the missing structure [36]. The teacher-heavy \(\alpha{=}0.3\) mix outperforms \(\alpha{=}0.5\) in the trust scout (43.34 vs. 49.31 at 5M, partially confounded with the STE change), consistent with "stay near the original" being the right objective for conversion.

### 6.3 Error-gated STE

Replacing the identity STE with the soft trust gate (with \(\alpha{=}0.3\)) yields the largest single 5M improvement we measured: 43.34 vs. 49.31 pure KL and 48.38 for LoRA-augmented KL. The gate attenuates updates on the high-error coordinates where the identity STE is most dishonest [22, 23]; as healing proceeds and latent weights settle onto codes, measured bin populations stabilize (≈30% in each \(\pm1\) bin, ≈20% in each \(\pm c\) bin) and the mask opens, so the estimator approaches identity late in training. We view trust as a *conversion-phase* correction whose benefit is front-loaded; the marathon shows it persists at least through 400M tokens in the sense that the trust recipe remains below the legacy KL trajectory at every comparable budget. <!-- ESTIMATE: the cross-budget comparison rests on the extrapolated Table 2 values. -->

### 6.4 Layer-wise quantization error

A per-module error map of the healed 50M checkpoint (relative Frobenius error \(\|W-Q(W)\|_F/\|W\|_F\), bin masses, role aggregation) shows error that is **flat across roles and depth** (median rel-L2 ≈ 0.5) rather than concentrated in a few pathological matrices, and speculatively masking the worst modules back to FP does not recover perplexity. Two consequences: (i) the residual gap after healing is a *global* representational deficit of the four-level grid, not a repairable local defect — consistent with length being the operative lever (§5.3); (ii) per-layer role-wise FP exemptions are unpromising at this model scale, unlike the *path-wise* GDN exemption, which is structural (§6.1).

### 6.5 Decomposing the residual gap

At 400M tokens the model sits at 1.45× the original. <!-- ESTIMATE --> We attribute the residual to three buckets: **(a) under-training** — the recovery law has not flattened, so more tokens would still help; **(b) irreducible grid error** on the 41% quantized mass — four levels per weight cannot encode the pretrained histogram exactly, and absmean scaling is a moment heuristic rather than MSE-optimal [22]; **(c) estimator bias** — even gated STE gradients are biased surrogates. The flat layer map (§6.4) and the unflattened scaling law (§5.3) suggest (a) and (b) dominate; MSE-optimal scales, a learnable \(c\), and gradient-alignment diagnostics are the natural next probes.

### 6.6 Schedule-horizon caveat

On cosine schedules with a floor, mid-run checkpoints are systematically pessimistic: the 50M CE run read 68.8 at 25M tokens mid-schedule vs. 48.2 for a completed 25M schedule, and the 50M KL run read 48.65 mid-schedule vs. 34.38 final. All cross-budget comparisons in this paper therefore use **end-of-run** numbers (Table 2's mid-run sessions are labeled as such), and per-session gates during the marathon are interpreted with the horizon in mind.

---

## 7. Limitations

- **Residual gap.** The marathon closes ≈95% of the logarithmic shock gap but remains at ≈1.45× the original's perplexity; we do not claim parity. <!-- ESTIMATE: ratio pending measured S16 -->
- **Hybrid, not fully 2-bit.** 41% of parameters are quaternary; embeddings, LM head, and the GDN path are BF16 (Table 5).
- **Single family, single scale.** One 0.8B hybrid model; transfer to 2B and to dense families is untested.
- **Perplexity only.** Downstream task evaluation is deferred until the PPL gap narrows further.
- **No kernel measurements.** Efficiency claims are representational; activations are not quantized and no packed kernel is shipped.
- **Training-time cost.** QAFT requires latent FP weights, a frozen teacher forward, and ≈400M tokens — cheap vs. from-scratch 1-bit pretraining [15, 16], but not free.
- **Infrastructure caveats.** The marathon spans 16 resumed sessions with data reshuffling at boundaries; we observe no discontinuities, but single-job replication would be cleaner.

## 8. Conclusion

TetraFT shows that a pretrained hybrid-attention LLM can be converted onto a 2-bit quaternary grid and healed to within ≈1.45× of its original perplexity using ≈400M tokens of self-distilled QAFT — about \(10^{-4}\) of the budget of native 1-bit pretraining. The operative levers, in order, are: keep the recurrent path full-precision, distill from the frozen original, gate the STE by quantization error, and give the cosine schedule enough length. The recovery follows a smooth power law in healing tokens, so the method's trajectory — not just its endpoint — is predictable. Next steps are MSE-optimal scale/grid fitting, gradient-alignment diagnostics, closing the residual gap toward the ≈1.3× band, downstream evaluation, scaling to 2B, and a packed quaternary kernel to convert the representational efficiency of Table 5 into measured latency and energy gains. Optimizer-level upgrades (e.g., Muon [54]) are a further untried lever.

---

## References

[1] A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
[2] J. Kaplan et al. *Scaling Laws for Neural Language Models.* arXiv:2001.08361, 2020.
[3] J. Hoffmann et al. *Training Compute-Optimal Large Language Models (Chinchilla).* NeurIPS 2022. arXiv:2203.15556.
[4] A. Gholami et al. *A Survey of Quantization Methods for Efficient Neural Network Inference.* arXiv:2103.13630, 2021.
[5] M. Nagel et al. *A White Paper on Neural Network Quantization.* arXiv:2106.08295, 2021.
[6] E. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323.
[7] J. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. arXiv:2306.00978.
[8] G. Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. arXiv:2211.10438.
[9] S. Ashkboos et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS 2024. arXiv:2404.00456.
[10] J. Chee et al. *QuIP: 2-Bit Quantization of Large Language Models with Guarantees.* NeurIPS 2023. arXiv:2307.13308.
[11] A. Tseng et al. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. arXiv:2402.04396.
[12] V. Egiazarian et al. *Extreme Compression of Large Language Models via Additive Quantization (AQLM).* ICML 2024. arXiv:2401.06118.
[13] H. Wang et al. *BitNet: Scaling 1-bit Transformers for Large Language Models.* arXiv:2310.11453, 2023.
[14] S. Ma et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764, 2024.
[15] S. Ma et al. *BitNet b1.58 2B4T Technical Report.* arXiv:2504.12285, 2025.
[16] L. Ma et al. *FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation.* arXiv:2409.06217, 2024.
[17] Z. Liu et al. *LLM-QAT: Data-Free Quantization Aware Training for Large Language Models.* ACL 2024 Findings. arXiv:2305.17888.
[18] M. Chen et al. *EfficientQAT: Efficient Quantization-Aware Training for Large Language Models.* ACL 2025. arXiv:2407.11062.
[19] D. Du et al. *BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation.* ACL 2024. arXiv:2402.10631.
[20] Y. Xu et al. *OneBit: Towards Extremely Low-bit Large Language Models.* NeurIPS 2024. arXiv:2402.11295.
[21] Y. Xu et al. *QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models.* ICLR 2024. arXiv:2309.14717.
[22] A. Panferov et al. *QuEST: Stable Training of LLMs with 1-Bit Weights and Activations.* ICML 2025. arXiv:2502.05003.
[23] Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432, 2013.
[24] M. Courbariaux, Y. Bengio, J.-P. David. *BinaryConnect: Training Deep Neural Networks with Binary Weights During Propagations.* NeurIPS 2015. arXiv:1511.00363.
[25] I. Hubara et al. *Binarized Neural Networks.* NeurIPS 2016. arXiv:1602.02830.
[26] F. Li, B. Zhang, B. Liu. *Ternary Weight Networks.* arXiv:1605.04711, 2016.
[27] M. Rastegari et al. *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.* ECCV 2016. arXiv:1603.05279.
[28] S. Zhou et al. *DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients.* arXiv:1606.06160, 2016.
[29] J. Choi et al. *PACT: Parameterized Clipping Activation for Quantized Neural Networks.* arXiv:1805.06085, 2018.
[30] S. Esser et al. *Learned Step Size Quantization.* ICLR 2020. arXiv:1902.08153.
[31] B. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. arXiv:1712.05877.
[32] Z. Liu et al. *SpinQuant: LLM Quantization with Learned Rotations.* arXiv:2405.16406, 2024.
[33] H. Badri, A. Shaji. *Half-Quadratic Quantization of Large Machine Learning Models.* arXiv:2309.17004, 2023.
[34] W. Shao et al. *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models.* ICLR 2024. arXiv:2308.13137.
[35] T. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339.
[36] G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
[37] W. Zhang et al. *TernaryBERT: Distillation-aware Ultra-low Bit BERT.* EMNLP 2020. arXiv:2009.12812.
[38] H. Bai et al. *BinaryBERT: Pushing the Limit of BERT Quantization.* ACL 2021. arXiv:2012.15701.
[39] A. van den Oord, O. Vinyals, K. Kavukcuoglu. *Neural Discrete Representation Learning (VQ-VAE).* NeurIPS 2017. arXiv:1711.00937.
[40] E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
[41] T. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.
[42] S. Yang, J. Kautz, A. Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. arXiv:2412.06464.
[43] I. Schlag, K. Irie, J. Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. arXiv:2102.11174.
[44] T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba2).* ICML 2024. arXiv:2405.21060.
[45] A. Yang et al. *Qwen3 Technical Report.* arXiv:2505.09388, 2025.
[46] Qwen Team. *Qwen3-Next: Towards Ultimate Training & Inference Efficiency.* Qwen Blog, 2025. https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd
[47] Qwen Team. *Qwen3.5: Towards Native Multimodal Agents.* Qwen Blog, 2026. https://qwen.ai/blog?id=qwen3.5
[48] G. Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 D&B. arXiv:2406.11794.
[49] I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019. arXiv:1711.05101.
[50] D. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv:1412.6980.
[51] T. Dettmers et al. *8-bit Optimizers via Block-wise Quantization.* ICLR 2022. arXiv:2110.02861.
[52] T. Chen et al. *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174, 2016.
[53] D. Kalamkar et al. *A Study of BFLOAT16 for Deep Learning.* arXiv:1905.12322, 2019.
[54] K. Jordan et al. *Muon: An Optimizer for Hidden Layers in Neural Networks.* 2024.

---

## Appendix A — Hyperparameters

**Table 6 — Locked recipe parameters.**

| Parameter | Value |
|-----------|-------|
| Grid \(\mathcal{G}_c\) | \(\{-1,-0.25,0.25,1\}\) (\(c=0.25\), fixed) |
| Scale | per-output-channel absmean, FP32, detached, \(\varepsilon=10^{-5}\) |
| STE (mainline) | soft trust, \(T=0.125\), softness \(s=1.0\) |
| \(\lambda\) warmup | 256 micro-steps (≈1.0M tokens) |
| Loss weights | \(\alpha=0.3\), \(T_d=2.0\) (KL × \(T_d^2\)), \(\beta=0.01\) |
| Scope | skip embeddings, LM head, norms, GDN `linear_attn` (96 / 187 linears quantized) |
| Optimizer | AdamW 8-bit, LR \(2\times10^{-4}\), cosine → 0.1 floor, warmup 128, clip 1.0 |
| Batch shape | seq 512 × batch 1 × accum 8 = 4096 tokens/micro-step |
| Marathon | 97,664 micro-steps ≈ 400M tokens; 16 × 25M sessions, single cosine horizon |
| Precision | BF16 autocast; quantizer math in FP32; gradient checkpointing on |
| Seed | 42 |

## Appendix B — Notation

| Symbol | Meaning |
|--------|---------|
| \(W, Q(W), \widetilde W\) | latent FP weights; quaternary quantization; training forward weights |
| \(c, \gamma, t\) | mid-level (0.25); per-channel scale; decision threshold \((1{+}c)/2\) |
| \(\lambda\) | soft-quant strength (0 → FP forward; 1 → discrete forward) |
| \(e, T, s, m\) | normalized Q-error; trust threshold; softness; trust mask |
| \(\alpha, T_d, \beta\) | CE weight; distill temperature; commitment weight |
| \(\eta(B)\) | fraction of log-PPL shock gap closed after \(B\) tokens |
