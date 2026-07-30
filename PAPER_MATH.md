# TetraFT — Equation sheet & theory appendix

**Status:** paper math draft. Implementation law remains [`RESEARCH.md`](RESEARCH.md); this file expands derivations and discussion for the write-up.  
**Prose / outline:** [`PAPER.md`](PAPER.md).  
**Empirics:** [`RESULTS.md`](RESULTS.md).

If math and code diverge, fix `RESEARCH.md` + code first, then sync this file.

---

## 0. Notation

| Symbol | Meaning |
|--------|---------|
| \(W \in \mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\) | Latent (trainable) full-precision weights |
| \(Q(W)\) | Quaternary quantization of \(W\) |
| \(\widetilde{W}\) | Training forward weights (soft-quant path) |
| \(\mathcal{G}_c = \{-1,-c,c,1\}\) | Normalized quaternary codebook |
| \(c \in (0,1)\) | Mid-level design parameter (default \(0.25\)) |
| \(\gamma\) | Positive scale (scalar or per-output-channel) |
| \(t = (1+c)/2\) | Decision threshold on normalized axis |
| \(\lambda \in [0,1]\) | Soft-quant strength |
| \(\varepsilon\) | Numerical floor (\(10^{-5}\)) |
| \(e\) | Normalized quantization error |
| \(T\) | Trust half-spacing threshold (STE); also distill temperature in loss (context-clear) |
| \(s\) | Trust softness (`trust_softness`) |
| \(m\) | Soft trust STE mask |
| \(\alpha\) | CE weight in distill mix (`distill_alpha`) |
| \(\beta\) | Grid commitment weight (`quant_reg_beta`) |
| \(p_t, p_s\) | Teacher / student next-token distributions |
| \(\mathcal{Q}\) | Set of `QuantizedLinear` weight tensors |
| \(\Xi_\ell\) | Gradient alignment at module/block \(\ell\) |
| \(\mathrm{sg}(\cdot)\) | Stop-gradient |

**Default empirical DNA (mainline):** \(c=0.25\), `absmean_channel`, `ste_mode=trust`, \(s=1.0\), \(\alpha=0.3\), distill \(T=2\), \(\beta=0.01\), `skip_linear_attn=True`.

---

## 1. Quaternary grid

Weights on each quantized linear layer use a symmetric 4-level set:

\[
\mathcal{G}_c = \{-1,\,-c,\,c,\,1\}, \qquad c \in (0,1).
\]

**Bits.** Four codes \(\Leftrightarrow\) **2 bits** per weight index (plus scales stored in higher precision).

**No explicit zero.** Mid-magnitude capacity sits on \(\pm c\). This differs from ternary grids \(\{-1,0,1\}\) (BitNet-style) and from uniform INT2 mid-rise grids that include \(0\).

| Choice | Role |
|--------|------|
| \(c = 0.25\) | Primary default; power-of-two friendly under some scales |
| \(c = 0.5\) | Ablation (more even spacing under some normalizations) |
| Learnable \(c\) | Optional; changes bin geometry |

### 1.1 Why four levels at 2 bits

An alphabet of size \(4\) is the full codebook available at 2 bits/weight. Relative to ternary (\(3\) levels, often still stored/packed with similar systems tricks), quaternary adds an interior magnitude without spending a third bit. Relative to FP16/BF16, the information budget per weight drops by \(8\times\) before counting scale overhead.

---

## 2. Forward quantization \(Q(W)\)

### 2.1 Scale \(\gamma\)

**Default — per-output-channel absmean:**

\[
\gamma_i
=
\frac{1}{d_{\mathrm{in}}}
\sum_{j=1}^{d_{\mathrm{in}}}
\bigl|W_{i,j}\bigr|
+ \varepsilon,
\qquad
i = 1,\ldots,d_{\mathrm{out}}.
\]

**Ablations:**

| Mode | Definition |
|------|------------|
| `absmean_channel` | above (default) |
| `absmean_tensor` | \(\gamma = \mathrm{mean}(|W|) + \varepsilon\) |
| `absmax_channel` | \(\gamma_i = \max_j |W_{i,j}| + \varepsilon\) |
| `absmax_tensor` | \(\gamma = \max |W| + \varepsilon\) |

**Implementation constraints:** compute \(\gamma\) in FP32; **do not** backprop through \(\gamma\) (detach). Broadcast \(\gamma\) over the input dimension.

### 2.2 Normalization and segmentation

\[
X = W / \gamma
\quad\text{(broadcast)}.
\]

Threshold (midpoint between \(c\) and \(1\) on the positive axis; symmetric on the negative):

\[
t = \frac{1+c}{2}.
\]

\[
\mathrm{sign\_segment}(x)
=
\begin{cases}
-1 & x < -t \\
-c & -t \le x < 0 \\
\;c & 0 \le x < t \\
\;1 & x \ge t.
\end{cases}
\]

\[
Q(W) = \gamma \cdot \mathrm{sign\_segment}(W / \gamma).
\]

### 2.3 Derivation: Voronoi / nearest-code thresholds on \(\mathbb{R}\)

On the normalized line, codes are \(\mathcal{G}_c\). For squared error, the optimal assignment regions are Voronoi cells: boundaries at midpoints between adjacent codes.

Ordered codes: \(-1 < -c < c < 1\) (since \(c\in(0,1)\)).

- Midpoint \((-1,-c)\): \(\displaystyle -\frac{1+c}{2} = -t\)
- Midpoint \((-c,c)\): \(0\) (symmetric grid)
- Midpoint \((c,1)\): \(\displaystyle \frac{1+c}{2} = t\)

Hence \(\mathrm{sign\_segment}\) is exactly **nearest-neighbor quantization** onto \(\mathcal{G}_c\) in Euclidean distance on \(\mathbb{R}\), then rescaled by \(\gamma\).

**Caveat.** This is optimal for fixed \((c,\gamma)\) in 1D MSE on the *normalized* coordinate. It does **not** by itself choose the MSE-optimal pair \((c,\gamma)\) for a given weight matrix (see §8).

### 2.4 Linear forward

Bias stays full precision:

\[
\mathrm{Linear}(x)
=
x\,\widetilde{W}^{\top} + b.
\]

At pure discrete inference (\(\lambda=1\)): \(\widetilde{W}=Q(W)\).

---

## 3. Soft quantization \(\lambda\) and the training path

### 3.1 Forward (training)

\[
\widetilde{W}
=
W
+
\lambda \cdot \mathrm{sg}\bigl(Q(W) - W\bigr),
\qquad
\lambda \in [0,1].
\]

| \(\lambda\) | Behavior |
|-------------|----------|
| \(0\) | Pure latent FP forward |
| \(1\) | Fully discrete forward \(Q(W)\) |
| \((0,1)\) | Convex combination in value space with STE path through \(W\) |

Equivalently:

\[
\widetilde{W}
=
(1-\lambda)\,W + \lambda\,Q(W)
\quad\text{in value, with } Q(W) \text{ detached from the residual path}.
\]

(The implementation form with \(\mathrm{sg}(Q-W)\) yields the same forward values and the intended STE.)

### 3.2 Anneal schedule

Default linear ramp:

\[
\lambda(s)
=
\min\!\left(1,\; \frac{s}{s_{\mathrm{quant\_warmup}}}\right).
\]

With micro-step token budget \(4096\) tok/step and \(s_{\mathrm{quant\_warmup}}=256\):

\[
\text{tokens to }\lambda=1 \approx 256 \times 4096 = 1.05\times 10^{6}
\quad\text{(not \(16\)M).}
\]

**Homotopy view.** \(\lambda\) traces a continuous path from the pretrained FP forward to the discrete quaternary forward, so early optimization is not forced through a catastrophic hard quant. \(\lambda(s)\) is a **schedule**, not a learned parameter.

---

## 4. Straight-through estimators

Gradients flow to latent \(W\) through the STE approximation of \(\partial\widetilde{W}/\partial W\). Forward \(Q\) remains non-differentiable (piecewise constant).

### 4.1 Identity STE (baseline)

\[
\frac{\partial\mathcal{L}}{\partial W}
\approx
\frac{\partial\mathcal{L}}{\partial\widetilde{W}}.
\]

**Bias.** Treats \(Q\) as the identity. Coordinates with large \(\|W-Q(W)\|\) receive gradients as if they were already on-grid—classic STE dishonesty.

### 4.2 Clip STE (magnitude gate)

\[
\frac{\partial\mathcal{L}}{\partial W}
\approx
\frac{\partial\mathcal{L}}{\partial\widetilde{W}}
\odot
\mathbb{I}\!\left(\left|\frac{W}{\gamma}\right| \le 1\right).
\]

Gates on **normalized magnitude**, not quantization error. Different hypothesis from trust.

### 4.3 Soft trust STE (error gate)

Normalized error and half minimum grid spacing:

\[
e
=
\frac{\bigl|W - Q(W)\bigr|}{\gamma + \varepsilon},
\qquad
T
=
\frac{1}{2}\min\bigl(c,\, 1-c\bigr).
\]

For default \(c=0.25\): \(\min(c,1-c)=c=0.25\), so \(T=0.125\).

Soft mask with softness \(s>0\):

\[
m
=
\mathrm{clip}\!\left(
1 - \frac{e}{T\cdot s},\,
0,\,
1
\right),
\qquad
\frac{\partial\mathcal{L}}{\partial W}
\approx
m \odot \frac{\partial\mathcal{L}}{\partial\widetilde{W}}.
\]

**Properties:**

- Forward unchanged; mask only on backward.
- \(s \to \infty\) \(\Rightarrow\) \(m \to 1\) \(\Rightarrow\) identity STE.
- \(e=0\) \(\Rightarrow\) \(m=1\) (full credit on exact codes).
- \(e \ge T s\) \(\Rightarrow\) \(m=0\) (no STE update).

**Hard trust** \(m=\mathbb{I}(e\le T)\) can starve high-error entries right after conversion shock—**prefer soft** for conversion QAFT.

**Motivation (QuEST-related).** Identity STE is most wrong where \(Q\) disagrees with \(W\). Error-gating reduces updates on those coordinates. We import the *idea* of trust estimation; we do not adopt QuEST’s full from-scratch W+A stack.

### 4.4 Sketch: STE as a biased estimator

Let \(g^\star = \partial\mathcal{L}_{\mathrm{FP}}/\partial W\) be the gradient under an idealized FP forward with the same latent \(W\), and \(g_{\mathrm{STE}}\) the STE gradient under discrete forward. Then

\[
g_{\mathrm{STE}} - g^\star
\]

is the STE bias. Soft trust multiplies coordinates by \(m\le 1\), trading variance/bias: it does not cancel bias analytically, but empirically can improve recovery when identity STE is too aggressive on far-from-grid weights (`RESULTS.md` §5.10).

---

## 5. Training objective

### 5.1 Combined recovery loss

\[
\mathcal{L}
=
\alpha\,\mathrm{CE}(y,\, p_s)
+
(1-\alpha)\,T^{2}\,
\mathrm{KL}\bigl(p_t \,\|\, p_s\bigr)
+
\beta\cdot
\frac{1}{|\mathcal{Q}|}
\sum_{W\in\mathcal{Q}}
\bigl\|
W - \mathrm{sg}\bigl(Q(W)\bigr)
\bigr\|_{2}^{2}.
\]

| Term | Role |
|------|------|
| CE | Next-token fit on continual-pretrain data (FineWeb-Edu sample) |
| KL | Match frozen **original FP** teacher logits (temperature \(T\), scaled by \(T^{2}\)) |
| Commitment | Pull latent \(W\) toward discrete codes (VQ-style; sg on \(Q\)) |

- \(p_t\): teacher; no grad; never quantized.
- \(p_s\): student at current \(\lambda\).
- KL is next-token, `ignore_index`-aware (same shift as causal LM).
- Defaults in pure-CE mode: \(\alpha=1\), \(\beta=0\). Mainline KL DNA uses \(\alpha<1\), small \(\beta\).

### 5.2 Distillation scaling

Softmax at temperature \(T\):

\[
p^{(T)}(\cdot)
\propto
\exp\bigl(\mathrm{logits}/T\bigr).
\]

The factor \(T^{2}\) on KL is the standard Hinton et al. compensation so gradient magnitudes stay comparable as \(T\) grows.

**Interpretation of \(\alpha\).** \(\alpha=1\): data-only CE. \(\alpha\to 0\): pure teacher matching. Mainline \(\alpha=0.3\) puts more weight on the original model’s distribution—useful when the discrete student must stay close to the pretrained checkpoint rather than only fitting the CPT sample.

### 5.3 Commitment term

Same spirit as VQ-VAE commitment: encourage \(W \approx Q(W)\) so STE becomes less dishonest and the latent does not drift arbitrarily far from the grid.

With \(\beta=0.01\), logs often show \(\mathrm{reg}\approx 0\) relative to CE/KL—the term is a light regularizer, not the main recovery driver. Larger \(\beta\) is an open micro-ablation.

---

## 6. What gets quantized (module policy)

Replace eligible `nn.Linear` with `QuantizedLinear`.

**Always skip (v1):** `lm_head` (and tied output if separate), token embeddings, vision / multimodal encoder, MTP or auxiliary heads, norms (not Linear).

**Language body (Qwen3.5 hybrid):**

- FFN (`mlp.*`)
- Full gated attention (`self_attn.{q,k,v,o}_proj`)
- Gated DeltaNet (`linear_attn.*`) — **optional skip**

**Default heal / mainline scope:** `skip_linear_attn=True` → GDN path stays FP.

**Inventory (0.8B, approximate):**

| Scope | % params quantized |
|-------|-------------------:|
| All eligible Linear | ~**66%** |
| Skip GDN (`skip_linear_attn`) | ~**41%** |

Report **% quantized** with every quality number. Claims are for a **hybrid** model (quaternary body + FP residuals), not a fully 2-bit network.

---

## 7. Evaluation metrics

### 7.1 Parity

Always compare under the **same** val protocol as the frozen original:

\[
\mathrm{ratio}
=
\frac{\mathrm{PPL}_{Q}}{\mathrm{PPL}_{\mathrm{orig}}}.
\]

Target band for strong “near original” claims (research plan): \(\mathrm{ratio} \lesssim 1.3\).

### 7.2 Shock and recovery

- **Shock:** \(\lambda=1\), zero FT steps.
- **Recovery curves:** \(\mathrm{PPL}\) vs tokens (and vs steps after \(\lambda\to 1\)).
- **Fraction of shock closed** (optional):

\[
\eta(B)
=
\frac{
\log\mathrm{PPL}_{\mathrm{shock}} - \log\mathrm{PPL}(B)
}{
\log\mathrm{PPL}_{\mathrm{shock}} - \log\mathrm{PPL}_{\mathrm{orig}}
}
\quad\text{(define carefully if shock is infinite / unstable).}
\]

Prefer reporting raw PPL tables when all-Linear shock is non-calibrated (\(\gg 10^{6}\)).

### 7.3 Diagnostics

- Per-module \(\mathrm{rel\_L2} = \|W-Q(W)\|_F / \|W\|_F\), bin mass on \(\mathcal{G}_c\).
- Gradient alignment (planned; §9).

---

## 8. Theory appendix — scale, codebook, and MSE

### 8.1 Absmean as a moment heuristic

Per-channel absmean matches the \(L^1\) scale of each output row. Under Laplacian-like weight tails, mean absolute deviation is a natural scale; under Gaussian rows, it is proportional to \(\sigma\sqrt{2/\pi}\). It is **simple and conversion-friendly**, not MSE-optimal for \(Q(\cdot)\) onto \(\mathcal{G}_c\).

### 8.2 MSE-optimal scale (fixed \(c\))

For fixed segmentation rule and fixed \(c\),

\[
\gamma_i^\star
\in
\arg\min_{\gamma>0}
\bigl\|
W_{i,:}
-
\gamma\cdot
\mathrm{sign\_segment}(W_{i,:}/\gamma)
\bigr\|_{2}^{2}.
\]

**Note:** \(\mathrm{sign\_segment}(W/\gamma)\) depends on \(\gamma\) through thresholds, so the map \(\gamma \mapsto Q\) is piecewise; optimize by 1D search / grid / fixed-point per channel (or tensor).

**Variants (discussion — not all implemented):**

| Variant | Optimize | Freeze |
|---------|----------|--------|
| A. γ-MSE | \(\gamma\) | \(c\) |
| B. c-MSE | \(c\in(0,1)\) | scale = absmean |
| C. Joint | \((\gamma,c)\) | — |
| D. Offline | fit once at replace | then freeze |

**Failed anti-pattern:** `weight_calib=unit_absmean` forces channel scale toward \(1\) and destroys pretrained magnitudes. Any MSE fit must keep a **free** \(\gamma\) that maps codes back to weight units: \(Q=\gamma\cdot\mathrm{seg}(\cdot)\).

### 8.3 Interaction with \(\lambda\)-anneal

Fitting \(\gamma\) on latent \(W\) while the forward uses \(\widetilde{W}\) can disagree early in warmup. Offline-at-replace is safer for conversion; online refit is closer to QuEST-style training but costlier.

### 8.4 Gate by PPL, not only rel_L2

Lower weight MSE does not imply lower PPL. Any scale/grid scout must beat a locked PPL gate (`RESULTS.md`).

---

## 9. Theory appendix — gradient alignment \(\Xi\)

**Status:** definition for analysis; tool optional / planned.

For a val batch, two paths from the same student latents:

| Path | Forward | Backward |
|------|---------|----------|
| Q | \(\lambda=1\), quaternary | current STE |
| FP ref | \(\lambda=0\) (or non-quant clone) | full grad |

At module or block \(\ell\):

\[
\Xi_\ell
=
\frac{
\bigl\langle g^{(\mathrm{Q})}_\ell,\, g^{(\mathrm{FP})}_\ell \bigr\rangle
}{
\bigl\|g^{(\mathrm{Q})}_\ell\bigr\|_2\,
\bigl\|g^{(\mathrm{FP})}_\ell\bigr\|_2
}.
\]

| Pattern | Reading |
|---------|---------|
| \(\Xi\) high everywhere | STE OK; bottleneck is representation / data / budget |
| \(\Xi\) low / drops with depth | STE dishonest → trust / better \(Q\) |
| \(\Xi\) low only on some roles | role-sensitive STE or keep those mats FP |

**Not** a parity claim—diagnostic only.

---

## 10. Theory appendix — inference footprint (efficiency model)

**Honest scope.** Custom quaternary kernels and activation quant are **out of scope for v1 capability runs**. Efficiency claims are **representational** (bits, structure), BitNet-class *in principle*, not measured tokens/s unless a systems track lands.

### 10.1 Storage model

Let \(N_Q\) be the number of quantized weight elements, \(d_{\mathrm{out}}^{(k)}\) channels in layer \(k\), and \(N_{\mathrm{FP}}\) elements kept in high precision (embeds, lm_head, skipped Linears, biases, norms).

**Index packing (ideal):**

\[
\mathrm{bits}_{\mathrm{idx}}
=
2\, N_Q.
\]

**Scales** (e.g. FP16 per output channel):

\[
\mathrm{bits}_{\gamma}
\approx
16 \sum_{k \in \mathcal{Q}} d_{\mathrm{out}}^{(k)}.
\]

**Residual FP** (e.g. BF16):

\[
\mathrm{bits}_{\mathrm{FP}}
\approx
16\, N_{\mathrm{FP}}.
\]

Total:

\[
\mathrm{bits}_{\mathrm{model}}
\approx
\mathrm{bits}_{\mathrm{idx}}
+
\mathrm{bits}_{\gamma}
+
\mathrm{bits}_{\mathrm{FP}}.
\]

For heal DNA (~41% of params in \(N_Q\)), savings are large on the quantized subset but the full model is **not** \(2\)-bit end-to-end.

### 10.2 Matmul structure (aspirational)

With \(Q(W)=\gamma \odot \mathrm{codes}\), \(C_{ij}\in\mathcal{G}_c\):

\[
(x Q(W)^{\top})_i
=
\gamma_i \sum_j x_j\, C_{ij}.
\]

The sum is a **low-bit / lookup / bit-shift-style** contraction *in principle* (systems literature around BitNet-style kernels). TetraFT v1 still runs dense BF16 matmuls on dequantized or latent-path weights during training and standard eval.

### 10.3 Train vs inference cost

| Phase | Precision reality |
|-------|-------------------|
| QAFT train | Latent FP \(W\), STE, optional FP teacher (KL \(\approx 2\times\) forward) |
| Discrete eval | Forward at \(\lambda=1\); still typically dequant to BF16 in current stack |
| Deploy target | Packed indices + \(\gamma\) + FP residuals; custom kernel optional later |

Do not equate training VRAM with inference footprint.

---

## 11. Theory appendix — hybrid module scope

Qwen3.5-0.8B is a **hybrid** stack (Gated DeltaNet + full attention). Empirically:

- Quantizing **all** eligible Linears: shock PPL non-calibrated (\(\gg 10^{6}\)).
- **Skipping** `linear_attn`: shock finite (\(\sim 1.8\times 10^{4}\)) and recovery much stronger at matched budgets.

**Qualitative error-propagation view.** Discrete errors in the recurrent/linear-attention path can compound across sequence state; keeping that path FP is structured mixed precision. The paper should treat `skip_linear_attn` as a **first-class method choice**, not a silent implementation detail.

Optional later ablations: FFN-only quant; quantize GDN with longer heal; role-wise FP on worst mats (layer map).

---

## 12. Optional adapters (not main pure-quaternary claim)

Config-only extras (defaults off for mainline):

| Knob | Forward effect |
|------|----------------|
| `pre_rms` | RMSNorm on activations before matmul |
| `weight_calib=unit_absmean` | **Avoid** (bundle FAIL) |
| LoRA \(r\) | \(y \mathrel{+}= (\alpha/r)\, x A^{\top} B^{\top}\), \(B=0\) init |

LoRA results are **hybrid quaternary + adapters**, not pure weight-only 2-bit. Report adapter param count when enabled.

---

## 13. Consistency checklist (paper ↔ code)

- [ ] \(c\), `scale_mode`, STE formulas match `quantize.py` / `RESEARCH.md`
- [ ] Loss matches `train.py` (CE shift, KL \(T^{2}\), commitment mean over \(\mathcal{Q}\))
- [ ] Skip policy matches `model.py`
- [ ] Mainline knobs match `heal_kl_trust_400m` preset in `config.py`
- [ ] Efficiency section does not claim measured kernel speedups without data
- [ ] BitNet appears as related motivation, not a silent competitor baseline

---

## 14. Pointers

| Doc | Role |
|-----|------|
| [`RESEARCH.md`](RESEARCH.md) | Implementation law |
| [`PAPER.md`](PAPER.md) | Paper outline, claims, tables |
| [`RESULTS.md`](RESULTS.md) | Numbers, gates, decision tree |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | Phases |
