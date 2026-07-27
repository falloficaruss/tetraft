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
| **`polish_kl_5m`** | B weights; lr 2e-5 const; +1280 | **+5.24M** | **≥34.38** | ❌ FAIL — stop polish |
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
6. **Keep \(c=0.25\)**, λw=**256**, α=0.5 / T=2 until a **gated** D2 ablation wins.
7. **Two-session resume** (full opt ckpt + `schedule_max_steps`) is production path for long KL.
8. **Parity still open** (target ≲1.3× / ~23 PPL).
9. **Polish FAIL ≠ length FAIL**; static α/T @ 5M **null** — next science is **§5.8** (D0→D1/D2).

---

## 4. What we did *not* settle

| Question | Status |
|----------|--------|
| Longer KL (100M+) | Open — **§5.8 D1** primary length lever |
| α–T static scouts @ 5.2M | **Null signal** — lock 0.5/2.0; stop burning 5M only on α/T (§5.4) |
| Layer-wise Q-error / sensitivity map | Open — **§5.8 D0** do first |
| FP CPT control (same tokens, no quant) | Recommended paper control |
| Codebook \(c\) / learnable \(c\) / role-wise FP | Open — §5.8 D2 after D0 |
| Downstream vs orig | Deferred until closer PPL |
| Longer λ / all-Linear + KL | Deferred (λ math note §5.8) |
| Bit-shift kernels | Systems track (parallel) |

---

## 5. Current recipe (frozen KL DNA)

| Knob | Value |
|------|--------|
| Best | **`heal_kl_50m` → ~34.38 @ 50M** (after/orig ~1.95) |
| α / T / β | **0.5** / **2.0** / **0.01** |
| λ_warmup | **256** (~**1.0M** tokens @ 4096 tok/step — not 16M) |
| LR | 2e-4, cosine → 0.1, horizon = run length |
| Scope | `skip_linear_attn=True`, c=0.25, ~41% quantized |
| Teacher | frozen original FP |

### 5.3 Science options next  ← **see §5.8 decision tree**

**A. Now (ordered)**  
1. **§5.8 D0** — layer-wise quant-error / sensitivity on KL-50M B (+ cold re-eval).  
2. **§5.8 D1** — fresh **`heal_kl_100m`** same DNA (unless D0 shows a few modules dominate).  
3. **§5.8 D2** — one-knob representation scouts @ 5.2M (gate &lt;49.31): \(c\), learn-\(c\), skip q/o, output scale.

**B. Paper hygiene (cheap, parallel)**  
1. **FP continued-pretrain control** — same FineWeb, ~25–50M tokens, **no quant**.  
2. Freeze tables: Original | Shock | CE-50M | KL-5M | KL-50M | (% quantized).

**C. Closed / stop**  
1. ~~§5.5 polish~~ — **FAIL**; no more small-LR continue on B.  
2. ~~§5.4 static α/T as main lever~~ — **null @ 5M**; keep α=0.5 T=2.0.

**D. Later / optional**  
α schedule (not only static α); longer λ; STE `clip`; β micro-sweep; GDN+KL; post-quant RMSNorm (after output scale); Muon §5.7; 2-bit pack.

**E. Deprioritize**  
More polish on B, CE-only longer, BitNet baselines, chat SFT, 2B, free-form “extra RMSNorm everywhere” without D0 — until gap ≲1.5×.

---

### 5.4 Option 1 — Fresh α / T scout @ ~5.2M  ⚪ **null — stop as main lever**

**Goal (historical):** Beat locked scout **`scout_kl_5m` ~49.31** by changing only distill knobs.  
**Result:** post-polish 5M distill tweak(s) brought **no significant improvement** over ~49.31. Treat **static α/T as locally saturated at 5.2M**.  
**Lock:** α=**0.5**, T=**2.0**. Do **not** spend further Kaggle units on static α/T-only scouts.  
Optional later (only if D1–D2 stall): **α schedule** (more teacher early), not another static grid.

**DNA (match scout, if replaying):** skip GDN, λw=**256**, linear→0, lr **2e-4**, AdamW8bit, 1280 steps (~5.24M), β=0.01, c=0.25.

| Run ID | α | T | Notes |
|--------|---|---|--------|
| `scout_kl_5m` (done) | 0.5 | 2.0 | **baseline gate 49.31** |
| `scout_kl_a03_t2` (etc.) | … | … | no clear win vs gate → abandoned as priority |

| Gate | Action (resolved) |
|------|-------------------|
| End PPL **&lt; 49.31** | Would lock new α/T → fresh long KL |
| All ≥ 49.31 / no significant Δ | **Keep 0.5/2.0** → §5.8 D0 then D1 |

---

### 5.5 Option 2 — Polish from heal_kl_50m B @ lr 2e-5 (~5M)  ❌ **FAIL**

**Result:** did **not** beat **~34.38**. Treat KL-50M B as **plateau under locked DNA** for small-LR continue.  
**Polish FAIL ≠ length FAIL:** session B still fell hard under a proper cosine (48.65→34.38). Next length test is **fresh `heal_kl_100m`**, not more polish.  
**Do not** run 10M polish or further P sessions on the same B weights unless DNA changes first via a **from-scratch** long run.

| Knob (for replay) | Value |
|------|--------|
| Init | `--resume` B `checkpoint-final` (weights OK) |
| `max_steps` | **13487** (= 12207 + 1280) |
| lr | **2e-5** constant (`polish_kl_5m`) |
| Gate | &lt; **34.38** → **missed** |

```bash
python run_smoke.py --preset polish_kl_5m \
  --resume /kaggle/input/.../checkpoint-final \
  --skip-shock --skip-orig \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_polish_kl_5m
```

---

### 5.6 How options relate (post–α/T null)

| | §5.4 α/T | §5.5 polish | §5.8 D0/D1/D2 | §5.7 Muon |
|--|----------|-------------|-----------------|-----------|
| Status | **null — stop** | **FAIL — stop** | **next** | design only |
| Init | Fresh | KL-50M B | B (analyze) / fresh train | Fresh |
| Focus | α, T | lr 2e-5 continue | error map → 100M → rep scouts | optimizer |

**Path now:** §5.8 D0 → D1 (`heal_kl_100m`) and/or D2 (one representation knob). Hygiene parallel. Muon only after recipe settled.

---

### 5.7 Future — Muon optimizer 5M ablation (**not implemented**)

**Status:** design note only. Code still **AdamW / AdamW8bit** only (`train._build_optimizer`). Do **not** claim Muon numbers until this lands.

**Motivation:** matrix-aware updates on 2D `Linear`/`QuantizedLinear` weights; optional lever if AdamW scouts plateau. **Not** a seamless resume from AdamW ckpts; **not** a polish-on-B substitute.

| Knob | Proposed value |
|------|----------------|
| Run ID | `scout_kl_muon_5m` (or `scout_ce_muon_5m` first if VRAM tight) |
| Init | **Fresh** pretrained |
| Tokens / steps | **1280** micro-steps (~5.24M), match `scout_kl_5m` |
| Quant DNA | skip GDN, λw=**256**, c=0.25, β=0.01 |
| Distill | **0.5 / 2.0** (locked) |
| LR schedule | linear→0 (or cosine+floor); **retune peak lr** (Muon ≠ AdamW 2e-4) |
| Optimizer | **Hybrid:** Muon on 2D `requires_grad` weights; **AdamW** on 1D (bias, norms), embeds if trained |
| Memory | Expect **higher** opt state than AdamW8bit; KL teacher still ~2× model — CE-only Muon scout OK to isolate optimizer |
| Gate | End PPL **&lt; 49.31** (KL) or **&lt; 60.6** (CE skip-GDN) |
| Promote? | Only if clearly beats matched AdamW scout → then consider long KL with Muon |

**Impl sketch (when building):** `config.optimizer ∈ {adamw8bit, adamw, muon}`; split param groups in `_build_optimizer`; vendor small Muon (Newton–Schulz) to avoid heavy Kaggle deps; one-step finite-loss test; preset `scout_kl_muon_5m`. Full AdamW↔Muon resume **unsupported**.

**Deprioritize vs:** §5.8 D0–D2, paper hygiene.

---

### 5.8 Decision tree — post null α/T & polish FAIL  ← **current**

**Context:** SOTA **~34.38** @ 50M (~1.95× orig). Polish-on-B FAIL. Static α/T @ 5.2M null. Still short of parity (≲1.3× / ~23 PPL).

**Working split of residual gap:**

| Bucket | Meaning | Evidence |
|--------|---------|----------|
| **A. Under-trained discrete student** | more proper tokens still help | KL-50M B half still fell hard; polish ≠ length |
| **B. Irreducible Q-error on sensitive mats** | codebook / role / scale | unknown until D0 |
| **C. Local opt of distill/STE** | α/T, STE, β | 5M α/T null; β `reg≈0` in logs |

Do **not** crown fixed \(c=0.25\) as *proven* #1 bottleneck without a gated scout. Do **not** treat external “λw=256 ≈ 16M tokens” claims — at **4096 tok/step**, λ ramp ≈ **1.0M tokens**.

#### D0 — Diagnose (cheap; do first)

**Tool:** `run_layer_map.py` (API: `per_module_quant_stats` / `aggregate_layer_map` in `quantize.py`).

On KL-50M B `checkpoint-final` / best (analysis job; little or no train):

1. **Cold re-eval** full val protocol — confirm ~34.38 (`--max-eval-batches`, not `--skip-ppl`).  
2. **Per-module** \(\mathrm{rel\_L2}\), mean \|W−Q\|/γ, bin mass → `layer_map.json` / `.csv`.  
3. Rank by role × depth: `q,k,v,o` / `gate,up,down` → `layer_map_summary.json`.  
4. Optional upper bound: `--fp-mask-topk N` sets λ=0 on worst modules (latent W, not orig pretrained) → PPL lift.  
5. Optional: `--compare-pretrained` weight map on fresh post-replace latents.

```bash
# Weight-only (CPU OK)
python run_layer_map.py \
  --checkpoint /path/to/checkpoint-final \
  --preset heal_kl_50m \
  --skip-ppl \
  --output-dir ./layer_map_out

# Full D0 on Kaggle (PPL + FP-mask)
python run_layer_map.py \
  --checkpoint /kaggle/input/.../checkpoint-final \
  --preset heal_kl_50m \
  --val-data ... \
  --max-eval-batches 20 \
  --fp-mask-topk 8 \
  --output-dir /kaggle/working/layer_map_b
```

Paste summary (role table + `suggest=`) into this file when the B run finishes. Act RMS vs teacher = later add-on (not in v1).

| D0 readout | Next |
|------------|------|
| Few modules dominate Q-error or FP-mask lift | **D2 `scout_skip_qo`** (or role-wise FP) before/parallel long run |
| Error flat; residual **scale** drift vs teacher | **D2 `scout_out_scale`** (identity init); RMSNorm only if scales fail |
| Error flat; scales OK | **D1 `heal_kl_100m`** primary |
| Unclear | D1 + one D2 scout if GPU-rich |

#### D1 — Length: fresh `heal_kl_100m` (same DNA)

| Knob | Value |
|------|--------|
| Init | **Fresh** pretrained (not polish B) |
| DNA | skip GDN, λw=**256**, α=**0.5**, T=**2**, β=0.01, c=0.25 |
| Schedule | cosine + min_lr_ratio=0.1; **horizon = full 100M** steps; 2-session resume OK |
| Gate | end PPL **&lt; 30** good; stretch &lt; 28 |
| Stop length-as-main | flat **~32–34** by ~70M → go D2 / D3 |

Polish FAIL does **not** block D1.

#### D2 — Representation scouts @ ~5.2M (one knob; gate **&lt; 49.31**)

Promote winner only via **fresh** long KL — do not only re-polish B.

| Run ID | Change | Hypothesis |
|--------|--------|------------|
| `scout_c050` | \(c=0.5\) | fixed spacing |
| `scout_c_learn` | learnable \(c\) (global or per-layer), init 0.25 | codebook bottleneck |
| `scout_skip_qo` | keep `q_proj` and/or `o_proj` FP | attention sensitivity |
| `scout_out_scale` | per-out-channel scale on `QuantizedLinear`, init **1** | scale mismatch (prefer before extra RMSNorm) |

**Extra RMSNorm:** Tier-A only after `out_scale` null **and** D0 shows scale drift. Prefer identity-init adapters on **quantized** mats only — not free-form norms on already-normalized blocks.

#### D3 — Optimization (only if D1–D2 stall)

- λw ∈ {1024, 2048} @ 5M then promote  
- α **schedule** (e.g. more teacher early), not static grid  
- STE `clip` @ 5M  
- β ∈ {0, 0.001, 0.01} — cheap, low expected Δ (`reg≈0`)

#### Explicitly do not

More polish-on-B · CE-only marathon · BitNet competitor runs · 2B · chat SFT · Muon before recipe settle · “RMSNorm everywhere” without D0 · more static α/T 5M cells

#### §5.9 Bundle scout R3+R4+R5 @ 5.2M  ← **optional next smoke**

**Goal:** Beat **`scout_kl_5m` ~49.31** with one combined adapter stack (attribution only if PASS).

| Piece | Config | Init |
|-------|--------|------|
| R3 pre-RMS | `pre_rms=True` | γ=**1** on each `QuantizedLinear` input |
| R4 calib | `weight_calib=unit_absmean` | one-shot at replace |
| R5 LoRA | `lora_rank=8`, `lora_alpha=8` | **B=0** (identity residual) |

| Knob | Value |
|------|--------|
| Preset | **`scout_kl_bundle_r345_5m`** |
| Base DNA | skip GDN, λw=256, linear→0, lr 2e-4, α=0.5 T=2 β=0.01, c=0.25 |
| Gate | end PPL **&lt; 49.31** (20-batch) |
| Claim note | hybrid quaternary **+ adapters** (report LoRA/pre_rms param count) |

```bash
python run_smoke.py --preset scout_kl_bundle_r345_5m \
  --train-data ... --val-data ... \
  --output-dir /kaggle/working/checkpoints_scout_kl_bundle_r345_5m
```

| Gate | Action |
|------|--------|
| **&lt; 49.31** | Leave-one-out @ 5M (`--no-pre-rms` / `--weight-calib none` / `--lora-rank 0`) before long KL |
| **≥ 49.31** | Bundle null @ 5M; try single-knob R5 or R3; do not promote bundle to 100M |

CLI overrides: `--pre-rms` / `--no-pre-rms`, `--weight-calib none|unit_absmean`, `--lora-rank`, `--lora-alpha`.

#### Decision flowchart

```text
Cold re-eval B + layerwise Q-error / scale snapshot   (§5.8 D0)
        │
        ├─ FP CPT control (parallel if GPU allows)
        │
        ▼
   D0 says few layers dominate? ──yes──► scout_skip_qo / role FP @ 5M
        │ no
        ▼
   Fresh heal_kl_100m (locked DNA)                  (§5.8 D1)
        │
        ├─ PPL < 30 and falling → continue length / Phase 3
        ├─ flat ~32–34 → D2 one-knob rep scout @ 5M
        └─ D0 scale drift early → D2 out_scale (then optional post-RMS)
```

#### External-review mapping (for the record)

| External claim | TetraFT stance |
|----------------|----------------|
| 50M too early | **Agree** as hypothesis → D1; polish FAIL ≠ length FAIL |
| Fixed \(c=0.25\) is #1 bug | **Hypothesis only** → D2 after D0; not proven |
| STE identity insufficient | Unproven → D3 `clip` later |
| Equal proj sensitivity | Plausible → D0 rank + `scout_skip_qo` |
| β=0.01 critical | **Weak** (`reg≈0`) → micro-sweep only |
| KL too weak; try 0.2/0.8 | Static α/T **null @ 5M**; optional α schedule later |
| λw=256 ≈ 16M tokens | **False** @ 4096 tok/step → ~**1.0M** tokens |
| Per-layer \|W−Q(W)\| | **Do first** (D0) |
| Extra RMSNorms | Only after out_scale + D0 scale evidence |

---

## 6. Success criteria

| Gate | Criterion | Status |
|------|-----------|--------|
| Eng | Multi-hour / multi-session resume | ✅ |
| CE scale | PPL **&lt; 48.2** at 50M | ✅ **~43.77** |
| KL scout | PPL **&lt; 60.6** at 5.24M | ✅ **~49.31** |
| KL 50M vs CE | PPL **&lt; 43.77** | ✅ **~34.38** |
| KL strong | PPL **≲ 35** | ✅ **~34.38** |
| Polish +5M @ 2e-5 | PPL **&lt; 34.38** | ❌ FAIL — stop polish |
| α/T scout | PPL **&lt; 49.31** | ⚪ null — lock 0.5/2.0 |
| D0 layer map | error/sensitivity logged on B | open (**next**) |
| KL 100M | PPL **&lt; 30** (stretch &lt; 28) | open |
| D2 rep scout | PPL **&lt; 49.31** @ 5.2M | open |
| Parity path | after/orig ≲ **1.3** (~23) | open (~1.95) |

---

## 7. How to cite

- Numbers + next science: **this file** (§5.8 current)  
- Phases: `RESEARCH_PLAN.md`  
- Math: `RESEARCH.md`  
- Kaggle: `KAGGLE.md`  

Update when a new controlled run finishes.
