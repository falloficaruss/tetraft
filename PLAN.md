# TetraFT — Resources, Data & Platform Plan

**Platform:** Kaggle (primary). Local/Colab optional for unit tests.  
**Math/method:** `RESEARCH.md`  
**Research roadmap:** `RESEARCH_PLAN.md`

---

## 1. Model ladder

| Stage | Checkpoint | Role |
|-------|------------|------|
| **Now** | `Qwen/Qwen3.5-0.8B-Base` | All method development + first recovery runs |
| Next | `Qwen/Qwen3.5-0.8B` (instruct) | Optional after base PPL recovery |
| Scale-up | `Qwen/Qwen3.5-2B-Base` | Main paper-scale model once 0.8B recipe is stable |

**Notes:**

- Qwen3.5 dense small models are **hybrid** (Gated DeltaNet + Gated Attention), not MoE.
- Prefer **Base** for conversion science; language path only; skip vision.
- One family first (Qwen3.5). Other architectures only after 0.8B→2B works.

---

## 2. Dataset plan (Kaggle)

### 2.1 Train: FineWeb-Edu **fixed sample** (custom Kaggle Dataset)

| Item | Decision |
|------|----------|
| Source | Hugging Face `HuggingFaceFW/fineweb-edu` |
| In this repo? | **No** — build a fixed sample |
| Full 1.3T? | **No** |
| Delivery | Upload a **custom Kaggle Dataset** after one-time sampling |

**Recommended sample sizes**

| Name | Tokens | Disk (approx.) | Use |
|------|--------|----------------|-----|
| smoke | 5–10M | tens of MB | pipeline |
| ablate | 50M | ~200–400 MB text | recipe search |
| **main 0.8B** | **100–200M** | **~0.7–1.5 GB** JSONL; less if parquet/gz | primary recovery |
| stretch | 500M–1B | multi-GB | only if large gap remains |

**200M tokens ≈ ~1 GB raw text** (often 0.5–1.5 GB as JSONL/parquet depending on compression).

**Protocol**

1. One-time notebook/script: stream HF FineWeb-Edu → write `train.jsonl` (+ optional `val.jsonl`).
2. Fix **seed**, document HF revision, max tokens, packing `seq_length`.
3. Publish as Kaggle Dataset (e.g. `tetraft-fineweb-edu-200m`).
4. All training notebooks attach that dataset; prefer **internet off** during train.

### 2.2 Validation (required)

- Held-out FineWeb-Edu docs **or** a fixed WikiText-style val set.
- **Never** mixed into train.
- Same val for: original, zero-FT quant, every TetraFT checkpoint.

### 2.3 Instruction data

- **Deferred** until PPL is near the original.
- Not used for Phase 0–1 method work.

### 2.4 What not to use as main heal data

- Alpaca-only / tiny SFT as the sole recovery set  
- Full raw CommonCrawl without filtering  
- Unversioned random Kaggle FineWeb dumps as the paper corpus (OK for smoke only)

---

## 3. QAFT VRAM model

Latent weights stay high precision during training. Forward simulates quaternary \(Q(W)\).

\[
M_{\mathrm{total}} \approx M_{\mathrm{weights}} + M_{\mathrm{grads}} + M_{\mathrm{optim}} + M_{\mathrm{acts}} + M_{\mathrm{overhead}}
\]

| Pool | FP32-heavy | Lean Kaggle stack |
|------|------------|-------------------|
| Weights | 4 B/param | **2 B/param (BF16)** |
| Grads | 4 B/param | 2–4 B/param |
| AdamW | 8 B/param | **~2 B/param (8-bit Adam)** |
| Acts | seq × batch × width | **gradient checkpointing** |

Rough parameter-only floor (ignore acts): \(P_{\mathrm{billions}} \times (4+4+8)\) GB for full FP32 Adam ≈ \(16P\) GB.

---

## 4. Concrete scenarios

### A. Qwen3.5-0.8B (current target)

Assume \(P \approx 0.8\,\mathrm{B}\).

| Stack | Rough total (w/ checkpointing, seq 512–1024, small microbatch) |
|-------|------------------------------------------------------------------|
| FP32 + FP32 Adam | often **≥ 16–20 GB** — tight / OOM on 16 GB |
| **BF16 + 8-bit Adam + grad ckpt** | **~10–14 GB** — target for Kaggle T4/P100-class |
| Microbatch | 1–2; accumulate for effective larger batch |

**Kaggle default recipe (0.8B):** BF16, 8-bit AdamW, gradient checkpointing, `seq_length=512` or `1024`, `batch_size=1–2`, grad accum as needed.

### B. Qwen3.5-2B (later)

Assume \(P \approx 2\,\mathrm{B}\).

| Stack | Rough |
|-------|--------|
| FP32 Adam full | **~32 GB+** acts — needs large GPU |
| BF16 + 8-bit Adam + ckpt | **~18–28 GB** — 24 GB possible if careful; 40 GB comfortable |

Do not scale to 2B until 0.8B recovery pipeline is stable.

---

## 5. Kaggle layout

```
Kaggle Dataset: tetraft-code     → flat .py modules (this repo)
Kaggle Dataset: fineweb-edu-Xm  → train (+ val) text sample
Kaggle Notebook                 → train / eval scripts
```

| Item | Policy |
|------|--------|
| Code layout | **Flat root** `.py` (Kaggle flattens packages) |
| Notebook | Optional glue only; **logic lives in modules** |
| Internet | On for one-time data build / model download; off for reproducible train if cached |
| Checkpoints | `/kaggle/working` → Dataset or drive export |

**Dependencies (typical):** `torch`, `transformers`, `datasets`, `accelerate`, `bitsandbytes`, `pytest` (local).

---

## 6. Token budgets (0.8B)

| Run | Tokens | Purpose |
|-----|--------|---------|
| Smoke | 1–5M | NaN/OOM/shock→movement |
| Ablations | ~50M each | \(c\), scale, \(\lambda\), scope |
| Main | 100–200M | Recovery vs original |
| Optional | 500M+ | Close residual gap |

---

## 7. Software stack

| Component | Choice |
|-----------|--------|
| Training | Raw PyTorch `QAFTTrainer` (no HF Trainer) |
| Models | Hugging Face `transformers` |
| Optimizer | AdamW; **bitsandbytes 8-bit Adam** on Kaggle |
| Precision | BF16 compute; quant math in FP32 |
| Logging | CSV / stdout first; W&B optional |
| Eval v1 | Held-out PPL vs original |
| Eval later | lm-eval-harness |

---

## 8. Efficiency claims (later)

- Disk: pack 2-bit indices + scales + FP leftovers (embed/lm_head).  
- Training still uses latent high-precision weights (state clearly in paper).  
- Custom quaternary kernels: **not required** for capability results.

---

## 9. First resource actions (with Phase 0)

1. Confirm Kaggle GPU quota / T4 or better.  
2. After quant core works: download `Qwen3.5-0.8B-Base` once, cache.  
3. Build **50M** FineWeb-Edu sample first (faster iterate); expand to **200M** for main runs.  
4. Keep val split fixed from day one of data builds.
