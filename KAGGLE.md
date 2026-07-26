# TetraFT — Kaggle checklist

**Empirical baselines & next architecture:** [`RESULTS.md`](RESULTS.md) (source of truth for numbers).

## Datasets

| Kaggle Dataset | Contents |
|----------------|----------|
| **`tetraft-code`** | Flat root `.py` + `notebooks/` |
| **`tetraft-fineweb-edu-50m`** | `train.jsonl`, `val.jsonl`, `sample_meta.json` |

Zip code without `.venv` / `data/` / checkpoints. Refresh `tetraft-code` when modules change.

---

## A. FineWeb-Edu 50M sample ✅

CPU, Internet ON → `notebooks/build_fineweb_sample.ipynb` → Dataset `tetraft-fineweb-edu-50m`.

---

## B. Recorded runs (do not re-litigate)

| Run | ≈ tokens | Val PPL | Notes |
|-----|---------:|--------:|-------|
| Original | — | **~17.67** | FP baseline |
| Shock (all Linear) | 0 | ≫1e6 | λ=1, zero FT |
| Shock (skip GDN) | 0 | ~1.78e4 | milder |
| `short` | ~0.8M | ~472 | pipeline |
| `full_smoke` | 5.2M | ~79.4 | all Linear |
| `full_smoke` + skip GDN | 5.2M | ~60.6 | scope win; **5M CE gate** |
| `heal_25m` | 25.0M | ~48.2 | CE heal |
| **`heal_50m`** | **50.0M** | **~43.77** | **best CE; after/orig ~2.48** |
| `scale_25m` (partial) | ~21M | ~68.6 | old DNA; not default |

Stack: BF16 + AdamW8bit + grad ckpt, seq 512, batch 1, accum 8 → **4096 tok/step**.

If `KeyError: qwen3_5`:  
`%pip install -U "git+https://github.com/huggingface/transformers.git"`

**Sanity before train:** with skip GDN, inventory should be ~**96 eligible / 91 skipped**, ~**41%** quantized — not 186/1 and 66%.

---

## C. Disk lesson (implemented)

`scale_25m` filled Kaggle working disk via repeated **full model + optimizer** checkpoints.

**Defaults now (in code):**
- Weights-only `best` / `final` (`save_optimizer=False`)
- `save_steps=0` — no periodic `step_*`; if enabled, prune via `max_step_checkpoints`
- Loss/PPL logged to `metrics.jsonl` under `output_dir` (not full state)
- Resume: pass `--save-optimizer` only when you need full state

**Ops:** clear output dir before long jobs.

---

## D. What to run next — `scout_kl_5m`

**CE heal is done** (`heal_50m` ~43.77). Next: **KL + quant-reg** at matched 5M schedule.

**Scout recipe (do not lengthen λ here):**
- Same as `full_smoke_no_gdn`: skip GDN, λw=**256**, linear→0, lr 2e-4, 1280 steps
- `distill_alpha=0.5`, `distill_temperature=2.0`, `quant_reg_beta=0.01`
- Loads **frozen FP teacher** (2× model VRAM)
- **Gate:** end PPL **&lt; 60.6**

| Preset | ≈ tokens | Status |
|--------|---------:|--------|
| `full_smoke_no_gdn` | 5.2M | done ~60.6 (CE gate) |
| `heal_25m` / `heal_50m` | 25M / 50M | done ~48.2 / **~43.77** |
| **`scout_kl_5m`** | **5.2M** | **next** |
| `heal_kl_25m` | 25M | after scout pass |

```bash
python run_smoke.py --preset scout_kl_5m \
  --train-data /kaggle/input/.../train.jsonl \
  --val-data /kaggle/input/.../val.jsonl \
  --output-dir /kaggle/working/checkpoints_scout_kl_5m
```

Notebook: set `PRESET = "scout_kl_5m"`.

**Not recommended:** resume heal_25m weights-only as primary 50M; old `scale_25m`/`scale_50m`.

---

## Internet policy

| Step | Internet |
|------|----------|
| Build FineWeb / first model download | ON |
| Train after cache | OFF preferred |

---

## Phase status

| Phase | Status |
|-------|--------|
| 1 + 1b smoke | **COMPLETE** (5.2M → ~79 all-Linear) |
| 1c scale_25m | **PARTIAL** (obsolete DNA) |
| 2 scope | **COMPLETE** (skip GDN → ~60.6 @ 5.2M) |
| 2 length `heal_25m` / `heal_50m` | **COMPLETE** (~48.2 @ 25M; **~43.77 @ 50M**) |
| **Next** | **`scout_kl_5m`** (gate &lt;60.6) → `heal_kl_25m` if pass |
| 3 main 100–200M | after KL recipe freeze |
