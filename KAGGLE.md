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
| Shock | 0 | ≫1e6 | λ=1, zero FT |
| `short` | ~0.8M | ~472 | pipeline |
| **`full_smoke`** | **5.2M** | **~79.4** | **best sample efficiency** (λw=256, linear→0) |
| `scale_25m` (partial) | ~21M | **~68.6** | λw=512 + cosine; **disk stop**; worse early efficiency |

Stack: BF16 + AdamW8bit + grad ckpt, seq 512, batch 1, accum 8 → **4096 tok/step**.

If `KeyError: qwen3_5`:  
`%pip install -U "git+https://github.com/huggingface/transformers.git"`

---

## C. Disk lesson (implemented)

`scale_25m` filled Kaggle working disk via repeated **full model + optimizer** checkpoints.

**Defaults now (in code):**
- Weights-only `best` / `final` (`save_optimizer=False`)
- `save_steps=0` — no periodic `step_*`; if enabled, prune via `max_step_checkpoints`
- Loss/PPL logged to `metrics.jsonl` under `output_dir` (not full state)
- Resume: pass `--save-optimizer` only when you need full state

**Ops:** clear `/kaggle/working/checkpoints` before long jobs.

---

## D. What to run next (architecture — see RESULTS.md §5)

**Not recommended as default:** re-run `scale_25m` / `scale_50m` with the same knobs.

**Recommended direction:**
1. **Control DNA** = full_smoke (λ_warmup≈**256**, peak lr 2e-4, same quant defaults)
2. **Phase 2 scouts** @ ~5–10M, one factor at a time:
   - module scope: control vs `--skip-linear-attn` (exclude GDN path `linear_attn`)
   - \(c\) ∈ {0.25, 0.5}
   - scale_mode
3. **Long-run LR** as a separate decision: mid-run LR + late anneal/floor — don’t assume full-horizon linear→0 scales
4. Optional later: full_smoke knobs × 25M as a pure length test (not bundled with λw=512)

```bash
# Example once disk-safe + presets exist — control-style smoke still valid:
python run_smoke.py --preset full_smoke \
  --train-data /kaggle/input/.../train.jsonl \
  --val-data /kaggle/input/.../val.jsonl \
  --output-dir /kaggle/working/checkpoints
```

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
| 1 + 1b smoke | **COMPLETE** (5.2M → ~79) |
| 1c scale_25m | **PARTIAL** (~21M → ~69); recorded in RESULTS.md |
| **Next** | Eng (disk) + **Phase 2** recipe/schedule architecture |
| 3 main 100–200M | after recipe freeze |
