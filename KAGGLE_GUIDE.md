# Running TetraFT on Kaggle

## 1. Prerequisites

- A [Kaggle](https://www.kaggle.com) account (free)
- 30 GPU hours/week quota (resets every Monday)

## 2. Create a Notebook

1. Go to **Kaggle → Code → New Notebook**
2. Set the accelerator:
   - **Settings → Accelerator → GPU T4 x2** (preferred — 32 GB VRAM, two T4s)
   - Fallback: **GPU P100** (16 GB VRAM, also works for the 0.5B model)
3. **Persistence**: leave **Internet** toggled ON (needed to clone repo and download model)

## 3. Clone the Repo

Add this cell at the top of the notebook:

```python
# Clone TetraFT repo
!git clone https://github.com/YOUR_USERNAME/tetraft.git
%cd tetraft

# Install the package
!pip install -e .

# Verify
import tetraft
print(tetraft.__version__)
```

## 4. Open the Demo Notebook

Inside Kaggle's notebook interface:
- **File → Import Notebook → URL**
- Paste the raw GitHub URL:
  `https://raw.githubusercontent.com/YOUR_USERNAME/tetraft/main/notebooks/qaft_demo.ipynb`

Or copy the cells manually from `notebooks/qaft_demo.ipynb`.

## 5. Run Order

| Cell | What It Does | Expected Time |
|---|---|---|
| 1 | Install TetraFT | ~30s |
| 2 | Imports | ~5s |
| 3 | Configuration | instant |
| 4 | Load model (Qwen2.5-0.5B) | ~30–60s (downloads from HF) |
| 5 | Replace linear layers | ~5s |
| 6 | Prepare C4 dataset | ~2–3 min (streams 10k docs) |
| 7 | Train | ~30–60 min |
| 8 | Evaluate perplexity | ~1 min |
| 9 | Save checkpoint | ~5s |

**Total runtime: ~35–65 min** — fits easily within Kaggle's 9-hour session limit
and 30-hour weekly quota.

## 6. What Happens During Training

- **~500 steps** at batch_size=2, seq_len=512, grad_accum=4
- Effective batch size: 2 × 4 = 8
- Logs loss every 10 steps, evaluates perplexity every 100 steps
- Saves checkpoints at steps 250 and 500 (final)
- VRAM usage: ~10 GB with T4 x2, ~8.5 GB with P100

## 7. Saving Results

Checkpoints land in `tetraft/checkpoints/` (inside `/kaggle/working`):

| File | Contents |
|---|---|
| `checkpoint-step_250` | Mid-training snapshot |
| `checkpoint-step_500` | Final snapshot |
| `checkpoint-best` | Best perplexity snapshot |
| `checkpoint-final` | Last step |

### To keep results after the session ends:

**Option A — Download manually**
```python
from IPython.display import FileLink
FileLink('tetraft/checkpoints/checkpoint-final')
```

**Option B — Save as Kaggle Dataset output**
```python
import shutil
shutil.make_archive('/kaggle/working/tetraft-checkpoints', 'zip',
                    'tetraft/checkpoints')
```
Then go to **Kaggle → Your notebook → Output → "Add to Dataset"**.

## 8. Resuming Training

To continue from a previous checkpoint in a new Kaggle session:

```python
from tetraft import QAFTConfig, QAFTTrainer

# Re-run setup (clone repo, install, load model, replace layers)
# then:
trainer.load_checkpoint('/kaggle/input/your-dataset/checkpoint-best')
trainer.train(train_loader, eval_loader)
```

## 9. GPU Limits & Pitfalls

| Limit | Value | Notes |
|---|---|---|
| Session max | 9 hours | Training finishes in ~1h — no issue |
| Weekly GPU | 30 hours | ~30 runs/week with 0.5B model |
| Idle timeout | ~20 min | Keep the browser tab active |
| VRAM (P100) | 16 GB | Fits 0.5B with batch_size=2, seq_len=512 |
| VRAM (T4 x2) | 32 GB | Comfortable headroom for larger batch sizes |

**If you run out of VRAM (CUDA OOM):**
```python
# Reduce memory pressure
cfg.batch_size = 1           # halves activation memory
cfg.seq_length = 256         # halves sequence memory
cfg.gradient_accumulation_steps = 8  # restore effective batch size
```

**If the session disconnects:** all cells stop. Re-run from the start in a new
session — checkpoints are **not** saved unless you explicitly output them as a
Kaggle Dataset (see section 7).

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: tetraft` | Run the install cell again |
| `CUDA out of memory` | Reduce batch_size or seq_length |
| Hanging on C4 dataset load | Add `split='train[:1000]'` to load fewer docs |
| `datasets` library error | `!pip install -U datasets` |
| HF model download fails | Retry — Kaggle → Settings → Internet ON |
| Kernel died | Usually OOM; reduce memory and restart |
