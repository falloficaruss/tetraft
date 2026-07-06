# Running TetraFT on Kaggle

## 1. Prerequisites

- A [Kaggle](https://www.kaggle.com) account (free)
- 30 GPU hours/week quota (resets every Monday)

## 2. Upload the Code as a Kaggle Dataset

TetraFT is loaded directly from a Kaggle Dataset — no `pip install` required.

1. Clone/download this repo to your local machine
2. Remove `/.git/` (keep everything else)
3. Zip the folder: `tetraft.zip`
4. Go to **Kaggle → Datasets → New Dataset**
5. Upload `tetraft.zip`, name it `tetraft`

The notebook accesses it at `/kaggle/input/tetraft/`.

## 3. Create a Notebook

1. Go to **Kaggle → Code → New Notebook**
2. Set the accelerator:
   - **Settings → Accelerator → GPU T4 x2** (preferred — 32 GB VRAM)
   - Fallback: **GPU P100** (16 GB VRAM, also works for 0.5B)
3. **Internet**: leave ON (needed to download datasets and model from HuggingFace)
4. **Add Dataset**: click **+Add Input → Datasets → select "tetraft"**

## 4. Import the Demo Notebook

**File → Import Notebook → URL**

```
https://raw.githubusercontent.com/YOUR_USERNAME/tetraft/main/notebooks/qaft_demo.ipynb
```

Or create a new notebook and paste the cells manually.

## 5. Run Order

| Cell | What It Does | Expected Time |
|---|---|---|
| 1 | Load TetraFT (sys.path) + upgrade datasets | ~15s |
| 2 | Imports | ~5s |
| 3 | Configuration | instant |
| 4 | Load model (Qwen2.5-0.5B) | ~30–60s (downloads from HF) |
| 5 | Replace linear layers with quaternary | ~5s |
| 6 | Prepare C4 dataset (10k docs, streaming) | ~2–3 min |
| 7 | Train (500 steps) | ~30–60 min |
| 8 | Evaluate final perplexity | ~1 min |
| 9 | Save checkpoint | ~5s |

**Total: ~35–65 min** — well within Kaggle's 9-hour session limit.

## 6. What Happens During Training

- 500 steps, batch_size=2, seq_len=512, grad_accum=4
- Effective batch size: 8
- Loss logged every 10 steps, eval every 100 steps
- Checkpoints saved at steps 250 and 500
- VRAM: ~10 GB on T4 x2, ~8.5 GB on P100

## 7. Saving Checkpoints

Checkpoints land in `./checkpoints/` (inside `/kaggle/working`).

**To keep them after the session ends, zip and output as Kaggle Dataset:**

```python
import shutil
shutil.make_archive('/kaggle/working/tetraft-checkpoints', 'zip',
                    '.')
```

Then go to **Notebook → Output → "Add to Dataset"**.

## 8. Resuming in a New Session

Upload the checkpoint Dataset from step 7, then:

```python
from tetraft import QAFTConfig, QAFTTrainer

# In new session: re-run cells 1–6 (load model, replace layers, prepare data)
trainer.load_checkpoint('/kaggle/input/checkpoint-dataset/checkpoint-best')
trainer.train(train_loader, eval_loader)
```

## 9. GPU Limits

| Limit | Value | Notes |
|---|---|---|
| Session max | 9 hours | Training ~1h — fine |
| Weekly GPU | 30 hours | ~30 runs/week |
| Idle timeout | ~20 min | Keep tab active |
| VRAM (P100) | 16 GB | Fits 0.5B with batch=2, seq=512 |
| VRAM (T4 x2) | 32 GB | Comfortable headroom |

**OOM fix:** reduce batch_size to 1 or seq_length to 256.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'QAFTConfig' from 'tetraft'` | Verify "tetraft" Dataset is added to the notebook (+Add Input) |
| `CUDA out of memory` | Lower batch_size or seq_length |
| HF model download fails | Check Internet is ON in Settings |
| C4 dataset hangs | Force `split='train[:1000]'` for a lighter test |
| Kernel died without error | Usually OOM — reduce memory and restart |
