# Running TetraFT on Google Colab

## 1. Prerequisites

- A [Google Colab](https://colab.research.google.com) account (free)
- A GitHub account with this repo pushed to `YOUR_USERNAME/tetraft`
- Google Drive for checkpoint persistence

## 2. Setup

1. Push this repo to GitHub under `YOUR_USERNAME/tetraft`
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. **File → Upload Notebook →** select `notebooks/qaft_demo.ipynb`
4. **Runtime → Change runtime type → T4 GPU**
5. In cell 1, replace `YOUR_USERNAME` with your GitHub username
6. **Runtime → Run all**

## 3. Run Order

| Cell | What It Does | Expected Time |
|---|---|---|
| 1 | Clone repo from GitHub, pip datasets | ~20s |
| 2 | Imports | ~5s |
| 3 | Configuration | instant |
| 3b | Mount Drive, set checkpoint path | ~10s (auth popup) |
| 4 | Load model (Qwen2.5-0.5B) | ~30–60s |
| 5 | Replace linear layers | ~5s |
| 6 | Prepare FineWeb dataset (10k docs) | ~2–3 min |
| 7 | Train (500 steps) | ~30–60 min |
| 8 | Evaluate final perplexity | ~1 min |
| 9 | Save checkpoint (auto to Drive) | ~5s |

**Total: ~35–65 min**

## 4. What Happens During Training

- 500 steps, batch_size=2, seq_len=512, grad_accum=4
- Effective batch size: 8
- Loss logged every 10 steps, eval every 100 steps
- Checkpoints saved to `MyDrive/tetraft_checkpoints/`
- VRAM: ~10 GB on T4 (15 GB available)

## 5. Resuming in a New Session

```python
# Re-run cells 1–5 (clone, imports, config, Drive, model, replace layers, data)
# then:
trainer.load_checkpoint('/content/drive/MyDrive/tetraft_checkpoints/checkpoint-best')
trainer.train(train_loader, eval_loader)
```

## 6. Colab Limits & Troubleshooting

| Limit | Value | Notes |
|---|---|---|
| Session max | ~12 hours | Training ~1h — fine |
| Weekly GPU | ~15–30 hrs | Dynamic quota |
| Idle timeout | ~90 min | Keep tab active |
| VRAM (T4) | ~15 GB usable | Fits 0.5B model easily |

| Symptom | Fix |
|---|---|
| `git clone` fails | Check `YOUR_USERNAME` in cell 1, or repo is private |
| `CUDA out of memory` | Lower `cfg.batch_size = 1` or `cfg.seq_length = 256` |
| Drive mount hangs | Approve the auth link in a new tab |
| HF model download fails | Colab has internet by default |
| Kernel died | Usually OOM — restart with smaller batch |
