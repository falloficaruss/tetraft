#!/usr/bin/env python3
"""Paper figures for heal_kl_trust_400m from a run pack.

Usage::

    python scripts/plot_heal_kl_trust_400m.py /path/to/heal_kl_trust_400m
    python scripts/plot_heal_kl_trust_400m.py /path/to/pack --out /path/to/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _load_baselines(run_root: Path) -> Dict[str, Any]:
    p = run_root / "baselines.json"
    if p.is_file():
        with p.open() as f:
            return json.load(f)
    return {}


def plot_pack(run_root: Path, out_dir: Path) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib required for plotting: pip install matplotlib"
        ) from e

    from run_pack import merge_run_pack

    merge_run_pack(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = _load_baselines(run_root)
    written: List[Path] = []

    recovery = _read_csv(run_root / "curves" / "recovery_ppl.csv")
    if recovery:
        xs, ys = [], []
        for r in recovery:
            t, p = _f(r, "tokens_end"), _f(r, "ppl_end")
            if t is not None and p is not None:
                xs.append(t / 1e6)
                ys.append(p)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=2, ms=6, label="heal_kl_trust_400m")
        for key, style in (
            ("ppl_original", ("--", "C2", "original FP")),
            ("heal_kl_50m", (":", "C3", "heal_kl_50m")),
            ("scout_kl_trust_a03_5m", ("-.", "C4", "trust scout 5M")),
            ("parity_1_3x", ("--", "0.4", "1.3× parity")),
        ):
            if key in baselines:
                ax.axhline(
                    float(baselines[key]),
                    ls=style[0],
                    color=style[1],
                    lw=1.2,
                    label=style[2],
                )
        ax.set_xlabel("Tokens (M, repo convention)")
        ax.set_ylabel("Val PPL")
        ax.set_title("Recovery curve — soft trust + α=0.3")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        path = out_dir / "fig_recovery_ppl.pdf"
        fig.savefig(path, dpi=200)
        fig.savefig(out_dir / "fig_recovery_ppl.png", dpi=200)
        plt.close(fig)
        written.append(path)

        # after/orig
        xs, ys = [], []
        for r in recovery:
            t, a = _f(r, "tokens_end"), _f(r, "after_over_orig")
            if t is not None and a is not None:
                xs.append(t / 1e6)
                ys.append(a)
        if xs:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.plot(xs, ys, "s-", color="#d62728", lw=2, ms=6)
            ax.axhline(1.0, ls="--", color="0.3", label="parity")
            ax.axhline(1.3, ls=":", color="0.5", label="1.3×")
            ax.set_xlabel("Tokens (M, repo convention)")
            ax.set_ylabel("after / original PPL")
            ax.set_title("Gap to original")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = out_dir / "fig_after_over_orig.pdf"
            fig.savefig(path, dpi=200)
            fig.savefig(out_dir / "fig_after_over_orig.png", dpi=200)
            plt.close(fig)
            written.append(path)

    loss_rows = _read_csv(run_root / "curves" / "train_loss.csv")
    log_rows = [r for r in loss_rows if r.get("event") == "log"]
    if log_rows:
        def series(key: str) -> Tuple[List[float], List[float]]:
            xs, ys = [], []
            for r in log_rows:
                t, v = _f(r, "tokens"), _f(r, key)
                if t is not None and v is not None:
                    xs.append(t / 1e6)
                    ys.append(v)
            return xs, ys

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for key, lab in (("loss", "total"), ("ce", "CE"), ("kl", "KL")):
            xs, ys = series(key)
            if xs:
                ax.plot(xs, ys, lw=1.2, label=lab)
        ax.set_xlabel("Tokens (M)")
        ax.set_ylabel("Loss")
        ax.set_title("Train loss components")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "fig_loss_components.pdf"
        fig.savefig(path, dpi=200)
        fig.savefig(out_dir / "fig_loss_components.png", dpi=200)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        xs_l, ys_l = series("lambda")
        xs_r, ys_r = series("lr")
        if xs_l:
            ax.plot(xs_l, ys_l, color="C0", label="λ")
        ax2 = ax.twinx()
        if xs_r:
            ax2.plot(xs_r, ys_r, color="C1", label="lr")
        ax.set_xlabel("Tokens (M)")
        ax.set_ylabel("λ")
        ax2.set_ylabel("learning rate")
        ax.set_title("λ anneal and LR schedule")
        ax.grid(True, alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        fig.tight_layout()
        path = out_dir / "fig_lr_lambda.pdf"
        fig.savefig(path, dpi=200)
        fig.savefig(out_dir / "fig_lr_lambda.png", dpi=200)
        plt.close(fig)
        written.append(path)

    bins = _read_csv(run_root / "curves" / "quant_bins.csv")
    if bins:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        keys = [("frac_-1", "−1"), ("frac_-c", "−c"), ("frac_+c", "+c"), ("frac_+1", "+1")]
        for key, lab in keys:
            xs, ys = [], []
            for r in bins:
                t, v = _f(r, "tokens"), _f(r, key)
                if t is not None and v is not None:
                    xs.append(t / 1e6)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, lw=1.2, label=lab)
        ax.set_xlabel("Tokens (M)")
        ax.set_ylabel("Bin mass")
        ax.set_title("Quaternary codebook occupancy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        path = out_dir / "fig_bin_mass.pdf"
        fig.savefig(path, dpi=200)
        fig.savefig(out_dir / "fig_bin_mass.png", dpi=200)
        plt.close(fig)
        written.append(path)

    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot heal_kl_trust_400m paper figures")
    p.add_argument("run_root", type=str)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Figure output dir (default: <run_root>/figures)",
    )
    args = p.parse_args(argv)
    root = Path(args.run_root)
    out = Path(args.out) if args.out else root / "figures"
    paths = plot_pack(root, out)
    if not paths:
        print("No curves found — run at least one session or merge_run_pack first")
        return 1
    print("Wrote:")
    for path in paths:
        print(" ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
