"""D0 layer map: per-QuantizedLinear Q-error on a heal checkpoint (RESULTS.md §5.8).

Weight-only works on CPU. Optional cold PPL + FP-mask (λ=0 on worst modules) need
val data and usually a GPU.

Example::

    python run_layer_map.py \\
      --checkpoint /path/to/checkpoint-final \\
      --preset heal_kl_50m \\
      --output-dir ./layer_map_out \\
      --skip-ppl

    python run_layer_map.py \\
      --checkpoint /kaggle/input/.../checkpoint-final \\
      --preset heal_kl_50m \\
      --val-data /kaggle/input/.../val.jsonl \\
      --max-eval-batches 20 \\
      --fp-mask-topk 8 \\
      --output-dir /kaggle/working/layer_map_b
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from config import SMOKE_PRESETS, QAFTConfig, apply_smoke_preset
from data import (
    build_packed_dataloader,
    default_kaggle_data_candidates,
    resolve_data_path,
)
from eval import evaluate_perplexity, model_device
from model import replace_from_config, set_quant_lambda
from quantize import (
    QuantizedLinear,
    aggregate_layer_map,
    per_module_quant_stats,
    suggest_d0_next,
)

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="TetraFT D0 layer-wise quant error map")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to TetraFT checkpoint (model_state_dict). Omit with --compare-pretrained only.",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="heal_kl_50m",
        choices=sorted(SMOKE_PRESETS.keys()),
        help="Replace DNA (must match how the ckpt was trained)",
    )
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./layer_map_out")
    p.add_argument("--val-data", type=str, default=None)
    p.add_argument("--max-eval-batches", type=int, default=20)
    p.add_argument("--max-val-texts", type=int, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Weight stats only (no val forward)",
    )
    p.add_argument(
        "--fp-mask-topk",
        type=int,
        default=0,
        help="Set λ=0 on top-k modules by --sort-key and re-eval PPL (latent-W upper bound)",
    )
    p.add_argument(
        "--sort-key",
        type=str,
        default="rel_l2",
        choices=["rel_l2", "mean_abs_err_over_gamma", "mse", "mae"],
    )
    p.add_argument("--top-k", type=int, default=16, help="Top modules in summary")
    p.add_argument(
        "--compare-pretrained",
        action="store_true",
        help="Also dump weight map on fresh pretrained after replace (shock latents)",
    )
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument(
        "--skip-linear-attn",
        action="store_true",
        default=None,
        help="Force skip GDN (default comes from preset)",
    )
    p.add_argument(
        "--no-skip-linear-attn",
        action="store_true",
        help="Force quantize GDN",
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def _build_config(args) -> QAFTConfig:
    config = QAFTConfig()
    apply_smoke_preset(config, args.preset)
    if args.model_name:
        config.model_name = args.model_name
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.val_data:
        config.val_data_path = args.val_data
    if args.seq_length is not None:
        config.seq_length = args.seq_length
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.no_bf16:
        config.use_bf16 = False
    if args.no_skip_linear_attn:
        config.skip_linear_attn = False
    elif args.skip_linear_attn is True:
        config.skip_linear_attn = True
    if args.seed is not None:
        config.seed = args.seed
    return config


def _load_model_and_tokenizer(config: QAFTConfig, device_map: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if config.use_bf16 and torch.cuda.is_available() else torch.float32
    logger.info("Loading model: %s (dtype=%s, device_map=%s)", config.model_name, dtype, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=device_map if device_map != "cpu" else None,
        trust_remote_code=True,
    )
    if device_map == "cpu":
        model = model.to("cpu")
    model.config.use_cache = False
    return model, tokenizer


def _load_weights(model: torch.nn.Module, checkpoint: str) -> Dict[str, Any]:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"--checkpoint not found: {checkpoint}")
    logger.info("Loading checkpoint weights: %s", path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"checkpoint missing model_state_dict: {path}")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s ...", len(missing), list(missing)[:8])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s ...", len(unexpected), list(unexpected)[:8])
    meta = {
        "step": ckpt.get("step"),
        "best_perplexity": ckpt.get("best_perplexity"),
        "weights_only": ckpt.get("weights_only"),
        "path": str(path),
    }
    return meta


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _named_quant_modules(model: torch.nn.Module) -> List[tuple]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, QuantizedLinear)]


def _apply_fp_mask(
    model: torch.nn.Module,
    top_names: Sequence[str],
) -> List[str]:
    """Set λ=0 on named QuantizedLinear modules (latent FP forward). Returns applied names."""
    want = set(top_names)
    applied: List[str] = []
    for name, mod in _named_quant_modules(model):
        if name in want:
            mod.lambda_ = 0.0
            applied.append(name)
        else:
            mod.lambda_ = 1.0
    return applied


def _run_weight_map(
    model: torch.nn.Module,
    *,
    sort_key: str,
    top_k: int,
    tag: str,
) -> Dict[str, Any]:
    rows = per_module_quant_stats(model)
    summary = aggregate_layer_map(rows, top_k=top_k, sort_key=sort_key)
    suggestion = suggest_d0_next(summary)
    return {
        "tag": tag,
        "modules": rows,
        "summary": summary,
        "suggestion": suggestion,
    }


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    if not args.checkpoint and not args.compare_pretrained:
        logger.error("Provide --checkpoint and/or --compare-pretrained")
        return 2

    config = _build_config(args)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(config, device_map=args.device_map)
    logger.info(
        "replace_from_config: c=%.3f scale=%s skip_linear_attn=%s",
        config.quaternary_c,
        config.scale_mode,
        config.skip_linear_attn,
    )
    replace_from_config(model, config, verbose=True)
    set_quant_lambda(model, 1.0)

    results: Dict[str, Any] = {
        "preset": args.preset,
        "model_name": config.model_name,
        "skip_linear_attn": bool(config.skip_linear_attn),
        "quaternary_c": float(config.quaternary_c),
        "scale_mode": config.scale_mode,
        "sort_key": args.sort_key,
    }

    pretrained_bundle: Optional[Dict[str, Any]] = None
    if args.compare_pretrained:
        logger.info("Weight map on fresh pretrained latents (post-replace, no heal) …")
        pretrained_bundle = _run_weight_map(
            model, sort_key=args.sort_key, top_k=args.top_k, tag="pretrained_shock"
        )
        results["pretrained_shock"] = {
            "summary": pretrained_bundle["summary"],
            "suggestion": pretrained_bundle["suggestion"],
        }
        _write_csv(out_dir / "layer_map_pretrained.csv", pretrained_bundle["modules"])
        with (out_dir / "layer_map_pretrained.json").open("w", encoding="utf-8") as f:
            json.dump(pretrained_bundle["modules"], f, indent=2)

    ckpt_meta = None
    student_bundle: Optional[Dict[str, Any]] = None
    if args.checkpoint:
        ckpt_meta = _load_weights(model, args.checkpoint)
        results["checkpoint"] = ckpt_meta
        set_quant_lambda(model, 1.0)
        logger.info("Weight map on checkpoint …")
        student_bundle = _run_weight_map(
            model, sort_key=args.sort_key, top_k=args.top_k, tag="checkpoint"
        )
        results["checkpoint_map"] = {
            "summary": student_bundle["summary"],
            "suggestion": student_bundle["suggestion"],
        }
        _write_csv(out_dir / "layer_map.csv", student_bundle["modules"])
        with (out_dir / "layer_map.json").open("w", encoding="utf-8") as f:
            json.dump(student_bundle["modules"], f, indent=2)

    primary = student_bundle or pretrained_bundle
    assert primary is not None

    ppl_base = None
    ppl_fp_mask = None
    fp_mask_names: List[str] = []
    need_ppl = (not args.skip_ppl) and (
        args.fp_mask_topk > 0 or args.checkpoint is not None
    )
    if need_ppl:
        val_path = resolve_data_path(
            config.val_data_path or None,
            *default_kaggle_data_candidates("val"),
        )
        if not val_path:
            logger.error("PPL requested but no val data; pass --val-data or --skip-ppl")
            return 2
        logger.info("Building val loader: %s", val_path)
        val_loader = build_packed_dataloader(
            val_path,
            tokenizer,
            seq_length=config.seq_length,
            batch_size=config.batch_size,
            text_field=config.data_text_field,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            max_texts=args.max_val_texts,
        )
        device = model_device(model)
        set_quant_lambda(model, 1.0)
        logger.info("Cold eval PPL (λ=1, max_batches=%d) …", args.max_eval_batches)
        ppl_base = evaluate_perplexity(
            model, val_loader, max_batches=args.max_eval_batches, device=device
        )
        results["ppl_student"] = ppl_base
        logger.info("Student val PPL: %.4f", ppl_base)

        if args.fp_mask_topk > 0 and student_bundle is not None:
            ranked = sorted(
                student_bundle["modules"],
                key=lambda r: float(r[args.sort_key]),
                reverse=True,
            )
            top_names = [r["name"] for r in ranked[: args.fp_mask_topk]]
            fp_mask_names = _apply_fp_mask(model, top_names)
            logger.info(
                "FP-mask λ=0 on %d modules (latent W, not original pretrained): %s",
                len(fp_mask_names),
                fp_mask_names[:8],
            )
            ppl_fp_mask = evaluate_perplexity(
                model, val_loader, max_batches=args.max_eval_batches, device=device
            )
            results["ppl_fp_mask"] = ppl_fp_mask
            results["fp_mask_names"] = fp_mask_names
            results["fp_mask_note"] = (
                "λ=0 uses post-heal latent W on masked modules — upper bound, "
                "not original pretrained FP"
            )
            logger.info(
                "FP-mask val PPL: %.4f (Δ vs student: %+.4f)",
                ppl_fp_mask,
                ppl_fp_mask - ppl_base,
            )
            set_quant_lambda(model, 1.0)

    suggestion = suggest_d0_next(
        primary["summary"],
        fp_mask_ppl=ppl_fp_mask,
        baseline_ppl=ppl_base,
    )
    results["suggestion"] = suggestion

    summary_out = {
        **results,
        "primary_summary": primary["summary"],
    }
    with (out_dir / "layer_map_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    # Human-readable readout
    sm = primary["summary"]
    logger.info("=== D0 layer map summary (%s) ===", primary["tag"])
    logger.info(
        "modules=%s params=%s median_rel_l2=%.6f",
        sm.get("n_modules"),
        sm.get("n_params"),
        sm.get("median_rel_l2") or -1.0,
    )
    by_role = sm.get("by_role") or {}
    for role, st in sorted(by_role.items(), key=lambda kv: -kv[1]["rel_l2_wmean"]):
        logger.info(
            "  role %-12s n=%2d rel_l2_wmean=%.6f mae/γ_wmean=%.6f",
            role,
            st["n_modules"],
            st["rel_l2_wmean"],
            st["mean_abs_err_over_gamma_wmean"],
        )
    logger.info("top modules by %s:", args.sort_key)
    for t in (sm.get("top_modules") or [])[:10]:
        logger.info(
            "  [block=%s role=%s] rel_l2=%.6f mae/γ=%.6f %s",
            t.get("block"),
            t.get("role"),
            t["rel_l2"],
            t["mean_abs_err_over_gamma"],
            t["name"],
        )
    if ppl_base is not None:
        logger.info("ppl_student=%.4f", ppl_base)
    if ppl_fp_mask is not None:
        logger.info("ppl_fp_mask=%.4f names=%d", ppl_fp_mask, len(fp_mask_names))
    logger.info("suggest=%s", suggestion.get("suggest"))
    for r in suggestion.get("reasons") or []:
        logger.info("  reason: %s", r)
    logger.info("Wrote %s", out_dir / "layer_map_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
