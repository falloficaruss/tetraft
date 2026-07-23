"""QAFT entrypoint: inventory → original PPL → shock PPL → train.

Designed for Kaggle notebooks / scripts. Logic stays in flat modules; this file
only wires config, data paths, and the evaluation / train sequence.

Example (Kaggle, after attaching code + FineWeb sample datasets)::

    # Phase 1 short smoke (~0.8M tokens)
    python run_smoke.py --preset short --train-data ... --val-data ...

    # Phase 1c scale-up (~25M tokens, cosine LR + floor)
    python run_smoke.py --preset scale_25m --train-data ... --val-data ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from config import SMOKE_PRESETS, QAFTConfig, apply_smoke_preset
from data import (
    build_packed_dataloader,
    default_kaggle_data_candidates,
    resolve_data_path,
)
from eval import evaluate_perplexity, model_device
from model import dump_linear_inventory, replace_from_config, set_quant_lambda
from train import QAFTTrainer

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="TetraFT QAFT run (smoke / scale) on 0.8B")
    p.add_argument(
        "--preset",
        type=str,
        default="short",
        choices=sorted(SMOKE_PRESETS.keys()),
        help="Run preset (CLI flags override preset values)",
    )
    p.add_argument("--model-name", type=str, default=None, help="Override QAFTConfig.model_name")
    p.add_argument("--train-data", type=str, default=None)
    p.add_argument("--val-data", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="QAFT micro-steps (tokens ≈ steps×batch×seq×accum)",
    )
    p.add_argument("--max-eval-batches", type=int, default=20)
    p.add_argument("--max-train-texts", type=int, default=None, help="Cap docs loaded for packing")
    p.add_argument("--max-val-texts", type=int, default=None)
    p.add_argument("--skip-train", action="store_true", help="Only inventory + original + shock PPL")
    p.add_argument("--skip-shock", action="store_true")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--no-8bit-adam", action="store_true")
    p.add_argument("--quant-warmup-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None, help="LR warmup micro-steps")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="Override lr_scheduler_type",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="Cosine floor as fraction of peak LR (e.g. 0.1)",
    )
    p.add_argument("--logging-steps", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device-map", type=str, default="auto", help="HF device_map (auto|cpu|cuda)")
    return p.parse_args(argv)


def _apply_cli_overrides(config: QAFTConfig, args) -> QAFTConfig:
    preset = getattr(args, "preset", None) or "short"
    apply_smoke_preset(config, preset)

    if args.model_name:
        config.model_name = args.model_name
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seq_length is not None:
        config.seq_length = args.seq_length
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.no_bf16:
        config.use_bf16 = False
    if args.no_8bit_adam:
        config.use_8bit_adam = False
    if args.quant_warmup_steps is not None:
        config.quant_warmup_steps = args.quant_warmup_steps
    if getattr(args, "warmup_steps", None) is not None:
        config.warmup_steps = args.warmup_steps
    if getattr(args, "learning_rate", None) is not None:
        config.learning_rate = args.learning_rate
    if getattr(args, "lr_scheduler", None) is not None:
        config.lr_scheduler_type = args.lr_scheduler
    if getattr(args, "min_lr_ratio", None) is not None:
        config.min_lr_ratio = args.min_lr_ratio
    if getattr(args, "logging_steps", None) is not None:
        config.logging_steps = args.logging_steps
    if getattr(args, "eval_steps", None) is not None:
        config.eval_steps = args.eval_steps
    if getattr(args, "save_steps", None) is not None:
        config.save_steps = args.save_steps
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


def _build_loaders(config: QAFTConfig, tokenizer, args):
    train_path = resolve_data_path(
        config.train_data_path or None,
        *default_kaggle_data_candidates("train"),
    )
    val_path = resolve_data_path(
        config.val_data_path or None,
        *default_kaggle_data_candidates("val"),
    )
    if val_path is None:
        raise FileNotFoundError(
            "Could not find val.jsonl. Pass --val-data or place data under ./data/ or /kaggle/input/."
        )
    logger.info("val data: %s", val_path)
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

    train_loader = None
    if train_path is not None and not args.skip_train:
        logger.info("train data: %s", train_path)
        train_loader = build_packed_dataloader(
            train_path,
            tokenizer,
            seq_length=config.seq_length,
            batch_size=config.batch_size,
            text_field=config.data_text_field,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            max_texts=args.max_train_texts,
        )
    elif not args.skip_train:
        logger.warning("No train.jsonl found; will skip QAFT smoke train")

    return train_loader, val_loader, train_path, val_path


def run_smoke(args=None) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ns = _parse_args(args) if not isinstance(args, argparse.Namespace) else args
    # Namespace from notebooks may omit newer fields — fill defaults.
    if not hasattr(ns, "preset") or ns.preset is None:
        ns.preset = "short"
    for field, default in (
        ("warmup_steps", None),
        ("learning_rate", None),
        ("lr_scheduler", None),
        ("min_lr_ratio", None),
        ("logging_steps", None),
        ("eval_steps", None),
        ("save_steps", None),
    ):
        if not hasattr(ns, field):
            setattr(ns, field, default)

    config = _apply_cli_overrides(QAFTConfig(), ns)
    torch.manual_seed(config.seed)

    tokens_budget = config.tokens_budget()
    logger.info(
        "Run preset=%s max_steps=%d tokens_budget≈%s (batch=%d seq=%d accum=%d) "
        "lr_sched=%s min_lr_ratio=%.3f",
        ns.preset,
        config.max_steps,
        f"{tokens_budget:,}",
        config.batch_size,
        config.seq_length,
        config.gradient_accumulation_steps,
        config.lr_scheduler_type,
        config.min_lr_ratio,
    )

    results: Dict[str, Any] = {
        "model_name": config.model_name,
        "preset": ns.preset,
        "tokens_per_step": config.tokens_per_step(),
        "tokens_budget": tokens_budget,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
    }

    model, tokenizer = _load_model_and_tokenizer(config, device_map=ns.device_map)

    # 1.2 Linear inventory (pre-replace)
    inventory = dump_linear_inventory(
        model,
        skip_lm_head=config.skip_lm_head,
        skip_embed_tokens=config.skip_embed_tokens,
        skip_vision=config.skip_vision,
        skip_mtp=config.skip_mtp,
    )
    results["inventory_summary"] = inventory["summary"]
    inv_path = Path(config.output_dir) / "linear_inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    with inv_path.open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    logger.info("Linear inventory written to %s", inv_path)
    logger.info("Inventory summary: %s", inventory["summary"])

    train_loader, val_loader, train_path, val_path = _build_loaders(config, tokenizer, ns)
    results["train_data"] = train_path
    results["val_data"] = val_path

    # 1.3 Original val PPL
    logger.info("Measuring original (FP) val PPL …")
    ppl_orig = evaluate_perplexity(
        model, val_loader, max_batches=ns.max_eval_batches, device=model_device(model)
    )
    results["ppl_original"] = ppl_orig
    logger.info("Original val PPL: %.4f", ppl_orig)

    # 1.4 Shock: hard quant λ=1, zero FT
    ppl_shock = None
    if not ns.skip_shock:
        logger.info("Applying replace_from_config (quaternary forward) …")
        replace_from_config(model, config, verbose=True)
        set_quant_lambda(model, 1.0)
        logger.info("Measuring zero-FT shock PPL (λ=1) …")
        ppl_shock = evaluate_perplexity(
            model, val_loader, max_batches=ns.max_eval_batches, device=model_device(model)
        )
        results["ppl_shock"] = ppl_shock
        logger.info("Shock val PPL: %.4f (ratio vs orig: %.3f)", ppl_shock, ppl_shock / max(ppl_orig, 1e-8))

    # 1.5 Short QAFT smoke
    if not ns.skip_train and train_loader is not None:
        if ns.skip_shock:
            # Ensure quantized if we skipped shock path
            from quantize import QuantizedLinear

            has_q = any(isinstance(m, QuantizedLinear) for m in model.modules())
            if not has_q:
                replace_from_config(model, config, verbose=True)

        # Ultra-short runs: denser logs
        if config.max_steps <= 50:
            config.logging_steps = max(1, config.max_steps // 5)
            config.eval_steps = max(1, config.max_steps // 2)
            config.save_steps = config.max_steps

        logger.info(
            "Starting QAFT: max_steps=%d, tokens_budget≈%s, seq=%d, batch=%d, "
            "accum=%d, λ_warmup=%d, lr_warmup=%d, lr_sched=%s, min_lr_ratio=%.3f, "
            "bf16=%s, 8bit_adam=%s",
            config.max_steps,
            f"{tokens_budget:,}",
            config.seq_length,
            config.batch_size,
            config.gradient_accumulation_steps,
            config.quant_warmup_steps,
            config.warmup_steps,
            config.lr_scheduler_type,
            config.min_lr_ratio,
            config.use_bf16,
            config.use_8bit_adam,
        )
        trainer = QAFTTrainer(model, tokenizer, config)
        trainer.train(train_loader, eval_dataloader=val_loader)

        set_quant_lambda(model, 1.0)
        ppl_after = evaluate_perplexity(
            model, val_loader, max_batches=ns.max_eval_batches, device=model_device(model)
        )
        tokens_seen = trainer.global_step * config.tokens_per_step()
        results["ppl_after_smoke"] = ppl_after
        results["steps_ran"] = trainer.global_step
        results["tokens_seen"] = tokens_seen
        results["train_metrics"] = {
            "steps": trainer.metrics.get("step", []),
            "loss": trainer.metrics.get("loss", []),
            "lambda": trainer.metrics.get("lambda", []),
            "perplexity": trainer.metrics.get("perplexity", []),
        }
        logger.info(
            "Post-smoke val PPL: %.4f (steps=%d, tokens_seen≈%s)",
            ppl_after,
            trainer.global_step,
            f"{tokens_seen:,}",
        )
        if ppl_shock is not None:
            logger.info(
                "Recovery delta (shock - after): %.4f (positive ⇒ improved)",
                ppl_shock - ppl_after,
            )
        if ppl_orig is not None and ppl_orig > 0:
            logger.info(
                "Gap to original: after/orig = %.3f (target → 1.0)",
                ppl_after / ppl_orig,
            )

        # Finite-loss check
        losses = trainer.metrics.get("loss", [])
        if losses:
            finite = all(torch.isfinite(torch.tensor(x)).item() for x in losses)
            results["loss_finite"] = finite
            if not finite:
                logger.error("Non-finite loss detected during smoke train")
        else:
            results["loss_finite"] = None
    else:
        logger.info("Skipping QAFT smoke train")

    out_path = Path(config.output_dir) / "smoke_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Smoke results written to %s", out_path)
    return results


def main(argv=None) -> None:
    run_smoke(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    main()
