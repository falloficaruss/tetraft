"""Export a healed QAFT checkpoint to tetraft-packed-v1 (2-bit codes + residual).

Default path is RAM-safe: topology via ``from_config`` + pack from state_dict
(no full pretrained weight load + ckpt double occupancy).

Examples::

    python export_packed.py --strip-checkpoint checkpoint-final.zip \\
        /mnt/storage/tetraft-systems/ckpts/heal_kl_trust_400m_S04_weights.pt

    python export_packed.py \\
        --checkpoint /mnt/storage/tetraft-systems/ckpts/heal_kl_trust_400m_S04_weights.pt \\
        --preset heal_kl_trust_400m \\
        --output /mnt/storage/tetraft-systems/packed/heal_kl_trust_400m_S04_packed.pt
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="TetraFT packed quaternary export")
    p.add_argument(
        "--strip-checkpoint",
        nargs=2,
        metavar=("SRC", "DST"),
        default=None,
        help="Strip full trainer ckpt to weights-only and exit",
    )
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--preset", type=str, default="heal_kl_trust_400m")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--device-map", type=str, default="cpu")
    p.add_argument(
        "--parity-modules",
        type=int,
        default=3,
        help="Number of QuantizedLinear modules to parity-check before save",
    )
    p.add_argument(
        "--full-hf-load",
        action="store_true",
        help="Legacy path: load full pretrained weights then state_dict (high RAM)",
    )
    p.add_argument(
        "--from-state-dict",
        action="store_true",
        default=True,
        help="RAM-safe: from_config topology + pack from ckpt (default)",
    )
    return p.parse_args(argv)


def _torch_load_checkpoint(path: str, map_location: str = "cpu"):
    import torch

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    last_err: Optional[BaseException] = None
    load_kw = dict(map_location=map_location, weights_only=False)
    # mmap reduces peak when file is on disk (PyTorch 2+)
    try:
        return torch.load(str(p), mmap=True, **load_kw)
    except TypeError:
        pass
    except Exception as e:
        last_err = e
        logger.info("torch.load(mmap=True) failed (%s); retry without mmap", e)

    try:
        return torch.load(str(p), **load_kw)
    except Exception as e:
        last_err = e
        logger.info("direct torch.load failed (%s); trying zip extract", e)

    tmp_root = Path("/mnt/storage/tetraft-systems/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    extract_dir = tmp_root / "checkpoint-final"
    if extract_dir.exists():
        import shutil

        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(str(p)):
        with zipfile.ZipFile(str(p), "r") as zf:
            zf.extractall(str(extract_dir))
        candidates = [extract_dir, extract_dir / "checkpoint-final"]
        candidates.extend(sorted(extract_dir.rglob("*")))
        for c in candidates:
            try:
                return torch.load(str(c), **load_kw)
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"could not load checkpoint from zip {p}: {last_err}") from last_err
    raise RuntimeError(f"could not load checkpoint {p}: {last_err}") from last_err


def strip_checkpoint(src: str, dst: str) -> Dict[str, Any]:
    """Load full ckpt, drop optimizer/scheduler, write weights-only."""
    import torch

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint for strip: %s", src)
    ckpt = _torch_load_checkpoint(src, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint is not a dict: {type(ckpt)}")
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"checkpoint missing model_state_dict; keys={list(ckpt.keys())[:30]}"
        )

    if "optimizer_state_dict" in ckpt:
        del ckpt["optimizer_state_dict"]
    if "scheduler_state_dict" in ckpt:
        del ckpt["scheduler_state_dict"]
    gc.collect()

    out = {
        "step": ckpt.get("step"),
        "model_state_dict": ckpt["model_state_dict"],
        "best_perplexity": ckpt.get("best_perplexity"),
        "config": ckpt.get("config"),
        "weights_only": True,
        "max_steps": ckpt.get("max_steps"),
        "schedule_max_steps": ckpt.get("schedule_max_steps"),
    }
    del ckpt
    gc.collect()

    torch.save(out, str(dst_path))
    size_gb = dst_path.stat().st_size / (1024**3)
    logger.info("Wrote weights-only: %s (%.3f GiB)", dst_path, size_gb)

    cfg = out.get("config")

    def _g(obj, name, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    print("=== strip summary ===")
    print(f"step={out.get('step')}")
    print(f"best_perplexity={out.get('best_perplexity')}")
    print(f"quaternary_c={_g(cfg, 'quaternary_c')}")
    print(f"scale_mode={_g(cfg, 'scale_mode')}")
    print(f"ste_mode={_g(cfg, 'ste_mode')}")
    print(f"skip_linear_attn={_g(cfg, 'skip_linear_attn')}")
    print(f"model_name={_g(cfg, 'model_name')}")
    print(f"weights_only path={dst_path}")
    print(f"size_GiB={size_gb:.3f}")
    print(f"has_optimizer={False}")

    extract_dir = Path("/mnt/storage/tetraft-systems/tmp/checkpoint-final")
    if extract_dir.exists():
        import shutil

        shutil.rmtree(extract_dir, ignore_errors=True)
        logger.info("Removed extract tree %s", extract_dir)

    return {
        "path": str(dst_path),
        "step": out.get("step"),
        "best_perplexity": out.get("best_perplexity"),
        "size_GiB": size_gb,
    }


def _apply_ckpt_config_overrides(config, ckpt_cfg) -> None:
    if ckpt_cfg is None:
        return
    keys = (
        "quaternary_c",
        "scale_mode",
        "ste_mode",
        "trust_softness",
        "skip_linear_attn",
        "skip_lm_head",
        "skip_embed_tokens",
        "skip_vision",
        "skip_mtp",
        "model_name",
        "pre_rms",
        "lora_rank",
        "lora_alpha",
        "weight_calib",
    )
    for k in keys:
        if isinstance(ckpt_cfg, dict):
            if k in ckpt_cfg and ckpt_cfg[k] is not None:
                setattr(config, k, ckpt_cfg[k])
        elif hasattr(ckpt_cfg, k):
            v = getattr(ckpt_cfg, k)
            if v is not None:
                setattr(config, k, v)


def _build_topology(config):
    """Empty HF graph + QuantizedLinear replace (no pretrained weights)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from model import replace_from_config

    logger.info("Building topology from_config: %s", config.model_name)
    hf_cfg = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
    # Prefer empty weights; fall back if API differs
    try:
        model = AutoModelForCausalLM.from_config(
            hf_cfg,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)
    model = model.to("cpu")
    model.config.use_cache = False
    # Drop uninitialized param storage where possible (keep structure)
    replace_from_config(model, config, verbose=True)
    return model


def _naive_bf16_bytes_from_sd(sd: dict) -> int:
    n = 0
    for t in sd.values():
        if hasattr(t, "numel"):
            n += int(t.numel())
    return n * 2


def run_export_state_dict(args, config, ckpt_meta, sd) -> int:
    """RAM-safe export: topology + pack from state_dict."""
    from pack import (
        estimate_packed_footprint,
        export_packed_state_from_state_dict,
        save_packed,
    )
    from quantize import QuantizedLinear

    topology = _build_topology(config)
    n_q = sum(1 for m in topology.modules() if isinstance(m, QuantizedLinear))
    logger.info("Topology QuantizedLinear count: %d", n_q)

    # Free topology parameter storage we won't use (keep module types/names)
    # Actual weights come from sd only.
    packed = export_packed_state_from_state_dict(
        topology,
        sd,
        c=float(config.quaternary_c),
        scale_mode=str(config.scale_mode),
        model_name=str(config.model_name),
        extra_meta={
            "source_checkpoint": ckpt_meta["src"],
            "step": ckpt_meta["step"],
            "best_perplexity": ckpt_meta["best_perplexity"],
            "preset": args.preset,
            "export_mode": "state_dict",
        },
        parity_modules=int(args.parity_modules),
    )
    del topology
    gc.collect()

    fp = estimate_packed_footprint(packed)
    packed["meta"].update(fp)
    naive_bf16_bytes = _naive_bf16_bytes_from_sd(sd)
    del sd
    gc.collect()

    save_packed(args.output, packed)
    out_path = Path(args.output)
    file_bytes = out_path.stat().st_size
    step = ckpt_meta["step"]
    best_ppl = ckpt_meta["best_perplexity"]

    print("=== packed export ===")
    print(f"output={out_path}")
    print(f"export_mode=state_dict")
    print(f"n_quant_modules={fp['n_quant_modules']}")
    print(f"n_quant_params={fp['n_quant_params']:,}")
    print(f"n_residual_params={fp['n_residual_params']:,}")
    print(f"bits_idx={fp['bits_idx']:,}")
    print(f"bits_gamma={fp['bits_gamma']:,}")
    print(f"bytes_estimate={fp['bytes_estimate']:,}")
    print(f"file_bytes={file_bytes:,} ({file_bytes / (1024**3):.3f} GiB)")
    print(f"naive_bf16_full_bytes={naive_bf16_bytes:,} ({naive_bf16_bytes / (1024**3):.3f} GiB)")
    print(f"step={step} best_perplexity={best_ppl}")
    return 0


def run_export_full_hf(args, config, ckpt_meta, sd) -> int:
    """Legacy high-RAM path."""
    import torch
    from transformers import AutoModelForCausalLM

    from model import replace_from_config, set_quant_lambda
    from pack import estimate_packed_footprint, export_packed_state, save_packed
    from quantize import QuantizedLinear, compute_scale, quaternary_quant
    from pack import pack_weight_matrix, unpack_weight_matrix

    dtype = torch.float32
    device_map = args.device_map
    logger.info(
        "Loading HF model %s dtype=%s device_map=%s",
        config.model_name,
        dtype,
        device_map,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=device_map if device_map != "cpu" else None,
        trust_remote_code=True,
    )
    if device_map == "cpu":
        model = model.to("cpu")
    model.config.use_cache = False
    replace_from_config(model, config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info("load_state_dict missing=%d unexpected=%d", len(missing), len(unexpected))
    del sd
    gc.collect()
    set_quant_lambda(model, 1.0)

    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, QuantizedLinear)]
    for name, mod in mods[: int(args.parity_modules)]:
        W = mod.weight.data.float().cpu()
        blob = pack_weight_matrix(W, float(mod.c), str(mod.scale_mode))
        got = unpack_weight_matrix(blob)
        ref = quaternary_quant(W, float(mod.c), compute_scale(W, str(mod.scale_mode))).float()
        if not torch.equal(got, ref):
            raise AssertionError(f"parity fail on {name}")
        logger.info("parity OK: %s", name)

    packed = export_packed_state(
        model,
        c=float(config.quaternary_c),
        scale_mode=str(config.scale_mode),
        model_name=str(config.model_name),
        extra_meta={
            "source_checkpoint": ckpt_meta["src"],
            "step": ckpt_meta["step"],
            "best_perplexity": ckpt_meta["best_perplexity"],
            "preset": args.preset,
            "export_mode": "full_hf",
        },
    )
    fp = estimate_packed_footprint(packed)
    packed["meta"].update(fp)
    full_numel = sum(p.numel() for p in model.parameters())
    naive_bf16_bytes = full_numel * 2
    save_packed(args.output, packed)
    out_path = Path(args.output)
    file_bytes = out_path.stat().st_size
    print("=== packed export ===")
    print(f"output={out_path}")
    print(f"export_mode=full_hf")
    print(f"n_quant_modules={fp['n_quant_modules']}")
    print(f"n_quant_params={fp['n_quant_params']:,}")
    print(f"n_residual_params={fp['n_residual_params']:,}")
    print(f"bits_idx={fp['bits_idx']:,}")
    print(f"bits_gamma={fp['bits_gamma']:,}")
    print(f"bytes_estimate={fp['bytes_estimate']:,}")
    print(f"file_bytes={file_bytes:,} ({file_bytes / (1024**3):.3f} GiB)")
    print(f"naive_bf16_full_bytes={naive_bf16_bytes:,} ({naive_bf16_bytes / (1024**3):.3f} GiB)")
    print(f"step={ckpt_meta['step']} best_perplexity={ckpt_meta['best_perplexity']}")
    return 0


def run_export(args) -> int:
    from config import SMOKE_PRESETS, QAFTConfig, apply_smoke_preset

    if args.preset not in SMOKE_PRESETS:
        raise SystemExit(
            f"unknown preset {args.preset!r}; choices={sorted(SMOKE_PRESETS)}"
        )
    if not args.checkpoint:
        raise SystemExit("--checkpoint required for export")
    if not args.output:
        raise SystemExit("--output required for export")

    config = QAFTConfig()
    apply_smoke_preset(config, args.preset)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = _torch_load_checkpoint(args.checkpoint, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"missing model_state_dict; keys={list(ckpt.keys())[:30]}")

    _apply_ckpt_config_overrides(config, ckpt.get("config"))
    if args.model_name:
        config.model_name = args.model_name

    logger.info(
        "DNA: model=%s c=%s scale=%s ste=%s skip_linear_attn=%s",
        config.model_name,
        config.quaternary_c,
        config.scale_mode,
        config.ste_mode,
        getattr(config, "skip_linear_attn", None),
    )

    sd = ckpt["model_state_dict"]
    ckpt_meta = {
        "src": str(args.checkpoint),
        "step": ckpt.get("step"),
        "best_perplexity": ckpt.get("best_perplexity"),
    }
    del ckpt
    gc.collect()

    if args.full_hf_load:
        return run_export_full_hf(args, config, ckpt_meta, sd)
    return run_export_state_dict(args, config, ckpt_meta, sd)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    if args.strip_checkpoint:
        src, dst = args.strip_checkpoint
        strip_checkpoint(src, dst)
        return 0
    return run_export(args)


if __name__ == "__main__":
    sys.exit(main())
