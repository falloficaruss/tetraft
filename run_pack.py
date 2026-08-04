"""Paper-grade run pack: session summaries, ledger, curves for long QAFT marathons.

Layout (under run root, e.g. /kaggle/working/heal_kl_trust_400m)::

    RUN_META.json
    baselines.json
    ledger.jsonl
    LATEST.json
    sessions/S01/{session_summary,metrics,smoke_results,checkpoint-final,...}
    curves/*.csv

Only the **latest** session keeps ``checkpoint-final`` (pack size bounded);
older session checkpoints are pruned after each pack write. All small paper
artifacts (summaries, metrics, smoke_results, ledger, curves) are retained.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

RUN_ID_TRUST_400M = "heal_kl_trust_400m"
SESSION_MICRO_STEPS = 6104  # ≈ 25.00M tok @ 4096 tok/step
N_SESSIONS_TRUST_400M = 16
HORIZON_TRUST_400M = SESSION_MICRO_STEPS * N_SESSIONS_TRUST_400M  # 97664

# Session-end gates (post-session val PPL). Soft stretch at S16.
SESSION_GATES_TRUST_400M: Dict[int, float] = {
    1: 48.65,  # beat heal_kl_50m A
    2: 34.38,  # beat heal_kl_50m A+B
    4: 30.0,
    16: 23.0,  # stretch parity path ~1.3× orig
}

FROZEN_ORIG_PPL = 17.67

DEFAULT_BASELINES: Dict[str, Any] = {
    "ppl_original": FROZEN_ORIG_PPL,
    "ppl_shock_skip_gdn": 17800.0,
    "scout_kl_5m": 49.31,
    "scout_kl_r5_5m": 48.38,
    "scout_kl_trust_a03_5m": 43.34,
    "heal_25m_ce": 48.2,
    "heal_50m_ce": 43.77,
    "heal_kl_50m": 34.38,
    "parity_1_3x": round(FROZEN_ORIG_PPL * 1.3, 2),
}


def session_tag(session: int) -> str:
    return f"S{int(session):02d}"


def session_stop_step(session: int, session_steps: int = SESSION_MICRO_STEPS) -> int:
    return int(session) * int(session_steps)


def session_gate_ppl(session: int, gates: Optional[Dict[int, float]] = None) -> Optional[float]:
    g = gates if gates is not None else SESSION_GATES_TRUST_400M
    if int(session) in g:
        return float(g[int(session)])
    # nearest lower defined gate for go/no-go messaging
    keys = sorted(k for k in g if k <= int(session))
    return float(g[keys[-1]]) if keys else None


def ensure_run_root(run_root: Union[str, Path], *, run_id: str = RUN_ID_TRUST_400M) -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "curves").mkdir(parents=True, exist_ok=True)
    meta_path = root / "RUN_META.json"
    if not meta_path.is_file():
        write_json(
            meta_path,
            {
                "run_id": run_id,
                "session_micro_steps": SESSION_MICRO_STEPS,
                "n_sessions": N_SESSIONS_TRUST_400M,
                "horizon_micro_steps": HORIZON_TRUST_400M,
                "tokens_per_micro_step": 4096,
                "tokens_horizon_approx": HORIZON_TRUST_400M * 4096,
                "dna": {
                    "ste_mode": "trust",
                    "trust_softness": 1.0,
                    "distill_alpha": 0.3,
                    "distill_temperature": 2.0,
                    "quant_reg_beta": 0.01,
                    "skip_linear_attn": True,
                    "quaternary_c": 0.25,
                    "scale_mode": "absmean_channel",
                    "learning_rate": 2e-4,
                    "lr_scheduler_type": "cosine",
                    "min_lr_ratio": 0.1,
                    "quant_warmup_steps": 256,
                    "lora_rank": 0,
                },
                "gates": {str(k): v for k, v in SESSION_GATES_TRUST_400M.items()},
                "eval_protocol": {
                    "metric": "post_session_val_ppl",
                    "max_eval_batches": 20,
                    "lambda": 1.0,
                    "orig_ppl_ref": FROZEN_ORIG_PPL,
                },
            },
        )
    base_path = root / "baselines.json"
    if not base_path.is_file():
        write_json(base_path, DEFAULT_BASELINES)
    return root


def write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


def read_json(path: Union[str, Path]) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Union[str, Path], row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_prior_run_root(
    roots: Optional[Sequence[Union[str, Path]]] = None,
    *,
    run_id: str = RUN_ID_TRUST_400M,
) -> Optional[Path]:
    """Locate an attached prior run pack (Kaggle input or working)."""
    candidates: List[Path] = []
    if roots:
        candidates.extend(Path(r) for r in roots)
    else:
        for base in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
            if base.is_dir():
                candidates.append(base)
                try:
                    for sub in sorted(base.iterdir()):
                        if sub.is_dir():
                            candidates.append(sub)
                except OSError:
                    pass

    # Prefer explicit LATEST.json under run_id folder
    for c in candidates:
        for name in (run_id, f"tetraft-{run_id.replace('_', '-')}"):
            hit = c / name / "LATEST.json"
            if hit.is_file():
                return hit.parent
        if (c / "LATEST.json").is_file() and (c / "sessions").is_dir():
            return c
        # nested one level
        try:
            for sub in c.rglob("LATEST.json"):
                if sub.parent.name == run_id or (sub.parent / "sessions").is_dir():
                    return sub.parent
        except OSError:
            continue
    return None


def find_resume_checkpoint(run_root: Union[str, Path], session: int) -> Optional[Path]:
    """Checkpoint from session-1 for resuming session."""
    root = Path(run_root)
    if session <= 1:
        return None
    latest = root / "LATEST.json"
    if latest.is_file():
        meta = read_json(latest)
        p = meta.get("checkpoint_final") or meta.get("checkpoint")
        if p and Path(p).is_file():
            return Path(p)
        rel = meta.get("checkpoint_final_rel")
        if rel and (root / rel).is_file():
            return root / rel
    prev = root / "sessions" / session_tag(session - 1) / "checkpoint-final"
    if prev.is_file():
        return prev
    # any checkpoint-final under sessions
    hits = sorted(root.glob("sessions/S*/checkpoint-final"))
    return hits[-1] if hits else None


def dna_from_config(config) -> Dict[str, Any]:
    return {
        "ste_mode": getattr(config, "ste_mode", "identity"),
        "trust_softness": float(getattr(config, "trust_softness", 1.0)),
        "distill_alpha": float(getattr(config, "distill_alpha", 1.0)),
        "distill_temperature": float(getattr(config, "distill_temperature", 2.0)),
        "quant_reg_beta": float(getattr(config, "quant_reg_beta", 0.0)),
        "skip_linear_attn": bool(getattr(config, "skip_linear_attn", False)),
        "quaternary_c": float(getattr(config, "quaternary_c", 0.25)),
        "scale_mode": str(getattr(config, "scale_mode", "absmean_channel")),
        "learning_rate": float(getattr(config, "learning_rate", 0.0)),
        "lr_scheduler_type": str(getattr(config, "lr_scheduler_type", "linear")),
        "min_lr_ratio": float(getattr(config, "min_lr_ratio", 0.0)),
        "quant_warmup_steps": int(getattr(config, "quant_warmup_steps", 0)),
        "lora_rank": int(getattr(config, "lora_rank", 0) or 0),
        "max_steps": int(getattr(config, "max_steps", 0)),
        "schedule_max_steps": int(
            config.schedule_horizon_steps()
            if hasattr(config, "schedule_horizon_steps")
            else getattr(config, "max_steps", 0)
        ),
    }


def build_session_summary(
    *,
    run_id: str,
    session: int,
    step_start: int,
    step_end: int,
    tokens_per_step: int,
    ppl_end: Optional[float],
    ppl_best_in_session: Optional[float],
    ppl_original: Optional[float],
    ppl_shock: Optional[float],
    orig_ref: float = FROZEN_ORIG_PPL,
    dna: Optional[Dict[str, Any]] = None,
    resumed_from: Optional[str] = None,
    inventory_summary: Optional[Dict[str, Any]] = None,
    gates: Optional[Dict[int, float]] = None,
    max_eval_batches: int = 20,
) -> Dict[str, Any]:
    tokens_end = int(step_end) * int(tokens_per_step)
    tokens_start = int(step_start) * int(tokens_per_step)
    ref = float(ppl_original) if ppl_original is not None else float(orig_ref)
    after_over = (float(ppl_end) / ref) if ppl_end is not None and ref > 0 else None
    gate = session_gate_ppl(session, gates)
    gate_status = None
    if ppl_end is not None and gate is not None:
        gate_status = "PASS" if float(ppl_end) < float(gate) else "FAIL"
    return {
        "run_id": run_id,
        "session": int(session),
        "session_tag": session_tag(session),
        "step_start": int(step_start),
        "step_end": int(step_end),
        "tokens_start": tokens_start,
        "tokens_end": tokens_end,
        "tokens_per_step": int(tokens_per_step),
        "ppl_end": ppl_end,
        "ppl_best_in_session": ppl_best_in_session,
        "ppl_original": ppl_original,
        "ppl_shock": ppl_shock,
        "ppl_original_ref": ref,
        "after_over_orig": after_over,
        "gate_ppl": gate,
        "gate_status": gate_status,
        "resumed_from": resumed_from,
        "dna": dna or {},
        "inventory_summary": inventory_summary,
        "eval_protocol": {
            "metric": "post_session_val_ppl",
            "max_eval_batches": int(max_eval_batches),
            "lambda": 1.0,
        },
    }


def _prune_old_session_checkpoints(run_root: Union[str, Path], *, keep_tag: str) -> None:
    """Remove ``sessions/S*/checkpoint-final`` for every session except keep_tag.

    Keeps the run pack bounded at a single full checkpoint (the latest). Small
    session artifacts (summary, metrics, smoke_results) are left untouched.
    """
    root = Path(run_root)
    for ckpt in sorted((root / "sessions").glob("S*/checkpoint-final")):
        if ckpt.parent.name == keep_tag:
            continue
        try:
            ckpt.unlink()
            logger.info("Pruned stale session checkpoint: %s", ckpt)
        except OSError as e:
            logger.warning("Failed to prune %s: %s", ckpt, e)


def write_session_pack(
    run_root: Union[str, Path],
    summary: Dict[str, Any],
    *,
    smoke_results: Optional[Dict[str, Any]] = None,
    metrics_src: Optional[Union[str, Path]] = None,
    checkpoint_final_src: Optional[Union[str, Path]] = None,
    inventory_src: Optional[Union[str, Path]] = None,
    copy_checkpoint: bool = True,
    keep_only_latest_checkpoint: bool = True,
    delete_checkpoint_source: bool = False,
) -> Path:
    """Write sessions/Sxx artifacts, append ledger, update LATEST.json.

    ``keep_only_latest_checkpoint`` keeps the pack bounded: after a new
    ``checkpoint-final`` lands in the pack, every older ``sessions/S*/checkpoint-final``
    is removed. ``delete_checkpoint_source`` additionally removes the training-output
    source checkpoint once it has been copied (both only run when a new checkpoint
    was actually written into the pack, so LATEST.json never dangles).
    """
    root = ensure_run_root(run_root, run_id=str(summary.get("run_id") or RUN_ID_TRUST_400M))
    tag = summary.get("session_tag") or session_tag(int(summary["session"]))
    sess_dir = root / "sessions" / tag
    sess_dir.mkdir(parents=True, exist_ok=True)

    write_json(sess_dir / "session_summary.json", summary)
    if smoke_results is not None:
        write_json(sess_dir / "smoke_results.json", smoke_results)

    if metrics_src is not None and Path(metrics_src).is_file():
        shutil.copy2(metrics_src, sess_dir / "metrics.jsonl")

    if inventory_src is not None and Path(inventory_src).is_file():
        shutil.copy2(inventory_src, sess_dir / "linear_inventory.json")

    ckpt_rel = None
    ckpt_abs = None
    copied_new_ckpt = False
    dst = None
    if checkpoint_final_src is not None and Path(checkpoint_final_src).is_file():
        src = Path(checkpoint_final_src)
        dst = sess_dir / "checkpoint-final"
        if copy_checkpoint:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            ckpt_abs = str(dst)
            ckpt_rel = str(dst.relative_to(root))
            copied_new_ckpt = True
        else:
            ckpt_abs = str(src)
            ckpt_rel = str(src)

    if copied_new_ckpt:
        if keep_only_latest_checkpoint:
            _prune_old_session_checkpoints(root, keep_tag=tag)
        if delete_checkpoint_source and checkpoint_final_src is not None:
            src = Path(checkpoint_final_src)
            if dst is not None and src.resolve() != dst.resolve():
                try:
                    src.unlink()
                    logger.info(
                        "Removed training-output checkpoint after pack copy: %s", src
                    )
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", src, e)

    ledger_row = {
        "run_id": summary.get("run_id"),
        "session": summary.get("session"),
        "session_tag": tag,
        "step_end": summary.get("step_end"),
        "tokens_end": summary.get("tokens_end"),
        "ppl_end": summary.get("ppl_end"),
        "after_over_orig": summary.get("after_over_orig"),
        "gate_ppl": summary.get("gate_ppl"),
        "gate_status": summary.get("gate_status"),
        "ppl_best_in_session": summary.get("ppl_best_in_session"),
    }
    append_jsonl(root / "ledger.jsonl", ledger_row)

    latest = {
        "run_id": summary.get("run_id"),
        "session": summary.get("session"),
        "session_tag": tag,
        "step_end": summary.get("step_end"),
        "tokens_end": summary.get("tokens_end"),
        "ppl_end": summary.get("ppl_end"),
        "after_over_orig": summary.get("after_over_orig"),
        "gate_status": summary.get("gate_status"),
        "checkpoint_final": ckpt_abs,
        "checkpoint_final_rel": ckpt_rel,
        "session_dir": str(sess_dir),
    }
    write_json(root / "LATEST.json", latest)
    rebuild_curves(root)
    logger.info("Session pack written: %s (ledger+LATEST updated)", sess_dir)
    return sess_dir


def rebuild_curves(run_root: Union[str, Path]) -> Dict[str, Path]:
    """Rebuild curves/*.csv from ledger + per-session metrics."""
    root = Path(run_root)
    curves = root / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    ledger = read_jsonl(root / "ledger.jsonl")
    # de-dupe by session keeping last
    by_sess: Dict[int, Dict[str, Any]] = {}
    for row in ledger:
        by_sess[int(row["session"])] = row
    recovery_path = curves / "recovery_ppl.csv"
    with recovery_path.open("w", encoding="utf-8") as f:
        f.write("session,session_tag,step_end,tokens_end,ppl_end,after_over_orig,gate_ppl,gate_status\n")
        for s in sorted(by_sess):
            r = by_sess[s]
            f.write(
                f"{s},{r.get('session_tag', session_tag(s))},"
                f"{r.get('step_end','')},{r.get('tokens_end','')},"
                f"{r.get('ppl_end','')},{r.get('after_over_orig','')},"
                f"{r.get('gate_ppl','')},{r.get('gate_status','')}\n"
            )
    out["recovery_ppl"] = recovery_path

    loss_path = curves / "train_loss.csv"
    bins_path = curves / "quant_bins.csv"
    with loss_path.open("w", encoding="utf-8") as fl, bins_path.open(
        "w", encoding="utf-8"
    ) as fb:
        fl.write(
            "session,step,tokens,event,loss,ce,kl,reg,lr,lambda,perplexity\n"
        )
        fb.write(
            "session,step,tokens,frac_-1,frac_-c,frac_+c,frac_+1,n\n"
        )
        for sdir in sorted((root / "sessions").glob("S*")):
            metrics = sdir / "metrics.jsonl"
            if not metrics.is_file():
                continue
            try:
                sess = int(sdir.name[1:])
            except ValueError:
                continue
            for row in read_jsonl(metrics):
                step = row.get("step", "")
                tokens = row.get("tokens", "")
                fl.write(
                    f"{sess},{step},{tokens},{row.get('event','')},"
                    f"{row.get('loss','')},{row.get('ce','')},{row.get('kl','')},"
                    f"{row.get('reg','')},{row.get('lr','')},{row.get('lambda','')},"
                    f"{row.get('perplexity','')}\n"
                )
                bins = row.get("quant_bins") or row.get("bins")
                if isinstance(bins, dict) and bins.get("frac"):
                    frac = bins["frac"]
                    fb.write(
                        f"{sess},{step},{tokens},"
                        f"{frac.get('-1','')},{frac.get('-c','')},"
                        f"{frac.get('+c','')},{frac.get('+1','')},"
                        f"{bins.get('n','')}\n"
                    )
    out["train_loss"] = loss_path
    out["quant_bins"] = bins_path
    return out


def merge_run_pack(run_root: Union[str, Path]) -> Dict[str, Path]:
    """Rebuild ledger-derived curves (idempotent)."""
    root = ensure_run_root(run_root)
    # Rebuild ledger from session_summary if ledger missing/empty but sessions exist
    ledger_path = root / "ledger.jsonl"
    summaries = sorted((root / "sessions").glob("S*/session_summary.json"))
    if summaries and (not ledger_path.is_file() or ledger_path.stat().st_size == 0):
        for sp in summaries:
            s = read_json(sp)
            append_jsonl(
                ledger_path,
                {
                    "run_id": s.get("run_id"),
                    "session": s.get("session"),
                    "session_tag": s.get("session_tag"),
                    "step_end": s.get("step_end"),
                    "tokens_end": s.get("tokens_end"),
                    "ppl_end": s.get("ppl_end"),
                    "after_over_orig": s.get("after_over_orig"),
                    "gate_ppl": s.get("gate_ppl"),
                    "gate_status": s.get("gate_status"),
                    "ppl_best_in_session": s.get("ppl_best_in_session"),
                },
            )
    return rebuild_curves(root)
