"""Recover a marathon session pack after the train hop finished but the pack
write crashed (e.g. ENOSPC on /kaggle/working at the final copy).

Self-contained against attached tetraft-code: imports only the stable run_pack
helpers, so it can be pasted as a cell in the crashed Kaggle kernel and run
there (no dataset refresh needed). It moves (renames; copy fallback) the
finished checkpoint-final into the pack, prunes every other session checkpoint
first, writes session_summary/ledger/LATEST and rebuilds curves.

The finished train dir can be the live kernel's ``/kaggle/working`` output OR
an attached input dataset: when ``--train-dir`` is omitted, ``/kaggle/input``
is searched for ``train_*_Sxx`` (upload the extracted output zip of a dead
background run as a Dataset and attach it).

Usage::

    python recover_session_pack.py --session 6 \
        --train-dir /kaggle/working/train_heal_kl_trust_400m_S06 \
        --pack-root /kaggle/working/heal_kl_trust_400m

    # or, when the run output is attached as an input dataset:
    python recover_session_pack.py --session 6 \
        --pack-root /kaggle/working/heal_kl_trust_400m

Then publish the pack root as Dataset ``tetraft-heal-kl-trust-400m`` and start
the next hop normally.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from run_pack import (
    append_jsonl,
    build_session_summary,
    ensure_run_root,
    read_json,
    rebuild_curves,
    session_tag,
    write_json,
)

RUN_ID = "heal_kl_trust_400m"
REF_ORIG_PPL = 17.67


def find_train_dir(
    session: int,
    train_dir: Optional[Union[str, Path]] = None,
    input_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Locate a finished hop's training output.

    Checks an explicitly supplied path first, then the ``/kaggle/input`` tree
    (recovery dataset uploaded from a dead/background kernel's output zip).
    Shallow globs keep the search cheap even when the large pack dataset is
    attached; only dirs with both ``checkpoint-final`` and ``smoke_results.json``
    (i.e. a completed run) qualify.
    """
    tag = session_tag(session)
    name = f"train_*_{tag}"
    candidates: list[Path] = []
    if train_dir is not None:
        candidates.append(Path(train_dir))
    root = Path(input_root) if input_root is not None else Path("/kaggle/input")
    if root.is_dir():
        for depth in range(0, 4):
            pattern = name if depth == 0 else ("*/" * depth) + name
            candidates.extend(sorted(root.glob(pattern)))
    for cand in candidates:
        if (
            cand.is_dir()
            and (cand / "checkpoint-final").is_file()
            and (cand / "smoke_results.json").is_file()
        ):
            return cand
    return None


def recover_session_pack(
    session: int,
    *,
    train_dir: Optional[Union[str, Path]] = None,
    pack_root: Union[str, Path],
    tokens_per_step: int = 4096,
    orig_ref: Optional[float] = None,
) -> Dict[str, Any]:
    """Write the finished run's session pack from a crashed hop.

    Reads ``smoke_results.json`` from the training output dir, moves (renames,
    copy fallback) ``checkpoint-final`` into ``pack_root/sessions/Sxx``, prunes
    every other session's checkpoint, and appends the ledger + LATEST.json +
    curves.
    """
    train = find_train_dir(session, train_dir=train_dir)
    if train is None:
        raise FileNotFoundError(
            f"no finished train dir for {session_tag(session)} "
            "(needs checkpoint-final + smoke_results.json); pass --train-dir "
            "or attach the run output as an input dataset"
        )
    root = ensure_run_root(pack_root, run_id=RUN_ID)
    smoke_path = train / "smoke_results.json"
    if not smoke_path.is_file():
        raise FileNotFoundError(f"missing {smoke_path}")
    ckpt_src = train / "checkpoint-final"
    if not ckpt_src.is_file():
        raise FileNotFoundError(f"missing {ckpt_src}")

    smoke = read_json(smoke_path)
    tag = session_tag(session)
    step_end = int(smoke.get("steps_ran") or 0)
    if step_end <= 0:
        raise ValueError(f"smoke_results has no steps_ran: {smoke_path}")
    step_start = int(smoke.get("resumed_step") or 0)
    ppl_end = smoke.get("ppl_after_smoke")
    meta = read_json(root / "RUN_META.json") if (root / "RUN_META.json").is_file() else {}
    orig = orig_ref or REF_ORIG_PPL

    summary = build_session_summary(
        run_id=RUN_ID,
        session=session,
        step_start=step_start,
        step_end=step_end,
        tokens_per_step=tokens_per_step,
        ppl_end=float(ppl_end) if ppl_end is not None else None,
        ppl_best_in_session=smoke.get("ppl_best_in_session"),
        ppl_original=smoke.get("ppl_original"),
        ppl_shock=smoke.get("ppl_shock"),
        orig_ref=orig,
        dna=dict(meta.get("dna") or {}),
        resumed_from=smoke.get("resumed_from") or smoke.get("resume_from"),
        inventory_summary=smoke.get("inventory_summary"),
    )

    sess_dir = root / "sessions" / tag
    sess_dir.mkdir(parents=True, exist_ok=True)

    # Prune every other session's checkpoint FIRST so the rename below needs
    # no additional disk.
    for old_ckpt in sorted((root / "sessions").glob("S*/checkpoint-final")):
        if old_ckpt.parent.name == tag:
            continue
        try:
            old_ckpt.unlink()
        except OSError as e:
            print(f"warning: failed to prune {old_ckpt}: {e}")

    # Small artifacts (copy; these are KB-scale). Overwrite any stale partial
    # files left by the crashed pack write.
    for fn in ("smoke_results.json", "metrics.jsonl", "linear_inventory.json"):
        fsrc = train / fn
        if fsrc.is_file():
            shutil.copy2(fsrc, sess_dir / fn)

    write_json(sess_dir / "session_summary.json", summary)

    # Move the full checkpoint (rename: no extra disk on the same volume;
    # copy fallback covers cross-device moves e.g. /kaggle/input -> working).
    # Overwrite any partial dst from the crashed write. Then drop the
    # weights-only best to reclaim space for the still-open hop.
    ckpt_dst = sess_dir / "checkpoint-final"
    if ckpt_dst.resolve() == ckpt_src.resolve():
        pass
    else:
        if ckpt_dst.is_file():
            ckpt_dst.unlink()
        try:
            os.replace(ckpt_src, ckpt_dst)
        except OSError as e:
            print(f"warning: rename failed ({e}); copying checkpoint instead")
            shutil.copy2(ckpt_src, ckpt_dst)
            try:
                ckpt_src.unlink()
            except OSError:
                pass  # read-only source (e.g. /kaggle/input)
    best = train / "checkpoint-best"
    if best.is_file():
        try:
            best.unlink()
            print(f"removed {best}")
        except OSError as e:
            print(f"warning: failed to remove {best}: {e}")

    row = {
        "run_id": RUN_ID,
        "session": session,
        "session_tag": tag,
        "step_end": summary.get("step_end"),
        "tokens_end": summary.get("tokens_end"),
        "ppl_end": summary.get("ppl_end"),
        "after_over_orig": summary.get("after_over_orig"),
        "gate_ppl": summary.get("gate_ppl"),
        "gate_status": summary.get("gate_status"),
        "ppl_best_in_session": summary.get("ppl_best_in_session"),
    }
    append_jsonl(root / "ledger.jsonl", row)
    write_json(
        root / "LATEST.json",
        {
            "run_id": RUN_ID,
            "session": session,
            "session_tag": tag,
            "step_end": summary.get("step_end"),
            "tokens_end": summary.get("tokens_end"),
            "ppl_end": summary.get("ppl_end"),
            "after_over_orig": summary.get("after_over_orig"),
            "gate_status": summary.get("gate_status"),
            "checkpoint_final": str(ckpt_dst),
            "checkpoint_final_rel": str(ckpt_dst.relative_to(root)),
            "session_dir": str(sess_dir),
        },
    )
    rebuild_curves(root)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--train-dir", type=str, default=None)
    p.add_argument("--pack-root", type=str, required=True)
    p.add_argument("--tokens-per-step", type=int, default=4096)
    p.add_argument("--orig-ppl", type=float, default=None)
    args = p.parse_args(argv)

    s = recover_session_pack(
        args.session,
        train_dir=args.train_dir,
        pack_root=args.pack_root,
        tokens_per_step=args.tokens_per_step,
        orig_ref=args.orig_ppl,
    )
    print(
        f"RECOVERED {s.get('session_tag')}: ppl_end={s.get('ppl_end')} "
        f"after/orig={s.get('after_over_orig')} gate<{s.get('gate_ppl')} "
        f"-> {s.get('gate_status')}"
    )
    print(f" pack ready at: {args.pack_root}")
    print(" Next: publish as Dataset tetraft-heal-kl-trust-400m, then run SESSION = next hop.")


if __name__ == "__main__":
    main()