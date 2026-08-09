"""Tests for paper run pack helpers."""

from pathlib import Path

from run_pack import (
    HORIZON_TRUST_400M,
    N_SESSIONS_TRUST_400M,
    SESSION_MICRO_STEPS,
    build_session_summary,
    ensure_run_root,
    find_resume_checkpoint,
    merge_run_pack,
    prune_old_session_checkpoints,
    rebuild_curves,
    session_gate_ppl,
    session_stop_step,
    session_tag,
    write_session_pack,
)


def _summary(session: int, ppl: float):
    return build_session_summary(
        run_id="heal_kl_trust_400m",
        session=session,
        step_start=(session - 1) * SESSION_MICRO_STEPS,
        step_end=session * SESSION_MICRO_STEPS,
        tokens_per_step=4096,
        ppl_end=ppl,
        ppl_best_in_session=ppl,
        ppl_original=17.67,
        ppl_shock=17800.0,
        dna={"ste_mode": "trust", "distill_alpha": 0.3},
    )


def test_horizon_math():
    assert SESSION_MICRO_STEPS == 6104
    assert N_SESSIONS_TRUST_400M == 16
    assert HORIZON_TRUST_400M == 97664
    assert session_stop_step(1) == 6104
    assert session_stop_step(16) == 97664
    assert session_tag(3) == "S03"


def test_gates():
    assert session_gate_ppl(1) == 48.65
    assert session_gate_ppl(2) == 34.38
    assert session_gate_ppl(4) == 30.0
    assert session_gate_ppl(3) == 34.38  # nearest lower


def test_write_session_pack_and_merge(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    assert (root / "RUN_META.json").is_file()
    assert (root / "baselines.json").is_file()

    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"event":"log","step":100,"tokens":409600,"loss":1.0,"ce":2.0,'
        '"kl":0.5,"reg":0.0,"lr":1e-4,"lambda":0.5}\n'
        '{"event":"eval","step":256,"tokens":1048576,"perplexity":90.0,'
        '"quant_bins":{"n":10,"frac":{"-1":0.3,"-c":0.2,"+c":0.2,"+1":0.3}}}\n',
        encoding="utf-8",
    )
    ckpt = tmp_path / "checkpoint-final"
    ckpt.write_bytes(b"fake")

    summary = build_session_summary(
        run_id="heal_kl_trust_400m",
        session=1,
        step_start=0,
        step_end=6104,
        tokens_per_step=4096,
        ppl_end=40.0,
        ppl_best_in_session=41.0,
        ppl_original=17.67,
        ppl_shock=17800.0,
        dna={"ste_mode": "trust", "distill_alpha": 0.3},
    )
    assert summary["gate_status"] == "PASS"
    assert summary["tokens_end"] == 6104 * 4096
    assert abs(summary["after_over_orig"] - 40.0 / 17.67) < 1e-6

    sess = write_session_pack(
        root,
        summary,
        smoke_results={"ppl_after_smoke": 40.0},
        metrics_src=metrics,
        checkpoint_final_src=ckpt,
    )
    assert (sess / "session_summary.json").is_file()
    assert (sess / "metrics.jsonl").is_file()
    assert (sess / "checkpoint-final").is_file()
    assert (root / "ledger.jsonl").is_file()
    assert (root / "LATEST.json").is_file()
    assert (root / "curves" / "recovery_ppl.csv").is_file()

    summary2 = build_session_summary(
        run_id="heal_kl_trust_400m",
        session=2,
        step_start=6104,
        step_end=12208,
        tokens_per_step=4096,
        ppl_end=33.0,
        ppl_best_in_session=33.5,
        ppl_original=None,
        ppl_shock=None,
    )
    assert summary2["gate_status"] == "PASS"
    write_session_pack(root, summary2, smoke_results={"ppl_after_smoke": 33.0})

    out = merge_run_pack(root)
    text = (root / "curves" / "recovery_ppl.csv").read_text(encoding="utf-8")
    assert "S01" in text and "S02" in text
    assert "recovery_ppl" in out


def test_pack_keeps_only_latest_checkpoint(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    s01_ckpt = tmp_path / "train_S01" / "checkpoint-final"
    s01_ckpt.parent.mkdir(parents=True)
    s01_ckpt.write_bytes(b"ckpt-s01")
    s01_metrics = tmp_path / "train_S01" / "metrics.jsonl"
    s01_metrics.write_text('{"event":"log","step":6104,"tokens":25001984}\n', encoding="utf-8")

    write_session_pack(
        root, _summary(1, 40.0), metrics_src=s01_metrics, checkpoint_final_src=s01_ckpt
    )
    assert (root / "sessions/S01" / "checkpoint-final").is_file()

    s02_ckpt = tmp_path / "train_S02" / "checkpoint-final"
    s02_ckpt.parent.mkdir(parents=True)
    s02_ckpt.write_bytes(b"ckpt-s02")

    write_session_pack(
        root,
        _summary(2, 33.0),
        checkpoint_final_src=s02_ckpt,
        delete_checkpoint_source=True,
    )

    assert not (root / "sessions/S01" / "checkpoint-final").exists()
    assert (root / "sessions/S02" / "checkpoint-final").is_file()
    assert not s02_ckpt.exists()

    for small in ("session_summary.json", "metrics.jsonl"):
        assert (root / "sessions/S01" / small).is_file()
    assert (root / "sessions/S02" / "session_summary.json").is_file()

    latest = __import__("json").loads(
        (root / "LATEST.json").read_text(encoding="utf-8")
    )
    assert latest["session_tag"] == "S02"
    resume = find_resume_checkpoint(root, session=3)
    assert resume is not None and resume.name == "checkpoint-final"
    assert resume.parent.name == "S02"


def test_pack_delete_source_keeps_dst_content(tmp_path: Path):
    """Same-fs delete path renames (or copies+removes) — dst must equal src."""
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    s01_ckpt = tmp_path / "train_S01" / "checkpoint-final"
    s01_ckpt.parent.mkdir(parents=True)
    s01_ckpt.write_bytes(b"payload-01")
    write_session_pack(
        root, _summary(1, 40.0), checkpoint_final_src=s01_ckpt, delete_checkpoint_source=True
    )
    assert (root / "sessions/S01" / "checkpoint-final").read_bytes() == b"payload-01"
    assert not s01_ckpt.exists()


def test_pack_keep_source_preserves_both(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    ckpt = tmp_path / "train_S01" / "checkpoint-final"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"payload-01")
    write_session_pack(
        root, _summary(1, 40.0), checkpoint_final_src=ckpt, delete_checkpoint_source=False
    )
    assert ckpt.is_file()
    assert (root / "sessions/S01" / "checkpoint-final").read_bytes() == b"payload-01"


def test_prune_old_session_checkpoints_public_alias(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    for tag in ("S01", "S02"):
        sdir = root / "sessions" / tag
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "checkpoint-final").write_bytes(b"x")
    prune_old_session_checkpoints(root, keep_tag="S02")
    assert not (root / "sessions/S01/checkpoint-final").exists()
    assert (root / "sessions/S02/checkpoint-final").is_file()


def test_pack_no_prune_without_new_checkpoint(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    s01_ckpt = tmp_path / "train_S01" / "checkpoint-final"
    s01_ckpt.parent.mkdir(parents=True)
    s01_ckpt.write_bytes(b"ckpt-s01")

    write_session_pack(root, _summary(1, 40.0), checkpoint_final_src=s01_ckpt)
    assert (root / "sessions/S01" / "checkpoint-final").is_file()

    write_session_pack(root, _summary(2, 33.0))
    assert (root / "sessions/S01" / "checkpoint-final").is_file()
    assert s01_ckpt.is_file()
    assert not (root / "sessions/S02" / "checkpoint-final").exists()


def test_pack_flags_disable_prune_and_delete(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    s01_ckpt = tmp_path / "train_S01" / "checkpoint-final"
    s01_ckpt.parent.mkdir(parents=True)
    s01_ckpt.write_bytes(b"ckpt-s01")
    s02_ckpt = tmp_path / "train_S02" / "checkpoint-final"
    s02_ckpt.parent.mkdir(parents=True)
    s02_ckpt.write_bytes(b"ckpt-s02")

    write_session_pack(root, _summary(1, 40.0), checkpoint_final_src=s01_ckpt)
    write_session_pack(
        root,
        _summary(2, 33.0),
        checkpoint_final_src=s02_ckpt,
        keep_only_latest_checkpoint=False,
        delete_checkpoint_source=False,
    )

    assert (root / "sessions/S01" / "checkpoint-final").is_file()
    assert (root / "sessions/S02" / "checkpoint-final").is_file()
    assert s01_ckpt.is_file() and s02_ckpt.is_file()
