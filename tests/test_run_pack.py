"""Tests for paper run pack helpers."""

from pathlib import Path

from run_pack import (
    HORIZON_TRUST_400M,
    N_SESSIONS_TRUST_400M,
    SESSION_MICRO_STEPS,
    build_session_summary,
    ensure_run_root,
    merge_run_pack,
    rebuild_curves,
    session_gate_ppl,
    session_stop_step,
    session_tag,
    write_session_pack,
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
