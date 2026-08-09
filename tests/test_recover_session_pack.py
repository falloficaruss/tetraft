"""Tests for the crashed-hop recovery helper (scripts/recover_session_pack)."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from recover_session_pack import find_train_dir, recover_session_pack
from run_pack import ensure_run_root, read_json, read_jsonl


def _make_train_dir(tmp_path, session, ppl=33.86):
    d = tmp_path / f"train_S{session:02d}"
    d.mkdir(parents=True)
    (d / "checkpoint-final").write_bytes(f"ckpt-{session}".encode())
    (d / "checkpoint-best").write_bytes(b"best-marker")
    smoke = {
        "preset": "heal_kl_trust_400m",
        "ppl_original": 17.67,
        "ppl_shock": 17800.0,
        "ppl_after_smoke": ppl,
        "ppl_best_in_session": ppl + 0.5,
        "steps_ran": 36624,
        "resumed_step": 24416,
        "inventory_summary": {"n_linear": 96, "n_eligible": 96},
        "distill": {"alpha": 0.3, "temperature": 2.0},
    }
    (d / "smoke_results.json").write_text(json.dumps(smoke), encoding="utf-8")
    (d / "metrics.jsonl").write_text('{"event":"log","step":36624}\n', encoding="utf-8")
    return d


def test_recover_writes_pack_and_moves_checkpoint(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    stale = root / "sessions" / "S04"
    stale.mkdir(parents=True)
    (stale / "checkpoint-final").write_bytes(b"stale-s04")
    (stale / "session_summary.json").write_text("{}", encoding="utf-8")

    train = _make_train_dir(tmp_path, 6)
    summary = recover_session_pack(6, train_dir=train, pack_root=root)

    assert summary["session_tag"] == "S06"
    assert summary["ppl_end"] == pytest.approx(33.86)
    assert summary["step_start"] == 24416
    assert summary["step_end"] == 36624

    dst = root / "sessions/S06/checkpoint-final"
    assert dst.is_file()
    assert dst.read_bytes() == b"ckpt-6"
    assert not (train / "checkpoint-final").exists()
    assert not (train / "checkpoint-best").exists()
    assert not (stale / "checkpoint-final").exists()
    assert (root / "sessions/S06/session_summary.json").is_file()
    assert (root / "sessions/S06/metrics.jsonl").is_file()

    latest = read_json(root / "LATEST.json")
    assert latest["session_tag"] == "S06"
    assert latest["checkpoint_final_rel"] == "sessions/S06/checkpoint-final"

    ledger = read_jsonl(root / "ledger.jsonl")
    assert any(r["session"] == 6 and r["step_end"] == 36624 for r in ledger)

    csv = (root / "curves" / "recovery_ppl.csv").read_text(encoding="utf-8")
    assert "S06" in csv


def test_recover_requires_finished_run(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    train = tmp_path / "train_S06"
    train.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        recover_session_pack(6, train_dir=train, pack_root=root)
    (train / "checkpoint-final").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        recover_session_pack(6, train_dir=train, pack_root=root)


def test_recover_copy_fallback_when_replace_fails(tmp_path: Path, monkeypatch):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    train = _make_train_dir(tmp_path, 6)

    def _boom(*args, **kwargs):
        raise OSError(18, "cross-device link")

    monkeypatch.setattr(os, "replace", _boom)
    summary = recover_session_pack(6, train_dir=train, pack_root=root)

    dst = root / "sessions/S06/checkpoint-final"
    assert dst.is_file()
    assert dst.read_bytes() == b"ckpt-6"
    assert not (train / "checkpoint-final").exists()
    assert summary["session_tag"] == "S06"


def test_recover_overwrites_partial_dst(tmp_path: Path):
    root = ensure_run_root(tmp_path / "heal_kl_trust_400m")
    sess = root / "sessions" / "S06"
    sess.mkdir(parents=True)
    (sess / "checkpoint-final").write_bytes(b"partial-600mb")
    (sess / "session_summary.json").write_text("{}", encoding="utf-8")

    train = _make_train_dir(tmp_path, 6)
    recover_session_pack(6, train_dir=train, pack_root=root)

    assert (sess / "checkpoint-final").read_bytes() == b"ckpt-6"
    assert not (train / "checkpoint-final").exists()
    latest = read_json(root / "LATEST.json")
    assert latest["session_tag"] == "S06"


def test_find_train_dir_searches_input_root(tmp_path: Path):
    inp = tmp_path / "input"
    train = _make_train_dir(inp, 6)
    train.rename(inp / "train_heal_kl_trust_400m_S06")
    train = inp / "train_heal_kl_trust_400m_S06"
    (inp / "unrelated").mkdir()

    assert find_train_dir(6, input_root=inp) == train
    assert find_train_dir(7, input_root=inp) is None
    assert find_train_dir(6, train_dir=train, input_root=inp) == train

    train_ckpt = train / "checkpoint-final"
    train_ckpt.unlink()
    assert find_train_dir(6, input_root=inp) is None