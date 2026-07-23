"""Unit tests for data packing / JSONL helpers (no HF download)."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from data import (
    JsonlTextDataset,
    PackedCausalLMDataset,
    pack_token_ids,
    read_jsonl,
    resolve_data_path,
    tokenize_and_pack,
    write_jsonl,
)


class _FakeTokenizer:
    """Minimal tokenizer stub: whitespace-split → stable ids."""

    def __init__(self):
        self.eos_token_id = 0
        self._vocab = {"<eos>": 0}

    def __call__(self, text, add_special_tokens=False, truncation=False, return_attention_mask=False):
        ids = []
        for tok in text.split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab[tok])
        return {"input_ids": ids}


class TestJsonl:
    def test_write_read_roundtrip(self, tmp_path: Path):
        path = tmp_path / "t.jsonl"
        rows = [{"text": "hello world"}, {"text": "foo bar"}]
        n = write_jsonl(path, rows)
        assert n == 2
        loaded = read_jsonl(path)
        assert loaded == rows

    def test_jsonl_text_dataset_skips_empty(self, tmp_path: Path):
        path = tmp_path / "t.jsonl"
        write_jsonl(
            path,
            [
                {"text": "keep me"},
                {"text": "   "},
                {"text": ""},
                {"other": "nope"},
                {"text": "also keep"},
            ],
        )
        ds = JsonlTextDataset(path)
        assert len(ds) == 2
        assert ds[0]["text"] == "keep me"
        assert ds[1]["text"] == "also keep"

    def test_jsonl_empty_raises(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        write_jsonl(path, [{"text": ""}])
        with pytest.raises(ValueError, match="No non-empty"):
            JsonlTextDataset(path)


class TestPackTokenIds:
    def test_basic_packing(self):
        # 10 tokens → 2 full blocks of 4, drop remainder 2
        ids = [list(range(10))]
        blocks = pack_token_ids(ids, seq_length=4)
        assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7]]

    def test_eos_between_docs(self):
        blocks = pack_token_ids([[1, 2], [3, 4, 5]], seq_length=3, eos_id=99)
        # flat: 1,2,99,3,4,5 → two blocks
        assert blocks == [[1, 2, 99], [3, 4, 5]]

    def test_short_input_yields_empty(self):
        assert pack_token_ids([[1, 2]], seq_length=8) == []

    def test_seq_length_validation(self):
        with pytest.raises(ValueError):
            pack_token_ids([[1, 2, 3]], seq_length=1)


class TestPackedDataset:
    def test_getitem_shapes(self):
        ds = PackedCausalLMDataset([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"].tolist() == [1, 2, 3, 4]
        assert item["labels"].tolist() == [1, 2, 3, 4]
        assert item["attention_mask"].tolist() == [1, 1, 1, 1]
        assert item["input_ids"].dtype == torch.long

    def test_empty_blocks_raise(self):
        with pytest.raises(ValueError):
            PackedCausalLMDataset([])


class TestTokenizeAndPack:
    def test_with_fake_tokenizer(self):
        tok = _FakeTokenizer()
        texts = [
            "a b c d e f g h",  # 8 tokens
            "i j k l",  # 4 tokens + eos between → enough for more blocks
        ]
        ds = tokenize_and_pack(texts, tok, seq_length=4)
        assert len(ds) >= 2
        batch = next(iter(DataLoader(ds, batch_size=2)))
        assert batch["input_ids"].shape == (2, 4)
        assert batch["labels"].shape == (2, 4)


class TestResolveDataPath:
    def test_explicit_wins(self, tmp_path: Path):
        p = tmp_path / "val.jsonl"
        p.write_text('{"text": "x"}\n', encoding="utf-8")
        assert resolve_data_path(str(p), "/nonexistent/val.jsonl") == str(p)

    def test_falls_through_candidates(self, tmp_path: Path):
        p = tmp_path / "train.jsonl"
        p.write_text('{"text": "x"}\n', encoding="utf-8")
        assert resolve_data_path(None, "/nope.jsonl", str(p)) == str(p)

    def test_none_when_missing(self):
        assert resolve_data_path(None, "/no/such/file.jsonl") is None
