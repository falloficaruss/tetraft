"""Unit tests for data packing / JSONL helpers (no HF download)."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from data import (
    JsonlTextDataset,
    MemmapPackedCausalLMDataset,
    PackedCausalLMDataset,
    build_memmap_pack,
    build_packed_dataloader_memmap,
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


class _FakeBatchTokenizer:
    """Batch-capable stub: whitespace-split → stable ids (eos=0)."""

    def __init__(self):
        self.eos_token_id = 0
        self._vocab = {"<eos>": 0}
        self.name_or_path = "fake-batch"
        self.n_calls = 0

    def _encode(self, text):
        ids = []
        for tok in text.split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab[tok])
        return ids

    def __call__(self, text, add_special_tokens=False, truncation=False, return_attention_mask=False):
        self.n_calls += 1
        if isinstance(text, list):
            return {"input_ids": [self._encode(t) for t in text]}
        return {"input_ids": self._encode(text)}


class TestMemmapPack:
    def _write_train(self, tmp_path: Path, texts) -> Path:
        path = tmp_path / "train.jsonl"
        write_jsonl(path, [{"text": t} for t in texts])
        return path

    def test_parity_with_inmemory_pack(self, tmp_path: Path):
        texts = [
            "a b c d e f g h",  # 8 tokens
            "i j k l",  # 4
            "",  # skipped
            "m n o p q r",  # 6
        ]
        path = self._write_train(tmp_path, texts)
        tok = _FakeBatchTokenizer()
        seq = 4
        tokens_path, n_tokens = build_memmap_pack(
            path, tok, cache_dir=tmp_path / "cache", chunk_texts=1
        )
        mem = MemmapPackedCausalLMDataset(tokens_path, n_tokens, seq)
        ref = tokenize_and_pack(
            ["a b c d e f g h", "i j k l", "m n o p q r"], tok, seq_length=seq
        )
        mem_blocks = [mem[i]["input_ids"].tolist() for i in range(len(mem))]
        ref_blocks = [list(b) for b in ref.blocks]
        assert mem_blocks == ref_blocks

    def test_cache_hit_skips_tokenizer(self, tmp_path: Path):
        path = self._write_train(tmp_path, ["a b c d", "e f g h"])
        tok = _FakeBatchTokenizer()
        p1, n1 = build_memmap_pack(path, tok, cache_dir=tmp_path / "cache")
        assert tok.n_calls > 0

        class _Poison:
            eos_token_id = 0
            name_or_path = "fake-batch"  # same key ingredient as builder tok

            def __call__(self, *a, **k):
                raise AssertionError("tokenizer called on cache hit")

        p2, n2 = build_memmap_pack(path, _Poison(), cache_dir=tmp_path / "cache")
        assert (p1, n1) == (p2, n2)

    def test_trailing_partial_block_dropped(self, tmp_path: Path):
        # 4 + 1(eos) + 4 = 9 tokens → 2 blocks of 4, drop 1
        path = self._write_train(tmp_path, ["a b c d", "e f g h"])
        loader = build_packed_dataloader_memmap(
            path,
            _FakeBatchTokenizer(),
            seq_length=4,
            batch_size=2,
            cache_dir=tmp_path / "cache",
        )
        ds = loader.dataset
        assert len(ds) == 2
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, 4)
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].shape == (2, 4)
        assert batch["attention_mask"].sum().item() == 8

    def test_max_texts_caps_docs(self, tmp_path: Path):
        path = self._write_train(tmp_path, ["a b c d", "e f g h", "i j k l"])
        _, n_tokens = build_memmap_pack(
            path, _FakeBatchTokenizer(), cache_dir=tmp_path / "cache", max_texts=1
        )
        assert n_tokens == 4  # only first doc, no eos prefix

    def test_empty_raises(self, tmp_path: Path):
        path = self._write_train(tmp_path, ["", "   "])
        with pytest.raises(ValueError, match="No non-empty"):
            build_memmap_pack(path, _FakeBatchTokenizer(), cache_dir=tmp_path / "cache")

    def test_no_full_block_raises(self, tmp_path: Path):
        path = self._write_train(tmp_path, ["a b"])
        tokens_path, n_tokens = build_memmap_pack(
            path, _FakeBatchTokenizer(), cache_dir=tmp_path / "cache"
        )
        with pytest.raises(ValueError, match="at least one full block"):
            MemmapPackedCausalLMDataset(tokens_path, n_tokens, seq_length=8)


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
