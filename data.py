"""FineWeb-Edu sampling and causal-LM dataloaders for TetraFT Phase 1."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Default HF source for one-time sample builds (not bundled in the repo).
DEFAULT_HF_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_HF_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_jsonl(path: Union[str, Path], records: Sequence[Dict[str, Any]]) -> int:
    """Write *records* as JSONL. Returns number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory (suitable for smoke / ablate samples)."""
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Stream JSONL rows without loading the full file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Sample builder (one-time; requires `datasets` + HF access)
# ---------------------------------------------------------------------------

@dataclass
class SampleBuildMeta:
    """Metadata written alongside a fixed FineWeb-Edu sample."""

    hf_dataset: str
    hf_split: str
    hf_revision: Optional[str]
    seed: int
    max_train_tokens: int
    max_val_tokens: int
    text_field: str
    train_docs: int
    val_docs: int
    train_tokens_est: int
    val_tokens_est: int


def _estimate_tokens(text: str) -> int:
    """Rough whitespace token estimate used only for sample budgets."""
    if not text:
        return 0
    return max(1, len(text.split()))


def build_fineweb_sample(
    output_dir: Union[str, Path],
    max_train_tokens: int = 50_000_000,
    max_val_tokens: int = 500_000,
    seed: int = 42,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_split: str = DEFAULT_HF_SPLIT,
    hf_revision: Optional[str] = None,
    text_field: str = DEFAULT_TEXT_FIELD,
    streaming: bool = True,
) -> SampleBuildMeta:
    """Stream FineWeb-Edu and write fixed ``train.jsonl`` + ``val.jsonl``.

    Val is taken first (held-out), then train — never mixed. Requires the
    Hugging Face ``datasets`` package and network access for the one-time build.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "build_fineweb_sample requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    meta_path = output_dir / "sample_meta.json"

    rng = random.Random(seed)
    load_kwargs: Dict[str, Any] = {
        "path": hf_dataset,
        "split": hf_split,
        "streaming": streaming,
    }
    if hf_revision is not None:
        load_kwargs["revision"] = hf_revision

    logger.info(
        "Streaming %s (split=%s, revision=%s) → %s",
        hf_dataset,
        hf_split,
        hf_revision,
        output_dir,
    )
    ds = load_dataset(**load_kwargs)
    # Deterministic-ish shuffle buffer for streaming; order still depends on HF.
    if streaming and hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

    train_tokens = 0
    val_tokens = 0
    train_docs = 0
    val_docs = 0
    log_every = 500  # docs; useful on long Kaggle streams

    with train_path.open("w", encoding="utf-8") as f_train, val_path.open(
        "w", encoding="utf-8"
    ) as f_val:
        for row in ds:
            text = row.get(text_field)
            if not text or not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            n_tok = _estimate_tokens(text)
            rec = {"text": text}

            # Fill val first so it is a clean held-out prefix of the stream.
            if val_tokens < max_val_tokens:
                f_val.write(json.dumps(rec, ensure_ascii=False) + "\n")
                val_tokens += n_tok
                val_docs += 1
                if val_docs % log_every == 0:
                    logger.info(
                        "val progress: docs=%d tokens_est=%d / %d",
                        val_docs,
                        val_tokens,
                        max_val_tokens,
                    )
                continue

            if train_tokens >= max_train_tokens:
                break

            # Light extra shuffle via occasional skip (streaming-friendly).
            if rng.random() < 0.02:
                continue

            f_train.write(json.dumps(rec, ensure_ascii=False) + "\n")
            train_tokens += n_tok
            train_docs += 1
            if train_docs % log_every == 0:
                pct = 100.0 * train_tokens / max(max_train_tokens, 1)
                logger.info(
                    "train progress: docs=%d tokens_est=%d / %d (%.1f%%)",
                    train_docs,
                    train_tokens,
                    max_train_tokens,
                    pct,
                )

    meta = SampleBuildMeta(
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        hf_revision=hf_revision,
        seed=seed,
        max_train_tokens=max_train_tokens,
        max_val_tokens=max_val_tokens,
        text_field=text_field,
        train_docs=train_docs,
        val_docs=val_docs,
        train_tokens_est=train_tokens,
        val_tokens_est=val_tokens,
    )
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)
        f.write("\n")

    logger.info(
        "Wrote sample: train_docs=%d (~%d tok), val_docs=%d (~%d tok) → %s",
        train_docs,
        train_tokens,
        val_docs,
        val_tokens,
        output_dir,
    )
    return meta


# ---------------------------------------------------------------------------
# Datasets / packing
# ---------------------------------------------------------------------------

class JsonlTextDataset(Dataset):
    """In-memory JSONL dataset yielding ``{"text": ...}`` rows."""

    def __init__(
        self,
        path: Union[str, Path],
        text_field: str = DEFAULT_TEXT_FIELD,
    ):
        self.path = Path(path)
        self.text_field = text_field
        raw = read_jsonl(self.path)
        self.rows: List[str] = []
        for r in raw:
            t = r.get(text_field, "")
            if isinstance(t, str) and t.strip():
                self.rows.append(t.strip())
        if not self.rows:
            raise ValueError(f"No non-empty texts in {self.path} (field={text_field})")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"text": self.rows[idx]}


def pack_token_ids(
    token_id_lists: Sequence[Sequence[int]],
    seq_length: int,
    eos_id: Optional[int] = None,
) -> List[List[int]]:
    """Concatenate token lists and slice into fixed-length blocks of *seq_length*.

    If *eos_id* is given, it is inserted between documents. Trailing partial
    blocks shorter than *seq_length* are dropped (no padding in the packed set).
    """
    if seq_length < 2:
        raise ValueError(f"seq_length must be >= 2, got {seq_length}")

    flat: List[int] = []
    for i, ids in enumerate(token_id_lists):
        if not ids:
            continue
        if i > 0 and eos_id is not None:
            flat.append(eos_id)
        flat.extend(int(t) for t in ids)

    n_full = len(flat) // seq_length
    if n_full == 0:
        return []
    return [flat[i * seq_length : (i + 1) * seq_length] for i in range(n_full)]


class PackedCausalLMDataset(Dataset):
    """Packed token blocks for causal LM: ``input_ids`` / ``labels`` / ``attention_mask``."""

    def __init__(self, blocks: Sequence[Sequence[int]]):
        self.blocks = [list(b) for b in blocks]
        if not self.blocks:
            raise ValueError("PackedCausalLMDataset requires at least one full block")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.blocks[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        # Standard CLM: predict next token; same sequence as labels (shift inside model).
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def tokenize_and_pack(
    texts: Sequence[str],
    tokenizer,
    seq_length: int,
    add_special_tokens: bool = False,
) -> PackedCausalLMDataset:
    """Tokenize raw texts and pack into a ``PackedCausalLMDataset``."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    all_ids: List[List[int]] = []
    for text in texts:
        enc = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_attention_mask=False,
        )
        ids = enc["input_ids"]
        if ids:
            all_ids.append(ids)
    blocks = pack_token_ids(all_ids, seq_length=seq_length, eos_id=eos_id)
    return PackedCausalLMDataset(blocks)


def build_packed_dataloader(
    path: Union[str, Path],
    tokenizer,
    seq_length: int,
    batch_size: int,
    text_field: str = DEFAULT_TEXT_FIELD,
    shuffle: bool = False,
    num_workers: int = 0,
    max_texts: Optional[int] = None,
) -> DataLoader:
    """Load JSONL → tokenize → pack → ``DataLoader``."""
    ds = JsonlTextDataset(path, text_field=text_field)
    texts = ds.rows if max_texts is None else ds.rows[:max_texts]
    packed = tokenize_and_pack(texts, tokenizer, seq_length=seq_length)
    return DataLoader(
        packed,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Memmap packing (large samples; bounded RAM)
# ---------------------------------------------------------------------------

# Schema bump when the flat-stream layout changes (EOS placement, dtype, ...).
_MEMMAP_PACK_VERSION = "v1"
_MEMMAP_DTYPE = np.dtype("<i4")  # int32: vocab ids < 2**31; 400M tokens ≈ 1.6 GB


def default_pack_cache_dir() -> Path:
    """Default root for token memmap caches (env → Kaggle working → home)."""
    env = os.environ.get("TETRAFT_PACK_CACHE")
    if env:
        return Path(env)
    if os.path.isdir("/kaggle/working"):
        return Path("/kaggle/working/tetraft_pack_cache")
    return Path.home() / ".cache" / "tetraft" / "pack"


def _pack_cache_key(
    path: Path,
    tokenizer,
    text_field: str,
    max_texts: Optional[int],
) -> str:
    st = path.stat()
    h = hashlib.sha1()
    h.update(_MEMMAP_PACK_VERSION.encode())
    h.update(str(path.resolve()).encode())
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())
    h.update(str(getattr(tokenizer, "name_or_path", "") or "").encode())
    h.update(text_field.encode())
    h.update(str(max_texts).encode())
    return h.hexdigest()[:16]


def build_memmap_pack(
    path: Union[str, Path],
    tokenizer,
    cache_dir: Union[str, Path],
    text_field: str = DEFAULT_TEXT_FIELD,
    max_texts: Optional[int] = None,
    chunk_texts: int = 1024,
) -> Tuple[Path, int]:
    """Tokenize JSONL in chunks into a flat int32 token stream on disk.

    Returns ``(tokens_path, n_tokens)``. EOS is inserted between documents
    (same semantics as ``pack_token_ids``); the trailing partial block is left
    for the dataset layer to drop. Results are cached under *cache_dir* keyed
    by (file, tokenizer, field, max_texts), so repeat sessions skip the build.
    """
    path = Path(path)
    cache_dir = Path(cache_dir)
    key = _pack_cache_key(path, tokenizer, text_field, max_texts)
    pack_dir = cache_dir / key
    tokens_path = pack_dir / "tokens.i32"
    meta_path = pack_dir / "meta.json"

    if meta_path.is_file() and tokens_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        n_tokens = int(meta["n_tokens"])
        logger.info("memmap pack cache hit: %s (%d tokens)", pack_dir, n_tokens)
        return tokens_path, n_tokens

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pack_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = pack_dir / "tokens.i32.tmp"

    n_tokens = 0
    n_docs = 0
    chunk: List[str] = []

    def _flush(fh) -> None:
        nonlocal n_tokens, n_docs, chunk
        if not chunk:
            return
        enc = tokenizer(
            chunk,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        for ids in enc["input_ids"]:
            if not ids:
                continue
            if n_docs > 0 and eos_id is not None:
                np.asarray([eos_id], dtype=_MEMMAP_DTYPE).tofile(fh)
                n_tokens += 1
            arr = np.asarray(ids, dtype=_MEMMAP_DTYPE)
            if arr.size and int(arr.min()) < 0:
                raise ValueError("token id out of int32 range (negative after cast)")
            arr.tofile(fh)
            n_tokens += int(arr.size)
            n_docs += 1
        if n_docs % 50_000 < len(chunk):
            logger.info("memmap pack: docs=%d tokens=%d", n_docs, n_tokens)
        chunk = []

    logger.info("Building memmap pack: %s → %s", path, pack_dir)
    with tmp_path.open("wb") as fh:
        for row in iter_jsonl(path):
            text = row.get(text_field, "")
            if not isinstance(text, str) or not text.strip():
                continue
            chunk.append(text.strip())
            if max_texts is not None and (n_docs + len(chunk)) >= max_texts:
                _flush(fh)
                break
            if len(chunk) >= chunk_texts:
                _flush(fh)
        else:
            _flush(fh)

    if n_docs == 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"No non-empty texts in {path} (field={text_field})")

    os.replace(tmp_path, tokens_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": _MEMMAP_PACK_VERSION,
                "source": str(path.resolve()),
                "n_tokens": n_tokens,
                "n_docs": n_docs,
                "max_texts": max_texts,
            },
            f,
            indent=2,
        )
        f.write("\n")
    logger.info("memmap pack done: docs=%d tokens=%d → %s", n_docs, n_tokens, tokens_path)
    return tokens_path, n_tokens


class MemmapPackedCausalLMDataset(Dataset):
    """Lazy packed blocks over a flat int32 token stream (memmap-backed).

    Block ``i`` is ``tokens[i*seq_length:(i+1)*seq_length]``; the trailing
    partial block is dropped (same as ``pack_token_ids``).
    """

    def __init__(self, tokens_path: Union[str, Path], n_tokens: int, seq_length: int):
        if seq_length < 2:
            raise ValueError(f"seq_length must be >= 2, got {seq_length}")
        self.tokens_path = Path(tokens_path)
        self.n_tokens = int(n_tokens)
        self.seq_length = seq_length
        self._tokens = np.memmap(
            self.tokens_path, dtype=_MEMMAP_DTYPE, mode="r", shape=(self.n_tokens,)
        )
        n_blocks = self.n_tokens // self.seq_length
        if n_blocks == 0:
            raise ValueError("MemmapPackedCausalLMDataset requires at least one full block")

    def __len__(self) -> int:
        return self.n_tokens // self.seq_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.seq_length
        ids = np.asarray(
            self._tokens[start : start + self.seq_length], dtype=np.int64
        )
        input_ids = torch.from_numpy(ids)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def build_packed_dataloader_memmap(
    path: Union[str, Path],
    tokenizer,
    seq_length: int,
    batch_size: int,
    cache_dir: Optional[Union[str, Path]] = None,
    text_field: str = DEFAULT_TEXT_FIELD,
    shuffle: bool = False,
    num_workers: int = 0,
    max_texts: Optional[int] = None,
    chunk_texts: int = 1024,
) -> DataLoader:
    """Memmap-backed twin of ``build_packed_dataloader`` (bounded RAM)."""
    tokens_path, n_tokens = build_memmap_pack(
        path,
        tokenizer,
        cache_dir=cache_dir if cache_dir is not None else default_pack_cache_dir(),
        text_field=text_field,
        max_texts=max_texts,
        chunk_texts=chunk_texts,
    )
    ds = MemmapPackedCausalLMDataset(tokens_path, n_tokens, seq_length)
    logger.info(
        "memmap packed dataset: %d blocks (seq=%d) from %d tokens",
        len(ds),
        seq_length,
        n_tokens,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def resolve_data_path(
    explicit: Optional[str],
    *candidates: str,
) -> Optional[str]:
    """Return the first existing path among *explicit* and *candidates*."""
    ordered: List[str] = []
    if explicit:
        ordered.append(explicit)
    ordered.extend(candidates)
    for p in ordered:
        if p and os.path.isfile(p):
            return p
    return None


def default_kaggle_data_candidates(split: str) -> List[str]:
    """Common Kaggle Dataset mount layouts for train/val JSONL."""
    name = f"{split}.jsonl"
    roots = [
        "/kaggle/input",
        "/kaggle/input/tetraft-fineweb-edu",
        "/kaggle/input/fineweb-edu-sample",
        "/kaggle/input/tetraft-fineweb-edu-50m",
        "/kaggle/input/tetraft-fineweb-edu-200m",
        "./data",
        "data",
    ]
    out: List[str] = []
    for root in roots:
        out.append(os.path.join(root, name))
        # Nested one level under /kaggle/input/<dataset>/
        if root == "/kaggle/input" and os.path.isdir(root):
            try:
                for sub in sorted(os.listdir(root)):
                    out.append(os.path.join(root, sub, name))
            except OSError:
                pass
    return out


# ---------------------------------------------------------------------------
# CLI: one-time sample build
# ---------------------------------------------------------------------------

def _parse_build_args(argv: Optional[Sequence[str]] = None):
    import argparse

    p = argparse.ArgumentParser(description="Build a fixed FineWeb-Edu JSONL sample")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-train-tokens", type=int, default=50_000_000)
    p.add_argument("--max-val-tokens", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-dataset", type=str, default=DEFAULT_HF_DATASET)
    p.add_argument("--hf-split", type=str, default=DEFAULT_HF_SPLIT)
    p.add_argument("--hf-revision", type=str, default=None)
    p.add_argument("--text-field", type=str, default=DEFAULT_TEXT_FIELD)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_build_args(argv)
    build_fineweb_sample(
        output_dir=args.output_dir,
        max_train_tokens=args.max_train_tokens,
        max_val_tokens=args.max_val_tokens,
        seed=args.seed,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_revision=args.hf_revision,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    main()
