#!/usr/bin/env python3
"""Rebuild curves/ledger for a heal_kl_trust_400m paper pack.

Usage::

    python scripts/merge_run_pack.py /path/to/heal_kl_trust_400m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on path when run as scripts/merge_run_pack.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_pack import merge_run_pack  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Merge TetraFT marathon run pack curves")
    p.add_argument("run_root", type=str, help="Path to heal_kl_trust_400m pack root")
    args = p.parse_args(argv)
    out = merge_run_pack(args.run_root)
    print("Rebuilt:")
    for k, path in out.items():
        print(f"  {k}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
