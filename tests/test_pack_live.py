"""Optional live pack parity against a real checkpoint + HF model.

Run::

    pytest tests/test_pack_live.py -v -m live
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

_DEFAULT_WEIGHTS = (
    "/mnt/storage/tetraft-systems/ckpts/heal_kl_trust_400m_S04_weights.pt"
)
_DEFAULT_PACKED = (
    "/mnt/storage/tetraft-systems/packed/heal_kl_trust_400m_S04_packed.pt"
)


def _skip_no_torch():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def weights_path():
    p = Path(os.environ.get("TETRAFT_WEIGHTS", _DEFAULT_WEIGHTS))
    if not p.is_file():
        pytest.skip(f"weights-only ckpt missing: {p}")
    return p


@pytest.fixture(scope="module")
def packed_path():
    p = Path(os.environ.get("TETRAFT_PACKED", _DEFAULT_PACKED))
    return p


def test_oracle_vs_packed_logits(weights_path, packed_path, tmp_path):
    _skip_no_torch()
    import gc

    import torch
    from transformers import AutoModelForCausalLM

    from config import QAFTConfig, apply_smoke_preset
    from model import replace_from_config, set_quant_lambda
    from pack import (
        export_packed_state,
        load_packed,
        materialize_packed_model,
        save_packed,
    )

    config = QAFTConfig()
    apply_smoke_preset(config, "heal_kl_trust_400m")

    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    except Exception as e:
        pytest.skip(f"could not load weights: {e}")

    if "model_state_dict" not in ckpt:
        pytest.skip("no model_state_dict")

    cfg = ckpt.get("config")
    if cfg is not None:
        for k in (
            "quaternary_c",
            "scale_mode",
            "ste_mode",
            "skip_linear_attn",
            "model_name",
            "trust_softness",
        ):
            v = cfg.get(k) if isinstance(cfg, dict) else getattr(cfg, k, None)
            if v is not None:
                setattr(config, k, v)

    try:
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        ).to("cpu")
    except Exception as e:
        pytest.skip(f"HF model load failed (OOM or missing): {e}")

    base.config.use_cache = False
    replace_from_config(base, config)
    base.load_state_dict(ckpt["model_state_dict"], strict=False)
    set_quant_lambda(base, 1.0)
    del ckpt
    gc.collect()

    if not packed_path.is_file():
        packed = export_packed_state(
            base,
            c=float(config.quaternary_c),
            scale_mode=str(config.scale_mode),
            model_name=str(config.model_name),
        )
        out = tmp_path / "live_packed.pt"
        save_packed(out, packed)
        packed_path = out

    # Oracle forward
    torch.manual_seed(0)
    input_ids = torch.randint(10, 1000, (1, 32), dtype=torch.long)
    base.eval()
    with torch.no_grad():
        logits_oracle = base(input_ids=input_ids).logits.float()

    # Packed model from fresh base
    try:
        packed_base = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        ).to("cpu")
    except Exception as e:
        pytest.skip(f"second HF load failed: {e}")

    packed_base.config.use_cache = False
    # Need Linear topology before materialize — replace first then overwrite quant
    replace_from_config(packed_base, config)
    blob = load_packed(packed_path)
    materialize_packed_model(packed_base, blob, strict_residual=True)
    packed_base.eval()
    with torch.no_grad():
        logits_p = packed_base(input_ids=input_ids).logits.float()

    assert torch.allclose(logits_p, logits_oracle, atol=1e-2, rtol=1e-2)
