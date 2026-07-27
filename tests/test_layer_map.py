import math

import torch
import torch.nn as nn

from quantize import (
    QuantizedLinear,
    aggregate_layer_map,
    parse_block_index,
    parse_module_role,
    per_module_quant_stats,
    quaternary_quant,
    compute_scale,
    suggest_d0_next,
)


class TestParseHelpers:
    def test_role_leaf(self):
        assert parse_module_role("model.layers.3.self_attn.q_proj") == "q_proj"
        assert parse_module_role("model.layers.0.mlp.gate_proj") == "gate_proj"
        assert parse_module_role("model.layers.1.mlp.down_proj") == "down_proj"
        assert parse_module_role("foo.bar") == "other"

    def test_block_index(self):
        assert parse_block_index("model.layers.12.mlp.up_proj") == 12
        assert parse_block_index("model.layers.0.self_attn.o_proj") == 0
        assert parse_block_index("lm_head") is None


class TestPerModuleStats:
    def test_known_error_and_bins(self):
        torch.manual_seed(0)
        layer = QuantizedLinear(8, 4, bias=False, c=0.25, scale_mode="absmean_channel")
        # Force weights into clear segments
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.9, 0.1, 0.05, -0.05, -0.1, -0.9, -1.0],
                        [0.5, 0.4, 0.3, 0.2, -0.2, -0.3, -0.4, -0.5],
                        [0.0, 0.01, -0.01, 0.02, -0.02, 0.0, 0.0, 0.0],
                        [2.0, -2.0, 1.5, -1.5, 0.8, -0.8, 0.1, -0.1],
                    ],
                    dtype=torch.float32,
                )
            )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                # fake nested name via ModuleDict path
                self.layers = nn.ModuleList([nn.Module()])
                self.layers[0].mlp = nn.Module()
                self.layers[0].mlp.up_proj = layer

        # named_modules will be layers.0.mlp.up_proj if we structure correctly
        m = nn.Module()
        m.layers = nn.ModuleList([nn.Module()])
        m.layers[0].mlp = nn.Module()
        m.layers[0].mlp.up_proj = layer

        rows = per_module_quant_stats(m)
        assert len(rows) == 1
        r = rows[0]
        assert r["role"] == "up_proj"
        assert r["block"] == 0
        assert r["n_params"] == 32
        assert r["mse"] >= 0.0
        assert r["rel_l2"] >= 0.0
        # bins should sum ~1
        s = r["frac_-1"] + r["frac_-c"] + r["frac_+c"] + r["frac_+1"]
        assert abs(s - 1.0) < 1e-5

        w = layer.weight.detach().float()
        gamma = compute_scale(w, layer.scale_mode).float()
        w_q = quaternary_quant(w, layer.c, gamma)
        exp_mse = float((w - w_q).pow(2).mean().item())
        assert math.isclose(r["mse"], exp_mse, rel_tol=1e-5, abs_tol=1e-6)

    def test_aggregate_and_suggest(self):
        rows = [
            {
                "name": "layers.0.self_attn.q_proj",
                "role": "q_proj",
                "block": 0,
                "n_params": 100,
                "rel_l2": 0.5,
                "mean_abs_err_over_gamma": 0.4,
                "mse": 0.1,
            },
            {
                "name": "layers.0.mlp.down_proj",
                "role": "down_proj",
                "block": 0,
                "n_params": 100,
                "rel_l2": 0.1,
                "mean_abs_err_over_gamma": 0.1,
                "mse": 0.01,
            },
            {
                "name": "layers.1.self_attn.q_proj",
                "role": "q_proj",
                "block": 1,
                "n_params": 100,
                "rel_l2": 0.45,
                "mean_abs_err_over_gamma": 0.35,
                "mse": 0.09,
            },
        ]
        summary = aggregate_layer_map(rows, top_k=2, sort_key="rel_l2")
        assert summary["n_modules"] == 3
        assert "q_proj" in summary["by_role"]
        assert summary["by_role"]["q_proj"]["rel_l2_wmean"] > summary["by_role"]["down_proj"][
            "rel_l2_wmean"
        ]
        assert len(summary["top_modules"]) == 2
        assert summary["top_modules"][0]["name"].endswith("q_proj")

        sug = suggest_d0_next(summary)
        assert "suggest" in sug
        assert sug["suggest"] in {
            "D1_heal_kl_100m",
            "D2_scout_skip_qo",
            "unclear",
        }

    def test_suggest_fp_mask_lift(self):
        rows = [
            {
                "name": f"layers.{i}.mlp.down_proj",
                "role": "down_proj",
                "block": i,
                "n_params": 10,
                "rel_l2": 0.2,
                "mean_abs_err_over_gamma": 0.2,
                "mse": 0.01,
            }
            for i in range(4)
        ]
        summary = aggregate_layer_map(rows, top_k=4)
        sug = suggest_d0_next(summary, fp_mask_ppl=28.0, baseline_ppl=34.0)
        assert sug["suggest"] == "D2_scout_skip_qo"
        assert any("fp_mask" in r for r in sug["reasons"])

    def test_empty(self):
        assert per_module_quant_stats(nn.Linear(4, 4)) == []
        s = aggregate_layer_map([])
        assert s["n_modules"] == 0
        assert suggest_d0_next(s)["suggest"] == "unclear"
