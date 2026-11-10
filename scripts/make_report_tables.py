# ruff: noqa: E501
"""Print Markdown tables for README.md / FINDINGS.md directly from committed results JSON.

Every number in the docs should come from this script, not be typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import spearmanr

RES = Path("results")
FAMILIES = ("mlp_direct", "seq_transformer", "latent_jepa", "conservation_wm")


def _fmt(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{nd}f}"


def intervention_arm_table(arm_key: str) -> str:
    m = json.loads((RES / "mainnet_intervention_results.json").read_text(encoding="utf-8"))
    arm = m[arm_key]
    c = arm["counts"]
    lines = [
        f"**{arm['arm_name']}**: train={c['n_train_obs']:,}, val={c['n_val_obs']:,}, "
        f"interventions={c['n_interventions_total']:,} across {c['n_held_out_checkpoints']} held-out checkpoints "
        f"(in-tick={c['regime_counts']['regime_1_smooth']}, cliff={c['regime_counts']['regime_2_cliff']}, "
        f"cascade={c['regime_counts']['regime_3_cascade']}). 5 seeds.",
        "",
        "| Model | Best obs val std-MSE | Matched obs val std-MSE | In-tick price RMSE (bps) | Cliff price RMSE (bps) | Cliff std-MSE (matched) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fam in FAMILIES:
        r = arm["models"][fam]
        mi = r["matched_checkpoint_interventions"]
        lines.append(
            f"| `{fam}` | {_fmt(r['best_obs_val_std_mse_mean'], 4)} | {_fmt(r['matched_obs_val_std_mse_mean'], 4)} | "
            f"{_fmt(mi['regime_1_smooth']['price_rmse_bps'])} | {_fmt(mi['regime_2_cliff']['price_rmse_bps'])} | "
            f"{_fmt(mi['regime_2_cliff']['std_mse'])} |"
        )
    lines.append(f"| `copy` (obs val only) | {_fmt(arm['copy_obs_val_std_mse'], 4)} | | | | |")
    lines += ["", "| Challenger vs `conservation_wm` | Within 5% obs loss | Cliff std-MSE gap [95% CI] | Holm p |", "| :--- | :---: | :---: | ---: |"]
    for t in arm["pairwise_gate_g1"]:
        g = t["matched_cliff_gap"]
        lines.append(
            f"| `{t['challenger']}` | {'yes' if t['matched_within_5pct'] else 'no'} ({t['obs_val_rel_diff']:.2%}) | "
            f"{g['gap']:+.2f} [{g['ci_low']:+.2f}, {g['ci_high']:+.2f}] | {g['holm_p']:.4f} |"
        )
    g2 = arm["rank_correlation_gate_g2"]["spearman_obs_vs_cliff_price_bps"]
    lines += [
        "",
        f"Gate G1: **{'PASS' if arm['gates']['gate_g1_matched_separation_pass'] else 'FAIL'}**. "
        f"Gate G2 (Spearman obs loss vs cliff bps, n=20): rho = {g2['rho']:+.3f}, p = {g2['p_value']:.4f}, "
        f"95% CI [{g2['ci_low']:+.3f}, {g2['ci_high']:+.3f}].",
    ]
    return "\n".join(lines)


def scaling_table() -> str:
    s = json.loads((RES / "scaling_sweep_results.json").read_text(encoding="utf-8"))
    d = s["dataset_summary"]
    a = s["anchor_conservation_wm"]
    lines = [
        f"train={d['train_swaps']:,}, val={d['val_swaps']:,}, interventions={d['counterfactual_interventions']:,} "
        f"across {d['held_out_checkpoints']} held-out checkpoints; 3 seeds per configuration.",
        "",
        "| Tier | Model | Params | Obs val std-MSE | Cliff price RMSE (bps) | Cascade price RMSE (bps) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
        f"| anchor | `conservation_wm` | {a['params']:,} | {a['obs_val_std_mse_mean']:.4f} | {a['cliff_rmse_bps_mean']:.2f} | {a['cascade_rmse_bps_mean']:,.1f} |",
    ]
    recs = [(r["obs_val_std_mse"], r["cliff_rmse_bps"]) for r in a["runs"]]
    for tier, fams in s["scaling_tiers"].items():
        for fam, v in fams.items():
            lines.append(
                f"| {tier} | `{fam}` | {v['params']:,} | {v['obs_val_std_mse_mean']:.4f} | "
                f"{v['cliff_rmse_bps_mean']:.2f} | {v['cascade_rmse_bps_mean']:,.1f} |"
            )
            recs += [(r["obs_val_std_mse"], r["cliff_rmse_bps"]) for r in v["runs"]]
    rho, p = spearmanr([x for x, _ in recs], [y for _, y in recs])
    neural = recs[len(a["runs"]) :]
    rho_n, p_n = spearmanr([x for x, _ in neural], [y for _, y in neural])
    lines += [
        "",
        f"Spearman (obs loss vs cliff bps), all {len(recs)} trained runs incl. anchor: rho = {rho:+.3f}, p = {p:.4f}. "
        f"Neural runs only (n={len(neural)}): rho = {rho_n:+.3f}, p = {p_n:.4f}.",
    ]
    return "\n".join(lines)


def research_v2_tables() -> str:
    v2 = RES / "research_v2"
    lines: list[str] = []
    if (v2 / "ablation.json").exists():
        ab = json.loads((v2 / "ablation.json").read_text(encoding="utf-8"))
        lines += [
            "## Conservation World Model Ablations (`results/research_v2/ablation.json`)",
            "",
            "| Variant | Obs val std-MSE | In-tick RMSE (bps) | Cliff RMSE (bps) | Cliff std-MSE | Cascade RMSE (bps) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k, v in ab["variants"].items():
            m = v["mean"]
            lines.append(
                f"| `{k}` | {m['obs_val_std_mse']:.4f} | "
                f"{m['in_tick_bps']:.2f} | {m['cliff_bps']:.2f} | "
                f"{m['cliff_std_mse']:.2f} | {m['cascade_bps']:,.1f} |"
            )
        lines.append("")

    if (v2 / "coverage.json").exists():
        cov = json.loads((v2 / "coverage.json").read_text(encoding="utf-8"))
        lines += [
            "## Training Coverage Sweep (`results/research_v2/coverage.json`)",
            "",
            "| Augmented cliff fraction | Added swaps | `mlp_direct` cliff (bps) | `seq_transformer` cliff (bps) | `conservation_wm` cliff (bps) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for k, entry in cov["results"].items():
            m = entry["families"]
            lines.append(
                f"| {float(k):.0%} | +{entry['n_added']:,} | "
                f"{m['mlp_direct']['mean']['cliff_bps']:.2f} ({m['mlp_direct']['mean']['cliff_std_mse']:.2f} std-MSE) | "
                f"{m['seq_transformer']['mean']['cliff_bps']:.2f} ({m['seq_transformer']['mean']['cliff_std_mse']:.2f} std-MSE) | "
                f"{m['conservation_wm']['mean']['cliff_bps']:.2f} ({m['conservation_wm']['mean']['cliff_std_mse']:.2f} std-MSE) |"
            )
        lines.append("")

    if (v2 / "llm.json").exists():
        llm = json.loads((v2 / "llm.json").read_text(encoding="utf-8"))
        lines += [
            "## Pretrained QLoRA Foundation Models (`results/research_v2/llm.json`)",
            "",
            "| Backbone | Backbone params | Trainable params | Obs val std-MSE | Beats copy? | In-tick RMSE (bps) | Cliff RMSE (bps) | Cliff std-MSE | Cascade RMSE (bps) |",
            "| :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
        ]
        for v in llm["runs"]:
            if "error" in v:
                continue
            lines.append(
                f"| `{v['model_id']}` (4-bit QLoRA) | {v['backbone_params']:,} | {v['trainable_params']:,} | "
                f"{v['obs_val_std_mse']:.4f} | {'yes' if v['valid_beats_copy'] else 'no'} | "
                f"{v['in_tick_bps']:.2f} | {v['cliff_bps']:.2f} | {v['cliff_std_mse']:.2f} | {v['cascade_bps']:,.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def heavy_suite_tables() -> str:
    hs = RES / "heavy_suite"
    lines: list[str] = []
    if (hs / "llm_ladder.json").exists():
        ldr = json.loads((hs / "llm_ladder.json").read_text(encoding="utf-8"))
        lines += [
            "## Full Foundation-Model Scaling Ladder (`results/heavy_suite/llm_ladder.json`)",
            "",
            "| Configuration | Backbone params | Trainable params | Obs val std-MSE | In-tick (bps) | Cliff RMSE (bps) | Cliff std-MSE | Rollout H=12 (bps) | Real Flow H=50 (bps) | Planner Viol % |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        runs_iter = ldr["runs"].values() if isinstance(ldr["runs"], dict) else ldr["runs"]
        for r in runs_iter:
            if "error" in r:
                lines.append(
                    f"| `{r['model_id']}` (r={r['lora_r']}) | — | — | OOM (22 GiB L4) | — | — | — | — | — | — |"
                )
                continue
            tag = "random-init scratch" if r["random_init"] else f"pretrained (r={r['lora_r']})"
            ro12 = r["rollout_horizons"]["h12"]["cum_price_rmse_bps"]
            rf50 = r["real_flow_rollout"]["all_trajectories"]["h50"]
            pv = r["planner_overall"]["violation_rate_pct"]
            lines.append(
                f"| `{r['model_id']}` {tag} | {r['backbone_params']:,} | {r['trainable_params']:,} | "
                f"{r['obs_val_std_mse']:.4f} | {r['in_tick_bps']:.2f} | **{r['cliff_bps']:.2f}** | "
                f"{r['cliff_std_mse']:.2f} | {ro12:.2f} | "
                f"{rf50['price_rmse_bps']:.2f} ({rf50['directional_acc_pct']:.1f}%) | "
                f"{pv:.2f}% |"
            )
        lines.append("")

    if (hs / "planner.json").exists():
        pl = json.loads((hs / "planner.json").read_text(encoding="utf-8"))
        lines += [
            "## Downstream Risk-Constrained Execution Planner (`results/heavy_suite/planner.json`)",
            "",
            "| Model | Params | Mean Viol Rate | Viol @ B=25 bps | Overshoot RMSE (bps) | Sizing Rel Err | Mean Excess Slippage (USD) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k, v in pl["models"].items():
            m = v["overall_mean"]
            b25 = v["by_budget_mean"]["B_25bps"]["violation_rate_pct"]
            lines.append(
                f"| `{k}` | {v['params']:,} | {m['violation_rate_pct']:.2f}% | {b25:.2f}% | "
                f"{m['overshoot_rmse_bps']:.2f} | {m['sizing_rel_err_pct']:.2f}% | "
                f"\\${m['mean_excess_slippage_usd']:,.2f} |"
            )
        lines.append("")

    if (hs / "rollout.json").exists():
        ro = json.loads((hs / "rollout.json").read_text(encoding="utf-8"))
        lines += [
            "## 12-Step Autoregressive Counterfactual Rollouts (`results/heavy_suite/rollout.json`)",
            "",
            "| Model | Params | H=1 RMSE (bps) | H=3 RMSE (bps) | H=5 RMSE (bps) | H=8 RMSE (bps) | H=12 RMSE (bps) | H=12 p95 (bps) | H=12 Liq RMSE |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k, v in ro["models"].items():
            by_h = v["mean_horizons"]
            lines.append(
                f"| `{k}` | {v['params']:,} | {by_h['h1']['cum_price_rmse_bps']:.2f} | "
                f"{by_h['h3']['cum_price_rmse_bps']:.2f} | {by_h['h5']['cum_price_rmse_bps']:.2f} | "
                f"{by_h['h8']['cum_price_rmse_bps']:.2f} | {by_h['h12']['cum_price_rmse_bps']:.2f} | "
                f"{by_h['h12']['cum_price_p95_bps']:.2f} | {by_h['h12']['cum_liquidity_rmse']:.4f} |"
            )
        lines.append("")

    if (hs / "forecast.json").exists():
        fc = json.loads((hs / "forecast.json").read_text(encoding="utf-8"))
        p1 = fc["real_flow_rollout_models"]
        lines += [
            "## Part 1: 50-Swap Real Mainnet Order-Flow Trajectory Forecasting (`results/heavy_suite/forecast.json`)",
            "",
            "| Model | H=1 RMSE (bps) | H=10 RMSE (bps) | H=20 RMSE (bps) | H=35 RMSE (bps) | H=50 RMSE (bps) | H=50 Dir Acc | H=50 Liq RMSE |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k, v in p1.items():
            bh = v["all_trajectories_mean"]
            lines.append(
                f"| `{k}` | {bh['h1']['price_rmse_bps']:.2f} | {bh['h10']['price_rmse_bps']:.2f} | "
                f"{bh['h20']['price_rmse_bps']:.2f} | {bh['h35']['price_rmse_bps']:.2f} | "
                f"{bh['h50']['price_rmse_bps']:.2f} | {bh['h50']['directional_acc_pct']:.2f}% | "
                f"{bh['h50']['liquidity_rmse']:.4f} |"
            )
        p2 = fc["unconditional_volatility_forecast"]
        lines += [
            "",
            "## Part 2: Unconditional 20-Swap Volatility, Excursion & Liquidity-Depletion Forecasting (`results/heavy_suite/forecast.json`)",
            "",
            "| Forecaster | Vol RMSE (bps) | Spearman Vol | Excursion RMSE (bps) | Spearman Excursion | Liq-Depletion RMSE | Spearman Liq-Depletion |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k, m in [
            ("ewma_persistence", p2["ewma_persistence"]),
            ("history_only_forecaster", p2["history_only_forecaster_mean"]),
            ("spatial_ticklens_forecaster", p2["spatial_ticklens_forecaster_mean"]),
        ]:
            lines.append(
                f"| `{k}` | {m['vol_rmse_bps']:.2f} | {m['spearman_vol']:+.4f} | "
                f"{m['excursion_rmse_bps']:.2f} | {m['spearman_excursion']:+.4f} | "
                f"{m['liq_depletion_rmse']:.4f} | {m['spearman_liq_depletion']:+.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def price_forecast_tables() -> str:
    p = RES / "forecast_suite" / "eth_price_forecast_results.json"
    if not p.exists():
        p = RES / "eth_price_forecast_results.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    ds = d["dataset_summary"]
    lines = [
        f"## Multi-Horizon ETH Price & Order-Flow Forecasting (`{p.as_posix()}`)",
        "",
        f"train={ds['train_samples']:,}, val={ds['val_samples']:,}, test={ds['test_samples']:,} "
        f"(Aug 5 crash test={ds['test_crash_aug5_samples']:,}, Oct test={ds['test_autumn_oct_samples']:,}).",
        "",
        "| Model | H=10 Price MAE ($) | H=25 Price MAE ($) | H=50 Price MAE ($) | H=10 Dir Acc (Active) | H=25 Dir Acc (Active) | H=50 Flow Sign Acc | Exec Timing Savings ($ USD) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mk, mv in d["models"].items():
        bh = mv["by_horizon"]
        bt_key = "trading_backtest_h50" if "trading_backtest_h50" in mv else "trading_backtest_h25"
        bt = mv[bt_key]
        lines.append(
            f"| `{mk}` | \\${bh['h10']['price_usd_mae']:.2f} | \\${bh['h25']['price_usd_mae']:.2f} | "
            f"\\${bh['h50']['price_usd_mae']:.2f} | {bh['h10']['directional_acc_active_pct']:.2f}% | "
            f"{bh['h25']['directional_acc_active_pct']:.2f}% | {bh['h50']['orderflow_sign_active_pct']:.2f}% | "
            f"\\${bt['execution_timing_total_savings_usd']:+,.2f} ({bt['execution_timing_mean_savings_bps']:+.2f} bps) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(price_forecast_tables())
    print("## Sensitivity arm (crash-exposed training)\n")
    print(intervention_arm_table("sensitivity_crash_exposed"))
    print("\n## Primary arm (chronologically sealed)\n")
    print(intervention_arm_table("primary"))
    print("\n## Scaling sweep\n")
    print(scaling_table())
    print("\n" + research_v2_tables())
    print("\n" + heavy_suite_tables())


if __name__ == "__main__":
    main()
