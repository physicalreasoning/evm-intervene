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


def main() -> None:
    print("## Sensitivity arm (crash-exposed training)\n")
    print(intervention_arm_table("sensitivity_crash_exposed"))
    print("\n## Primary arm (chronologically sealed)\n")
    print(intervention_arm_table("primary"))
    print("\n## Scaling sweep\n")
    print(scaling_table())
    print("\n" + research_v2_tables())


if __name__ == "__main__":
    main()
