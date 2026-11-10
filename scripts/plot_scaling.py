# ruff: noqa: E501
"""Render SVG figures in assets/ directly from committed results JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path

COLORS = {
    "mlp_direct": "#2563eb",
    "seq_transformer": "#7c3aed",
    "latent_jepa": "#d97706",
    "qwen_qlora": "#dc2626",
}
LABELS = {
    "mlp_direct": "Direct MLP (247k - 170M)",
    "seq_transformer": "Cross-attention Transformer (184k - 70M)",
    "latent_jepa": "Latent JEPA (340k - 127M)",
    "qwen_qlora": "Qwen2.5 LoRA/QLoRA (0.5B - 7.1B)",
}


def plot_scaling_svg() -> None:
    s = json.loads(Path("results/scaling_sweep_results.json").read_text(encoding="utf-8"))
    ldr_path = Path("results/heavy_suite/llm_ladder.json")
    v2_path = Path("results/research_v2/llm.json")
    if ldr_path.exists():
        raw_runs = json.loads(ldr_path.read_text(encoding="utf-8"))["runs"]
        llm_runs = list(raw_runs.values()) if isinstance(raw_runs, dict) else raw_runs
    elif v2_path.exists():
        llm_runs = json.loads(v2_path.read_text(encoding="utf-8"))["runs"]
    else:
        llm_runs = []

    anchor = s["anchor_conservation_wm"]
    series: dict[str, list[tuple[int, float, float]]] = {f: [] for f in COLORS}
    for fams in s["scaling_tiers"].values():
        for fam, v in fams.items():
            series[fam].append((int(v["params"]), float(v["cliff_rmse_bps_mean"]), float(v["obs_val_std_mse_mean"])))

    rand_pts: list[tuple[int, float, float]] = []
    for r in llm_runs:
        if r.get("valid_beats_copy"):
            pt = (int(r["backbone_params"]), float(r["cliff_bps"]), float(r["obs_val_std_mse"]))
            if r.get("random_init"):
                rand_pts.append(pt)
            else:
                series["qwen_qlora"].append(pt)

    W, H = 1120, 480
    panels = [
        (75, "A. Observational validation loss (std-MSE)", 2, 0.0, 0.60),
        (615, "B. Held-out counterfactual cliff RMSE (bps)", 1, 0.0, 230.0),
    ]
    pw, ph, top = 430, 330, 82
    xmin, xmax = math.log10(1.5e3), math.log10(1.2e10)

    def mx(x0: int, p: float) -> float:
        return x0 + (math.log10(p) - xmin) / (xmax - xmin) * pw

    def my(v: float, lo: float, hi: float) -> float:
        return top + ph - (v - lo) / (hi - lo) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<style>text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1f2937}</style>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="75" y="34" font-size="18" font-weight="700">Parameter scaling (2.5k to 7.1B): observational fit improves, cliff error plateaus</text>',
        f'<text x="75" y="56" font-size="12" fill="#4b5563">{s["dataset_summary"]["counterfactual_interventions"]:,} held-out '
        "counterfactual interventions across 62 checkpoints · scratch models (3 seeds) + Qwen2.5 0.5B/1.5B/3B/7B LoRA/QLoRA</text>",
    ]
    for x0, title, idx, lo, hi in panels:
        out.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
        out.append(f'<text x="{x0}" y="{top - 10}" font-size="13" font-weight="600">{title}</text>')
        for e in range(4, 11):
            xp = mx(x0, 10**e)
            out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
            out.append(
                f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" fill="#6b7280">10^{e}</text>'
            )
        for k in range(5):
            v = lo + (hi - lo) * k / 4
            yp = my(v, lo, hi)
            out.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x0 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
            out.append(
                f'<text x="{x0 - 8}" y="{yp + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{v:g}</text>'
            )
        out.append(
            f'<text x="{x0 + pw / 2}" y="{top + ph + 40}" font-size="12" text-anchor="middle">Total parameters (log scale)</text>'
        )
        av = anchor["cliff_rmse_bps_mean"] if idx == 1 else anchor["obs_val_std_mse_mean"]
        out.append(
            f'<line x1="{x0}" y1="{my(av, lo, hi):.1f}" x2="{x0 + pw}" y2="{my(av, lo, hi):.1f}" '
            'stroke="#059669" stroke-width="2.2" stroke-dasharray="6 4"/>'
        )
        out.append(
            f'<circle cx="{mx(x0, anchor["params"]):.1f}" cy="{my(av, lo, hi):.1f}" r="5.5" fill="#059669"/>'
        )
        for fam, pts in series.items():
            if not pts:
                continue
            pts = sorted(pts)
            path = " ".join(f"{mx(x0, p):.1f},{my(c if idx == 1 else o, lo, hi):.1f}" for p, c, o in pts)
            out.append(f'<polyline points="{path}" fill="none" stroke="{COLORS[fam]}" stroke-width="2.4"/>')
            for p, c, o in pts:
                val = c if idx == 1 else o
                r_px = 5.0 if fam == "qwen_qlora" else 4.0
                out.append(f'<circle cx="{mx(x0, p):.1f}" cy="{my(val, lo, hi):.1f}" r="{r_px}" fill="{COLORS[fam]}"/>')
        for p, c, o in rand_pts:
            val = c if idx == 1 else o
            out.append(
                f'<circle cx="{mx(x0, p):.1f}" cy="{my(val, lo, hi):.1f}" r="5.5" fill="#ffffff" stroke="#dc2626" stroke-width="2.2"/>'
            )

    ly = H - 20
    lx = 60
    for fam in COLORS:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="14" height="4" fill="{COLORS[fam]}"/>')
        out.append(f'<text x="{lx + 18}" y="{ly - 3}" font-size="11">{LABELS[fam]}</text>')
        lx += 238
    out.append(f'<rect x="{lx}" y="{ly - 9}" width="14" height="4" fill="#059669"/>')
    out.append(f'<text x="{lx + 18}" y="{ly - 3}" font-size="11">conservation_wm (2.5k)</text>')
    out.append("</svg>")
    Path("assets/scaling_sweep.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote assets/scaling_sweep.svg")


def plot_ablation_and_coverage_svg() -> None:
    ab = json.loads(Path("results/research_v2/ablation.json").read_text(encoding="utf-8"))["variants"]
    cov = json.loads(Path("results/research_v2/coverage.json").read_text(encoding="utf-8"))["results"]

    W, H = 1120, 460
    pw, ph, top = 430, 310, 82
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<style>text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1f2937}</style>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="75" y="34" font-size="18" font-weight="700">Structural ablations and training-support coverage sweep</text>',
        '<text x="75" y="56" font-size="12" fill="#4b5563">Mean over 3 seeds (45 epochs) on 1,145 held-out Regime II cliff interventions</text>',
    ]

    # Panel A: Ablation horizontal bars
    x0 = 75
    out.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(
        f'<text x="{x0}" y="{top - 10}" font-size="13" font-weight="600">A. conservation_wm structural ablations (cliff RMSE, bps)</text>'
    )
    variants = [
        ("no_bin_net", "Raw TickLens profile (no bin_net)", "#059669"),
        ("no_liq_coupling", "No Aave coupling (beta_liq = 0)", "#0d9488"),
        ("none", "Full conservation_wm (+ bin_net)", "#2563eb"),
        ("uniform_profile", "Uniform profile (erase TickLens)", "#dc2626"),
    ]
    max_bps = 160.0
    for idx, (key, label, col) in enumerate(variants):
        v_bps = float(ab[key]["mean"]["cliff_bps"])
        obs_v = float(ab[key]["mean"]["obs_val_std_mse"])
        by = top + 28 + idx * 68
        bw = (v_bps / max_bps) * (pw - 130)
        out.append(f'<text x="{x0 + 14}" y="{by}" font-size="12" font-weight="600">{label}</text>')
        out.append(f'<rect x="{x0 + 14}" y="{by + 8}" width="{bw:.1f}" height="22" rx="4" fill="{col}"/>')
        out.append(
            f'<text x="{x0 + 22 + bw:.1f}" y="{by + 24}" font-size="11.5" fill="#111827">{v_bps:.1f} bps (obs MSE {obs_v:.2f})</text>'
        )

    # Panel B: Coverage sweep lines
    x1 = 615
    out.append(f'<rect x="{x1}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(
        f'<text x="{x1}" y="{top - 10}" font-size="13" font-weight="600">B. Cliff RMSE vs. fraction of simulated cliff swaps in training</text>'
    )
    fracs = ["0.000", "0.010", "0.050", "0.150"]
    x_positions = [x1 + 35 + i * (pw - 70) / 3.0 for i in range(4)]
    for xp, fk in zip(x_positions, fracs, strict=True):
        pct_lbl = f"{float(fk) * 100:.0f}%"
        n_add = cov[fk]["n_added"]
        out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
        out.append(
            f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" fill="#6b7280">{pct_lbl} (+{n_add})</text>'
        )
    for v in [0, 50, 100, 150, 200]:
        yp = top + ph - (v / 220.0) * ph
        out.append(f'<line x1="{x1}" y1="{yp:.1f}" x2="{x1 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x1 - 8}" y="{yp + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{v}</text>')
    out.append(
        f'<text x="{x1 + pw / 2}" y="{top + ph + 40}" font-size="12" text-anchor="middle">Simulated cliff-crossing share of training set</text>'
    )

    fam_meta = [
        ("mlp_direct", "mlp_direct (3.4M)", "#2563eb"),
        ("seq_transformer", "seq_transformer (1.8M)", "#7c3aed"),
        ("conservation_wm", "conservation_wm (2.5k)", "#059669"),
    ]
    for fam, _lbl, col in fam_meta:
        pts = []
        for xp, fk in zip(x_positions, fracs, strict=True):
            val = float(cov[fk]["families"][fam]["mean"]["cliff_bps"])
            yp = top + ph - (val / 220.0) * ph
            pts.append((xp, yp, val))
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for px, py, val in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{col}"/>')
            out.append(f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" font-size="10.5" fill="{col}">{val:.1f}</text>')

    lx = x1
    ly = H - 18
    for _, lbl, col in fam_meta:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="14" height="4" fill="{col}"/>')
        out.append(f'<text x="{lx + 18}" y="{ly - 3}" font-size="11">{lbl}</text>')
        lx += 150

    out.append("</svg>")
    Path("assets/ablation_and_coverage.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote assets/ablation_and_coverage.svg")


def plot_heavy_suite_svg() -> None:
    hs = Path("results/heavy_suite")
    if not (hs / "rollout.json").exists():
        return
    ro = json.loads((hs / "rollout.json").read_text(encoding="utf-8"))["models"]
    fc = json.loads((hs / "forecast.json").read_text(encoding="utf-8"))["real_flow_rollout_models"]
    pl = json.loads((hs / "planner.json").read_text(encoding="utf-8"))["models"]

    W, H = 1220, 480
    pw, ph, top = 335, 310, 86
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<style>text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1f2937}</style>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="65" y="34" font-size="18" font-weight="700">Multi-step rollouts, 50-swap real Mainnet forecasting, and risk-constrained order planning</text>',
        '<text x="65" y="56" font-size="12" fill="#4b5563">Mean over 3 seeds · 248 12-step stress rollouts · 250 50-swap real Mainnet trajectories · 620 risk-constrained sizing tasks</text>',
    ]

    model_specs = [
        ("conservation_wm_none", "conservation_wm (2.5k)", "#059669"),
        ("transformer_M_1.5M", "seq_transformer M (1.5M)", "#7c3aed"),
        ("jepa_M_1.5M", "latent_jepa M (1.5M)", "#d97706"),
        ("mlp_M_1.5M", "mlp_direct M (1.5M)", "#2563eb"),
    ]

    # Panel A: 12-step rollout price RMSE
    x0 = 65
    out.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(f'<text x="{x0}" y="{top - 10}" font-size="12.5" font-weight="600">A. 12-step counterfactual rollout RMSE (bps)</text>')
    horizons_ro = [1, 2, 3, 5, 8, 12]
    for h in horizons_ro:
        xp = x0 + 24 + (h - 1) / 11.0 * (pw - 48)
        out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" fill="#6b7280">H={h}</text>')
    for v in [0, 200, 400, 600, 800]:
        yp = top + ph - (v / 800.0) * ph
        out.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x0 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x0 - 6}" y="{yp + 4:.1f}" font-size="10.5" text-anchor="end" fill="#6b7280">{v}</text>')
    out.append(f'<text x="{x0 + pw / 2}" y="{top + ph + 38}" font-size="11.5" text-anchor="middle">Autoregressive stress rollout step</text>')
    for mk, _, col in model_specs:
        pts = []
        for h in horizons_ro:
            xp = x0 + 24 + (h - 1) / 11.0 * (pw - 48)
            val = float(ro[mk]["mean_horizons"][f"h{h}"]["cum_price_rmse_bps"])
            yp = top + ph - (min(val, 800.0) / 800.0) * ph
            pts.append((xp, yp))
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for px, py in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{col}"/>')

    # Panel B: 50-swap real Mainnet order-flow forecasting
    x1 = 455
    out.append(f'<rect x="{x1}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(f'<text x="{x1}" y="{top - 10}" font-size="12.5" font-weight="600">B. 50-swap real Mainnet forecast RMSE (bps)</text>')
    horizons_fc = [1, 5, 10, 20, 35, 50]
    for h in horizons_fc:
        xp = x1 + 24 + (h - 1) / 49.0 * (pw - 48)
        out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" fill="#6b7280">{h}</text>')
    for v in [0, 250, 500, 750, 1000]:
        yp = top + ph - (v / 1000.0) * ph
        out.append(f'<line x1="{x1}" y1="{yp:.1f}" x2="{x1 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x1 - 6}" y="{yp + 4:.1f}" font-size="10.5" text-anchor="end" fill="#6b7280">{v}</text>')
    out.append(f'<text x="{x1 + pw / 2}" y="{top + ph + 38}" font-size="11.5" text-anchor="middle">Forward real historical swaps (open-loop)</text>')
    for mk, _, col in model_specs:
        pts = []
        for h in horizons_fc:
            xp = x1 + 24 + (h - 1) / 49.0 * (pw - 48)
            val = float(fc[mk]["all_trajectories_mean"][f"h{h}"]["price_rmse_bps"])
            yp = top + ph - (min(val, 1000.0) / 1000.0) * ph
            pts.append((xp, yp))
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for px, py in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{col}"/>')

    # Panel C: Planner budget violation rate & excess slippage bars
    x2 = 845
    out.append(f'<rect x="{x2}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(f'<text x="{x2}" y="{top - 10}" font-size="12.5" font-weight="600">C. Execution planner budget violation rate</text>')
    budgets = ["25", "50", "100", "200", "400"]
    for i, b in enumerate(budgets):
        xp = x2 + 30 + i * (pw - 60) / 4.0
        out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" fill="#6b7280">{b} bps</text>')
    for v in [0, 15, 30, 45, 60]:
        yp = top + ph - (v / 60.0) * ph
        out.append(f'<line x1="{x2}" y1="{yp:.1f}" x2="{x2 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x2 - 6}" y="{yp + 4:.1f}" font-size="10.5" text-anchor="end" fill="#6b7280">{v}%</text>')
    out.append(f'<text x="{x2 + pw / 2}" y="{top + ph + 38}" font-size="11.5" text-anchor="middle">Target price-impact budget B</text>')
    for mk, _, col in model_specs:
        pts = []
        for i, b in enumerate(budgets):
            xp = x2 + 30 + i * (pw - 60) / 4.0
            v_pct = float(pl[mk]["by_budget_mean"][f"B_{b}bps"]["violation_rate_pct"])
            yp = top + ph - (min(v_pct, 60.0) / 60.0) * ph
            pts.append((xp, yp))
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for px, py in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{col}"/>')

    lx = 65
    ly = H - 16
    for _, lbl, col in model_specs:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="14" height="4" fill="{col}"/>')
        out.append(f'<text x="{lx + 18}" y="{ly - 3}" font-size="11.5">{lbl}</text>')
        lx += 265

    out.append("</svg>")
    Path("assets/heavy_suite_summary.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote assets/heavy_suite_summary.svg")


if __name__ == "__main__":
    plot_scaling_svg()
    plot_ablation_and_coverage_svg()
    plot_heavy_suite_svg()
