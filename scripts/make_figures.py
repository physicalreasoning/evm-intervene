# ruff: noqa: E501
"""Generate editorial SVG figures in `assets/` directly from committed `results/` JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PALETTE = {
    "conservation_wm": ("#0b6d73", "Conservation WM (learned)"),
    "mlp_direct": ("#b0522f", "Direct MLP"),
    "seq_transformer": ("#dda12c", "Sequence Transformer"),
    "latent_jepa": ("#5c4d7d", "Latent JEPA + Probe"),
    "single_tick_xyk": ("#627078", "Single-Tick x*y=k"),
    "amm_only_exact": ("#2f7a4c", "AMM-Only Exact (no Aave)"),
}


def _render_regime_bars_svg(primary: dict[str, Any]) -> str:
    regimes = [
        ("obs_val", "Observational Val (Routine Swaps)"),
        ("regime_1_smooth", "Regime I: In-Tick Convex"),
        ("regime_2_cliff", "Regime II: Tick-Cliff Crossing"),
        ("regime_3_cascade", "Regime III: Aave Cascade"),
    ]
    models_order = [
        "conservation_wm",
        "mlp_direct",
        "seq_transformer",
        "latent_jepa",
        "single_tick_xyk",
    ]

    # Collect price RMSE (bps) for each regime x model.
    values: dict[str, dict[str, float]] = {r_key: {} for r_key, _ in regimes}
    for m in models_order:
        if m in primary["models"]:
            m_obj = primary["models"][m]
            values["obs_val"][m] = float(m_obj["matched_obs_val_price_rmse_bps"])
            ints = m_obj["matched_checkpoint_interventions"]
            for r_key, _ in regimes[1:]:
                values[r_key][m] = float(ints[r_key]["price_rmse_bps"])
        else:
            b_obj = primary["baselines"][m]
            values["obs_val"][m] = float(b_obj["obs_val_price_rmse_bps"] or 0.0)
            ints = b_obj["interventions"]
            for r_key, _ in regimes[1:]:
                values[r_key][m] = float(ints[r_key]["price_rmse_bps"])

    # Use log10(1 + bps) bar scaling or per-panel linear scaling so both 2 bps obs and 900 bps cascade read crisply.
    width, height = 1100, 620
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #17242b; }",
        ".title { font-family: Georgia, serif; font-size: 28px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #627078; }",
        ".panel-title { font-size: 15px; font-weight: 700; }",
        ".label { font-size: 12.5px; font-weight: 600; }",
        ".value { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#f7f4eb" rx="24"/>',
        '<text x="48" y="52" class="title">Matched on Routine Swaps, Divergent Across Cliffs</text>',
        '<text x="48" y="78" class="subtitle">Log-price RMSE in basis points (bps, lower is better) · 5-seed matched observational checkpoints on Ethereum Mainnet</text>',
    ]

    panel_positions = [
        (48, 108),
        (566, 108),
        (48, 342),
        (566, 342),
    ]
    pw, ph = 486, 210

    for (r_key, r_title), (px, py) in zip(regimes, panel_positions, strict=True):
        lines.append(
            f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="16" fill="#efeadd" stroke="#ded8c8" stroke-width="1.5"/>'
        )
        lines.append(f'<text x="{px + 20}" y="{py + 30}" class="panel-title">{r_title}</text>')

        max_val = max(max(values[r_key].values()), 1e-3)
        bar_max_w = 225.0

        for idx, m in enumerate(models_order):
            val = values[r_key][m]
            color, label = PALETTE[m]
            by = py + 50 + idx * 30
            bw = max(2.5, (val / max_val) * bar_max_w) if val > 0 else 2.0
            lines.append(f'<text x="{px + 20}" y="{by + 14}" class="label">{label}</text>')
            lines.append(
                f'<rect x="{px + 205}" y="{by}" width="{bar_max_w}" height="18" rx="9" fill="#e2ddd0"/>'
            )
            lines.append(
                f'<rect x="{px + 205}" y="{by}" width="{bw:.2f}" height="18" rx="9" fill="{color}"/>'
            )
            val_txt = f"{val:.2f} bps" if val < 10 else f"{val:.1f} bps"
            lines.append(
                f'<text x="{px + 205 + bar_max_w + 10}" y="{by + 14}" class="value">{val_txt}</text>'
            )

    g1 = primary["gates"]["gate_g1_matched_separation_pass"]
    rho = primary["rank_correlation_gate_g2"]["spearman_obs_vs_cliff_std_mse"]["rho"]
    status_str = "PASS" if g1 else "FAIL"
    lines.append(
        '<rect x="48" y="566" width="1004" height="38" rx="12" fill="#dcefed" stroke="#0b6d73" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="68" y="590" class="label" fill="#0b6d73">Gate G1 (5%-Matched Separation): {status_str} · Gate G2 Spearman rho(L_obs, E_cliff) = {rho:+.3f}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_scatter_svg(primary: dict[str, Any]) -> str:
    seeds = primary["seed_checkpoints"]
    width, height = 1100, 520

    xs = [float(r["best_obs_val_loss"]) for r in seeds]
    ys = [math.log10(max(float(r["best_cliff_std_mse"]), 1e-2)) for r in seeds]

    x_min, x_max = min(xs) * 0.92, max(xs) * 1.06
    y_min, y_max = min(ys) - 0.18, max(ys) + 0.18

    plot_x0, plot_y0 = 110.0, 110.0
    plot_w, plot_h = 660.0, 330.0

    def map_x(v: float) -> float:
        return plot_x0 + ((v - x_min) / max(x_max - x_min, 1e-9)) * plot_w

    def map_y(v: float) -> float:
        return plot_y0 + plot_h - ((v - y_min) / max(y_max - y_min, 1e-9)) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #17242b; }",
        ".title { font-family: Georgia, serif; font-size: 28px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #627078; }",
        ".axis { font-size: 13px; font-weight: 600; }",
        ".tick { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11.5px; fill: #627078; }",
        ".legend { font-size: 13px; font-weight: 600; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#f7f4eb" rx="24"/>',
        '<text x="52" y="54" class="title">Observational Validation Loss vs. Counterfactual Cliff Error</text>',
        '<text x="52" y="80" class="subtitle">20 trained world models (4 architectures x 5 seeds) · Lower observational loss does not imply lower intervention error</text>',
        f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" rx="12" fill="#efeadd" stroke="#cdc6b4" stroke-width="1.5"/>',
    ]

    # Gridlines & ticks.
    for i in range(5):
        xv = x_min + (x_max - x_min) * (i / 4.0)
        xp = map_x(xv)
        lines.append(
            f'<line x1="{xp:.1f}" y1="{plot_y0}" x2="{xp:.1f}" y2="{plot_y0 + plot_h}" stroke="#ded8c8" stroke-dasharray="4 4"/>'
        )
        lines.append(
            f'<text x="{xp:.1f}" y="{plot_y0 + plot_h + 22}" text-anchor="middle" class="tick">{xv:.3f}</text>'
        )

    for i in range(5):
        yv = y_min + (y_max - y_min) * (i / 4.0)
        yp = map_y(yv)
        raw_mse = 10.0**yv
        lines.append(
            f'<line x1="{plot_x0}" y1="{yp:.1f}" x2="{plot_x0 + plot_w}" y2="{yp:.1f}" stroke="#ded8c8" stroke-dasharray="4 4"/>'
        )
        lines.append(
            f'<text x="{plot_x0 - 12}" y="{yp + 4:.1f}" text-anchor="end" class="tick">{raw_mse:.1f}</text>'
        )

    lines.append(
        f'<text x="{plot_x0 + plot_w / 2:.1f}" y="{plot_y0 + plot_h + 52}" text-anchor="middle" class="axis">Observational Validation Standardized MSE (L_obs)</text>'
    )
    lines.append(
        f'<text x="34" y="{plot_y0 + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 34 {plot_y0 + plot_h / 2:.1f})" class="axis">Regime II+III Intervention Error (log scale)</text>'
    )

    for r in seeds:
        fam = r["family"]
        color, _ = PALETTE[fam]
        cx = map_x(float(r["best_obs_val_loss"]))
        cy = map_y(math.log10(max(float(r["best_cliff_std_mse"]), 1e-2)))
        lines.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7.5" fill="{color}" fill-opacity="0.88" stroke="#17242b" stroke-width="1.2"/>'
        )

    # Legend box on right.
    lx, ly = 800, 130
    lines.append(
        f'<rect x="{lx}" y="{ly}" width="252" height="220" rx="14" fill="#efeadd" stroke="#cdc6b4" stroke-width="1.5"/>'
    )
    lines.append(f'<text x="{lx + 20}" y="{ly + 32}" class="axis">World-Model Families</text>')
    for idx, fam in enumerate(("conservation_wm", "mlp_direct", "seq_transformer", "latent_jepa")):
        color, label = PALETTE[fam]
        iy = ly + 64 + idx * 34
        lines.append(
            f'<circle cx="{lx + 28}" cy="{iy - 5}" r="7" fill="{color}" stroke="#17242b" stroke-width="1"/>'
        )
        lines.append(f'<text x="{lx + 46}" y="{iy}" class="legend">{label}</text>')

    sp = primary["rank_correlation_gate_g2"]["spearman_obs_vs_cliff_std_mse"]
    lines.append(
        f'<text x="{lx + 20}" y="{ly + 198}" class="tick">Spearman rho = {sp["rho"]:+.3f} [{sp["ci_low"]:+.2f}, {sp["ci_high"]:+.2f}]</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/mainnet_intervention_results.json"))
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    primary = data["primary"]

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    (args.assets_dir / "intervention_regimes.svg").write_text(
        _render_regime_bars_svg(primary), encoding="utf-8"
    )
    (args.assets_dir / "obs_vs_int_scatter.svg").write_text(_render_scatter_svg(primary), encoding="utf-8")
    print(f"Wrote SVG figures to {args.assets_dir}/")


if __name__ == "__main__":
    main()
