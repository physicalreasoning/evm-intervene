# ruff: noqa: E501
"""Render flagship SVG figures for the Next-N-Minutes ETH Price & Order-Flow Forecaster."""

from __future__ import annotations

import json
from pathlib import Path


def _load_forecast_json() -> dict | None:
    for p in (
        Path("results/forecast_suite/eth_price_forecast_results.json"),
        Path("results/eth_price_forecast_results.json"),
    ):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def plot_price_and_flow_trajectories_svg() -> None:
    data = _load_forecast_json()
    if data is None:
        return
    traj = data["crash_aug5_out_of_sample_trajectory"]
    if not traj:
        return

    W, H = 1180, 560
    pw, ph = 1040, 195
    x0 = 80
    top1, top2 = 82, 330

    n = len(traj)
    spot = [float(t["spot_eth_usd"]) for t in traj]
    true_p50 = [float(t["true_future_eth_usd_h50"]) for t in traj]
    wm_p50 = [float(t["pred_wm_eth_usd_h50"]) for t in traj]
    mlp_p50 = [float(t["pred_mlp_eth_usd_h50"]) for t in traj]
    crash_prob = [float(t["wm_crash_prob"]) for t in traj]

    true_f50 = [float(t["true_net_flow_eth_h50"]) for t in traj]
    wm_f50 = [float(t["pred_wm_net_flow_eth_h50"]) for t in traj]
    mlp_f50 = [float(t["pred_mlp_net_flow_eth_h50"]) for t in traj]

    p_min = min(min(true_p50), min(wm_p50), min(spot)) - 15.0
    p_max = max(max(true_p50), max(wm_p50), max(spot)) + 15.0
    f_bound = max(max(abs(x) for x in true_f50), max(abs(x) for x in wm_f50), 200.0) * 1.05

    def mx(idx: int) -> float:
        return x0 + (idx / max(n - 1, 1)) * pw

    def my_p(val: float) -> float:
        return top1 + ph - ((val - p_min) / max(p_max - p_min, 1.0)) * ph

    def my_f(val: float) -> float:
        return top2 + ph - ((val + f_bound) / (2.0 * f_bound)) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<style>text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1f2937}</style>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="80" y="34" font-size="18" font-weight="700">Out-of-Sample ETH Price ($ USD) &amp; Net Order-Flow Forecast (H=50 Swaps / ~12 Min Ahead)</text>',
        '<text x="80" y="55" font-size="12" fill="#4b5563">Held-out August 5, 2024 ETH crash &amp; V-bottom recovery · LiquidityCoupledWorldForecaster vs. Time-Series MLP vs. Real Mainnet Path</text>',
    ]

    # Panel A: ETH Price ($ USD)
    out.append(f'<rect x="{x0}" y="{top1}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(
        f'<text x="{x0}" y="{top1 - 9}" font-size="13" font-weight="600">A. Realized vs. Predicted Future ETH Price ($ USD, 50 Swaps Ahead) + Air-Pocket Crash Warnings (shaded red)</text>'
    )

    p_thresh = sorted(crash_prob)[int(0.75 * len(crash_prob))]
    for i in range(n - 1):
        if crash_prob[i] >= p_thresh:
            x_left = mx(i)
            x_right = mx(i + 1)
            out.append(
                f'<rect x="{x_left:.1f}" y="{top1}" width="{max(1.5, x_right - x_left):.1f}" height="{ph}" fill="#fee2e2" opacity="0.75"/>'
            )

    for k in range(5):
        v = p_min + (p_max - p_min) * k / 4.0
        yp = my_p(v)
        out.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x0 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        out.append(
            f'<text x="{x0 - 8}" y="{yp + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">${v:,.0f}</text>'
        )

    for frac in (0.0, 0.25, 0.50, 0.75, 1.0):
        idx = int(frac * (n - 1))
        xp = mx(idx)
        blk = traj[idx]["block_number"]
        out.append(f'<line x1="{xp:.1f}" y1="{top1}" x2="{xp:.1f}" y2="{top1 + ph}" stroke="#e5e7eb"/>')
        out.append(
            f'<text x="{xp:.1f}" y="{top1 + ph + 16}" font-size="10.5" text-anchor="middle" fill="#6b7280">Block {blk:,}</text>'
        )

    path_true = " ".join(f"{mx(i):.1f},{my_p(true_p50[i]):.1f}" for i in range(n))
    path_mlp = " ".join(f"{mx(i):.1f},{my_p(mlp_p50[i]):.1f}" for i in range(n))
    path_wm = " ".join(f"{mx(i):.1f},{my_p(wm_p50[i]):.1f}" for i in range(n))

    out.append(f'<polyline points="{path_true}" fill="none" stroke="#111827" stroke-width="2.3"/>')
    out.append(f'<polyline points="{path_mlp}" fill="none" stroke="#2563eb" stroke-width="1.6" opacity="0.75"/>')
    out.append(f'<polyline points="{path_wm}" fill="none" stroke="#059669" stroke-width="2.1"/>')

    # Panel B: Net Signed Order Flow (ETH over next 50 swaps)
    out.append(f'<rect x="{x0}" y="{top2}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(
        f'<text x="{x0}" y="{top2 - 9}" font-size="13" font-weight="600">B. Realized vs. Predicted Future Net Order Flow (Signed ETH over Next 50 Swaps; &gt;0 Net Sell, &lt;0 Net Buy)</text>'
    )
    y_zero = my_f(0.0)
    out.append(
        f'<line x1="{x0}" y1="{y_zero:.1f}" x2="{x0 + pw}" y2="{y_zero:.1f}" stroke="#9ca3af" stroke-dasharray="4 3"/>'
    )
    for frac_v in (-0.8, -0.4, 0.0, 0.4, 0.8):
        v = frac_v * f_bound
        yp = my_f(v)
        out.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x0 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        out.append(
            f'<text x="{x0 - 8}" y="{yp + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{v:+.0f} ETH</text>'
        )

    path_f_true = " ".join(f"{mx(i):.1f},{my_f(true_f50[i]):.1f}" for i in range(n))
    path_f_mlp = " ".join(f"{mx(i):.1f},{my_f(mlp_f50[i]):.1f}" for i in range(n))
    path_f_wm = " ".join(f"{mx(i):.1f},{my_f(wm_f50[i]):.1f}" for i in range(n))

    out.append(f'<polyline points="{path_f_true}" fill="none" stroke="#111827" stroke-width="2.0" opacity="0.85"/>')
    out.append(f'<polyline points="{path_f_mlp}" fill="none" stroke="#2563eb" stroke-width="1.5" opacity="0.75"/>')
    out.append(f'<polyline points="{path_f_wm}" fill="none" stroke="#059669" stroke-width="2.1"/>')

    ly = H - 14
    legend_items = [
        ("#111827", "Actual Future ETH Price / Net Order Flow (Ground Truth)"),
        ("#059669", "LiquidityCoupledWorldForecaster (Order-Flow + TickLens Physics)"),
        ("#2563eb", "OrderFlowMLPForecaster (Time-Series Only)"),
        ("#fca5a5", "Predicted Air-Pocket / Crash Alert Zone"),
    ]
    lx = 80
    for col, lbl in legend_items:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="15" height="6" fill="{col}"/>')
        out.append(f'<text x="{lx + 21}" y="{ly - 3}" font-size="11.5">{lbl}</text>')
        lx += 275

    out.append("</svg>")
    Path("assets/eth_price_forecast_trajectories.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote assets/eth_price_forecast_trajectories.svg")


def _load_llm_forecast_json() -> dict | None:
    p = Path("results/forecast_suite/llm_price_forecaster.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def plot_horizon_and_pnl_svg() -> None:
    data = _load_forecast_json()
    if data is None:
        return
    models = dict(data["models"])
    llm_data = _load_llm_forecast_json()
    if llm_data and "runs" in llm_data:
        for k, v in llm_data["runs"].items():
            if "error" not in v:
                models[k] = v

    W, H = 1180, 490
    pw, ph, top = 475, 320, 84
    x0, x1 = 80, 635

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<style>text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1f2937}</style>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="80" y="34" font-size="18" font-weight="700">Multi-Horizon Directional Accuracy &amp; Out-of-Sample Smart Execution Timing Savings ($100,000 Notional)</text>',
        '<text x="80" y="55" font-size="12" fill="#4b5563">Evaluated across 9,586 held-out Mainnet swaps (Aug 5 crash + Oct 2024) · TickLens-coupled physics vs. pure time-series baselines</text>',
    ]

    specs = [
        ("Qwen2.5-1.5B_QLoRA", "Qwen2.5-1.5B QLoRA + TickLens Physics", "#0f766e"),
        ("Qwen2.5-0.5B_LoRA", "Qwen2.5-0.5B LoRA + TickLens Physics", "#0284c7"),
        ("liquidity_coupled_wm", "LiquidityCoupledWorldForecaster (203k)", "#059669"),
        ("liquidity_coupled_wm_xl", "LiquidityCoupledWorldForecaster-XL (1.85M)", "#10b981"),
        ("orderflow_transformer", "OrderFlowTransformerForecaster (558k)", "#7c3aed"),
        ("orderflow_mlp", "OrderFlowMLPForecaster (191k)", "#2563eb"),
    ]
    specs = [s for s in specs if s[0] in models]

    # Panel A: Directional Accuracy across horizons H=10, 25, 50, 100
    out.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(
        f'<text x="{x0}" y="{top - 10}" font-size="13" font-weight="600">A. Price Directional Hit Rate (%) on Active Moves (|ΔP| &gt; 10 bps)</text>'
    )
    hs = ["h10", "h25", "h50", "h100"]
    h_labels = ["H=10 (~2.5m)", "H=25 (~6m)", "H=50 (~12m)", "H=100 (~25m)"]
    for i, lbl in enumerate(h_labels):
        xp = x0 + 45 + i * (pw - 90) / 3.0
        out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
        out.append(
            f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" fill="#6b7280">{lbl}</text>'
        )

    lo_a, hi_a = 45.0, 56.0
    for v in (46, 48, 50, 52, 54, 56):
        yp = top + ph - ((v - lo_a) / (hi_a - lo_a)) * ph
        stroke = "#9ca3af" if v == 50 else "#e5e7eb"
        dash = ' stroke-dasharray="4 3"' if v == 50 else ""
        out.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x0 + pw}" y2="{yp:.1f}" stroke="{stroke}"{dash}/>')
        out.append(f'<text x="{x0 - 8}" y="{yp + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{v}%</text>')

    for mk, _, col in specs:
        pts = []
        for i, hk in enumerate(hs):
            xp = x0 + 45 + i * (pw - 90) / 3.0
            val = float(models[mk]["by_horizon"][hk]["directional_acc_active_pct"])
            val_clamped = min(max(val, lo_a), hi_a)
            yp = top + ph - ((val_clamped - lo_a) / (hi_a - lo_a)) * ph
            pts.append((xp, yp, val))
        path_str = " ".join(f"{px:.1f},{py:.1f}" for px, py, _ in pts)
        out.append(f'<polyline points="{path_str}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        for px, py, val in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.0" fill="{col}"/>')
            if mk in ("Qwen2.5-1.5B_QLoRA", "liquidity_coupled_wm") and val >= 53.8:
                out.append(
                    f'<text x="{px:.1f}" y="{py - 8:.1f}" font-size="10" font-weight="600" text-anchor="middle" fill="{col}">{val:.2f}%</text>'
                )

    # Panel B: Smart Execution Timing Savings ($ USD across 383 institutional execution windows)
    out.append(f'<rect x="{x1}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(
        f'<text x="{x1}" y="{top - 10}" font-size="13" font-weight="600">B. Smart Execution Timing Savings ($ USD vs. Immediate Fill)</text>'
    )

    bar_specs = [
        ("Qwen2.5-1.5B_QLoRA", "Qwen2.5-1.5B QLoRA + TickLens", "#0f766e"),
        ("liquidity_coupled_wm_xl", "LiquidityCoupledWM-XL (1.85M)", "#10b981"),
        ("liquidity_coupled_wm", "LiquidityCoupledWM (203k)", "#059669"),
        ("Qwen2.5-0.5B_LoRA", "Qwen2.5-0.5B LoRA + TickLens", "#0284c7"),
        ("orderflow_mlp_xl", "OrderFlowMLP-XL (1.60M)", "#60a5fa"),
        ("orderflow_transformer", "OrderFlowTransformer (558k)", "#7c3aed"),
        ("orderflow_mlp", "OrderFlowMLP (191k)", "#2563eb"),
        ("orderflow_transformer_xl", "OrderFlowTransformer-XL (2.70M)", "#9333ea"),
        ("momentum_reversion_ewma", "EWMA Reversion Baseline", "#6b7280"),
    ]
    bar_specs = [b for b in bar_specs if b[0] in models]

    min_s, max_s = -4200.0, 6200.0
    x_zero = x1 + ((0.0 - min_s) / (max_s - min_s)) * pw
    for v in (-4000, -2000, 0, 2000, 4000, 6000):
        xp = x1 + ((v - min_s) / (max_s - min_s)) * pw
        stroke = "#6b7280" if v == 0 else "#e5e7eb"
        dash = ' stroke-dasharray="4 3"' if v == 0 else ""
        out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="{stroke}"{dash}/>')
        out.append(
            f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="10.5" text-anchor="middle" fill="#6b7280">${v:+,.0f}</text>'
        )

    row_h = ph / len(bar_specs)
    for idx, (mk, lbl, col) in enumerate(bar_specs):
        bt_key = "trading_backtest_h50" if "trading_backtest_h50" in models[mk] else "trading_backtest_h25"
        sav = float(models[mk][bt_key]["execution_timing_total_savings_usd"])
        bps = float(models[mk][bt_key]["execution_timing_mean_savings_bps"])
        yc = top + (idx + 0.5) * row_h
        bh_bar = min(20.0, row_h * 0.62)
        xp = x1 + ((sav - min_s) / (max_s - min_s)) * pw
        bx = min(x_zero, xp)
        bw = max(2.0, abs(xp - x_zero))
        out.append(
            f'<rect x="{bx:.1f}" y="{yc - bh_bar / 2:.1f}" width="{bw:.1f}" height="{bh_bar:.1f}" fill="{col}" rx="2"/>'
        )
        if sav >= 0:
            out.append(
                f'<text x="{xp + 5:.1f}" y="{yc + 4:.1f}" font-size="10" font-weight="600" fill="#111827">${sav:+,.0f} ({bps:+.2f} bps)</text>'
            )
            out.append(
                f'<text x="{x_zero - 6:.1f}" y="{yc + 4:.1f}" font-size="10" text-anchor="end" fill="#374151">{lbl}</text>'
            )
        else:
            out.append(
                f'<text x="{xp - 5:.1f}" y="{yc + 4:.1f}" font-size="10" font-weight="600" text-anchor="end" fill="#991b1b">${sav:+,.0f}</text>'
            )
            out.append(
                f'<text x="{x_zero + 6:.1f}" y="{yc + 4:.1f}" font-size="10" fill="#374151">{lbl}</text>'
            )

    ly = H - 18
    lx = 60
    for _, lbl, col in specs[:5]:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="14" height="5" fill="{col}"/>')
        out.append(f'<text x="{lx + 18}" y="{ly - 3}" font-size="10.5">{lbl}</text>')
        lx += 222

    out.append("</svg>")
    Path("assets/forecast_horizon_and_pnl.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote assets/forecast_horizon_and_pnl.svg")


if __name__ == "__main__":
    plot_price_and_flow_trajectories_svg()
    plot_horizon_and_pnl_svg()
