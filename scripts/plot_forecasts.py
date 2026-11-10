# ruff: noqa: E501
"""Render flagship SVG figures for the Next-N-Minutes ETH Price & Order-Flow Forecaster."""

from __future__ import annotations

import json
from pathlib import Path


def plot_price_and_flow_trajectories_svg() -> None:
    path = Path("results/eth_price_forecast_results.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
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

    # Shaded crash warning zones where wm_crash_prob is elevated
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

    # Plot curves in Panel A
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

    # Legend
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


def plot_horizon_and_pnl_svg() -> None:
    path = Path("results/eth_price_forecast_results.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data["models"]

    W, H = 1180, 480
    pw, ph, top = 475, 320, 84
    x0, x1 = 80, 635

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<style>text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1f2937}</style>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="80" y="34" font-size="18" font-weight="700">Multi-Horizon Directional Accuracy &amp; Out-of-Sample Trading Strategy PnL ($100,000 Notional)</text>',
        '<text x="80" y="55" font-size="12" fill="#4b5563">Evaluated across 7,521 held-out Mainnet swaps (Aug 5 crash + Oct 2024) · Net of Uniswap V3 0.05% (5 bps) pool fees</text>',
    ]

    specs = [
        ("liquidity_coupled_wm", "LiquidityCoupledWorldForecaster (Ours)", "#059669"),
        ("orderflow_transformer", "OrderFlowTransformerForecaster", "#7c3aed"),
        ("orderflow_mlp", "OrderFlowMLPForecaster (Time-Series)", "#2563eb"),
        ("momentum_reversion_ewma", "EWMA Reversion Baseline", "#6b7280"),
    ]

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

    lo_a, hi_a = 45.0, 75.0
    for v in (45, 50, 55, 60, 65, 70, 75):
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
        out.append(f'<polyline points="{path_str}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for px, py, val in pts:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="{col}"/>')
            if mk == "liquidity_coupled_wm":
                out.append(
                    f'<text x="{px:.1f}" y="{py - 8:.1f}" font-size="10.5" font-weight="600" text-anchor="middle" fill="{col}">{val:.1f}%</text>'
                )

    # Panel B: Cumulative Net PnL ($ USD) on $100k Notional
    out.append(f'<rect x="{x1}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>')
    out.append(
        f'<text x="{x1}" y="{top - 10}" font-size="13" font-weight="600">B. Cumulative Net Trading Strategy Profit ($ USD, Net of 5 bps Pool Fee)</text>'
    )
    bt_key = "trading_backtest_h50" if "trading_backtest_h50" in models[specs[0][0]] else "trading_backtest_h25"
    all_curves = [models[mk][bt_key]["cum_net_pnl_curve_usd"] for mk, _, _ in specs]
    n_steps = len(all_curves[0])
    min_pnl = min(min(min(c) for c in all_curves), -500.0) * 1.1
    max_pnl = max(max(max(c) for c in all_curves), 1000.0) * 1.1

    for k in range(5):
        v = min_pnl + (max_pnl - min_pnl) * k / 4.0
        yp = top + ph - ((v - min_pnl) / max(max_pnl - min_pnl, 1.0)) * ph
        out.append(f'<line x1="{x1}" y1="{yp:.1f}" x2="{x1 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        out.append(
            f'<text x="{x1 - 8}" y="{yp + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">${v:+,.0f}</text>'
        )
    y0_pnl = top + ph - ((0.0 - min_pnl) / max(max_pnl - min_pnl, 1.0)) * ph
    out.append(
        f'<line x1="{x1}" y1="{y0_pnl:.1f}" x2="{x1 + pw}" y2="{y0_pnl:.1f}" stroke="#9ca3af" stroke-dasharray="4 3"/>'
    )
    out.append(
        f'<text x="{x1 + pw / 2}" y="{top + ph + 28}" font-size="11.5" text-anchor="middle" fill="#4b5563">Out-of-sample trading intervals (Aug 5 crash + Oct 2024)</text>'
    )

    for mk, _, col in specs:
        curve = models[mk][bt_key]["cum_net_pnl_curve_usd"]
        pts_b = []
        for i, val in enumerate(curve):
            xp = x1 + 12 + (i / max(n_steps - 1, 1)) * (pw - 24)
            yp = top + ph - ((val - min_pnl) / max(max_pnl - min_pnl, 1.0)) * ph
            pts_b.append((xp, yp))
        path_b = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts_b)
        out.append(f'<polyline points="{path_b}" fill="none" stroke="{col}" stroke-width="2.4"/>')

    ly = H - 16
    lx = 80
    for _, lbl, col in specs:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="15" height="5" fill="{col}"/>')
        out.append(f'<text x="{lx + 20}" y="{ly - 3}" font-size="11.5">{lbl}</text>')
        lx += 275

    out.append("</svg>")
    Path("assets/forecast_horizon_and_pnl.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote assets/forecast_horizon_and_pnl.svg")


if __name__ == "__main__":
    plot_price_and_flow_trajectories_svg()
    plot_horizon_and_pnl_svg()
