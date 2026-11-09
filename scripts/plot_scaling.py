"""Render assets/scaling_sweep.svg directly from results/scaling_sweep_results.json (no matplotlib)."""

from __future__ import annotations

import json
import math
from pathlib import Path

COLORS = {"mlp_direct": "#2563eb", "seq_transformer": "#7c3aed", "latent_jepa": "#d97706"}
LABELS = {"mlp_direct": "Direct MLP", "seq_transformer": "Cross-attention Transformer", "latent_jepa": "Latent JEPA"}


def main() -> None:
    s = json.loads(Path("results/scaling_sweep_results.json").read_text(encoding="utf-8"))
    anchor = s["anchor_conservation_wm"]
    series: dict[str, list[tuple[int, float, float]]] = {f: [] for f in COLORS}
    for fams in s["scaling_tiers"].values():
        for fam, v in fams.items():
            series[fam].append((v["params"], v["cliff_rmse_bps_mean"], v["obs_val_std_mse_mean"]))

    W, H = 1100, 470
    panels = [(70, "Observational validation loss (std-MSE)", 2, 0.0, 0.6),
              (610, "Held-out cliff price RMSE (bps)", 1, 0.0, 240.0)]
    pw, ph, top = 420, 330, 80
    xmin, xmax = math.log10(1e3), math.log10(3e8)

    def mx(x0: int, p: float) -> float:
        return x0 + (math.log10(p) - xmin) / (xmax - xmin) * pw

    def my(v: float, lo: float, hi: float) -> float:
        return top + ph - (v - lo) / (hi - lo) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<style>text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:#1f2937}</style>',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           '<text x="70" y="36" font-size="18" font-weight="700">Parameter scaling: observational fit improves, cliff error does not</text>',
           f'<text x="70" y="58" font-size="12" fill="#6b7280">{s["dataset_summary"]["counterfactual_interventions"]:,} held-out '
           'interventions; mean of 3 seeds per point; dashed line = conservation_wm (2,549 params)</text>']
    for x0, title, idx, lo, hi in panels:
        out.append(f'<rect x="{x0}" y="{top}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x0}" y="{top - 8}" font-size="13" font-weight="600">{title}</text>')
        for e in range(3, 9):
            xp = mx(x0, 10**e)
            out.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top + ph}" stroke="#e5e7eb"/>')
            out.append(f'<text x="{xp:.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" fill="#6b7280">10^{e}</text>')
        for k in range(5):
            v = lo + (hi - lo) * k / 4
            yp = my(v, lo, hi)
            out.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x0 + pw}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
            out.append(f'<text x="{x0 - 8}" y="{yp + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{v:g}</text>')
        out.append(f'<text x="{x0 + pw / 2}" y="{top + ph + 40}" font-size="12" text-anchor="middle">Parameters (log scale)</text>')
        av = anchor["cliff_rmse_bps_mean"] if idx == 1 else anchor["obs_val_std_mse_mean"]
        out.append(f'<line x1="{x0}" y1="{my(av, lo, hi):.1f}" x2="{x0 + pw}" y2="{my(av, lo, hi):.1f}" '
                   'stroke="#059669" stroke-width="2.5" stroke-dasharray="7 5"/>')
        out.append(f'<circle cx="{mx(x0, anchor["params"]):.1f}" cy="{my(av, lo, hi):.1f}" r="6" fill="#059669"/>')
        for fam, pts in series.items():
            pts = sorted(pts)
            path = " ".join(f"{mx(x0, p):.1f},{my(v[idx - 1] if idx == 1 else v[1], lo, hi):.1f}"
                            for p, *v in [(p, c, o) for p, c, o in pts])
            out.append(f'<polyline points="{path}" fill="none" stroke="{COLORS[fam]}" stroke-width="2.5"/>')
            for p, c, o in pts:
                val = c if idx == 1 else o
                out.append(f'<circle cx="{mx(x0, p):.1f}" cy="{my(val, lo, hi):.1f}" r="4" fill="{COLORS[fam]}"/>')
    ly = H - 22
    lx = 70
    for fam in COLORS:
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="14" height="4" fill="{COLORS[fam]}"/>')
        out.append(f'<text x="{lx + 20}" y="{ly - 3}" font-size="12">{LABELS[fam]}</text>')
        lx += 230
    out.append(f'<rect x="{lx}" y="{ly - 9}" width="14" height="4" fill="#059669"/>')
    out.append(f'<text x="{lx + 20}" y="{ly - 3}" font-size="12">conservation_wm (2,549 params)</text>')
    out.append("</svg>")
    Path("assets/scaling_sweep.svg").write_text("\n".join(out), encoding="utf-8")
    print("wrote assets/scaling_sweep.svg")


if __name__ == "__main__":
    main()
