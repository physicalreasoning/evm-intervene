"""Gate G0 empirical verification: test `F_AMM` against real Ethereum Mainnet Swap logs.

Pairs consecutive Swap events in the same block/window where no Mint/Burn altered liquidity
(or initializes from exact archive checkpoints) and measures basis-point error between our
deterministic simulator and actual on-chain EVM `sqrtPriceX96` and `liquidity` outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evm_intervene.simulator import (
    PoolState,
    eth_usd_from_raw_sqrt_price,
    raw_sqrt_price_from_x96,
    simulate_amm_swap_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/simulator_verification.json"))
    args = parser.parse_args()

    swaps_by_win = json.loads((args.data_dir / "mainnet_swaps.json").read_text(encoding="utf-8"))
    cps_by_win = json.loads((args.data_dir / "mainnet_checkpoints.json").read_text(encoding="utf-8"))

    price_err_bps: list[float] = []
    multi_tick_err_bps: list[float] = []
    liq_exact_matches = 0
    n_evaluated = 0
    n_multi_tick = 0

    for win_name, swaps in swaps_by_win.items():
        cps = cps_by_win[win_name]
        cp_blocks = [c["block_number"] for c in cps]

        for i in range(1, len(swaps)):
            prev_s = swaps[i - 1]
            cur_s = swaps[i]
            # Use nearest preceding checkpoint's tick map for tick-crossing deltas.
            b_cur = cur_s["block_number"]
            cp_idx = max(0, int(np.searchsorted(cp_blocks, b_cur, side="right")) - 1)
            cp = cps[cp_idx]

            # Filter out pairs where an intervening Mint/Burn changed active in-range liquidity
            # when the swap itself stayed within the same tick.
            prev_tick = int(prev_s["tick"])
            cur_tick = int(cur_s["tick"])
            prev_liq = float(int(prev_s["liquidity"]))
            cur_liq = float(int(cur_s["liquidity"]))
            if abs(cur_tick - prev_tick) == 0 and prev_liq != cur_liq:
                continue
            if b_cur - int(prev_s["block_number"]) > 25:
                continue

            state = PoolState(
                block_number=b_cur,
                sqrt_price_raw=raw_sqrt_price_from_x96(int(prev_s["sqrt_price_x96"])),
                tick=prev_tick,
                liquidity=prev_liq,
                ticks=tuple(int(t) for t in cp["ticks"]),
                liquidity_net=tuple(float(int(ln)) for ln in cp["liquidity_net"]),
            )

            amt0 = int(cur_s["amount0_raw"])
            amt1 = int(cur_s["amount1_raw"])
            if amt0 > 0:
                q_pred, _, L_pred, _, _, crossed = simulate_amm_swap_raw(
                    state, zero_for_one=True, amount_in_raw=float(amt0)
                )
            elif amt1 > 0:
                q_pred, _, L_pred, _, _, crossed = simulate_amm_swap_raw(
                    state, zero_for_one=False, amount_in_raw=float(amt1)
                )
            else:
                continue

            p_pred = eth_usd_from_raw_sqrt_price(q_pred)
            p_true = eth_usd_from_raw_sqrt_price(raw_sqrt_price_from_x96(int(cur_s["sqrt_price_x96"])))
            err_bps = abs(math.log(p_pred / p_true)) * 1e4

            price_err_bps.append(err_bps)
            n_evaluated += 1
            if abs(L_pred - cur_liq) / max(cur_liq, 1.0) < 1e-6:
                liq_exact_matches += 1
            if crossed > 0 or abs(cur_tick - prev_tick) >= 10:
                n_multi_tick += 1
                multi_tick_err_bps.append(err_bps)

    arr = np.array(price_err_bps, dtype=np.float64)
    arr_mt = np.array(multi_tick_err_bps, dtype=np.float64) if multi_tick_err_bps else np.zeros(1)

    summary = {
        "n_mainnet_swaps_verified": n_evaluated,
        "n_tick_crossing_swaps": n_multi_tick,
        "median_price_error_bps": float(np.median(arr)),
        "p95_price_error_bps": float(np.percentile(arr, 95)),
        "p99_price_error_bps": float(np.percentile(arr, 99)),
        "mean_price_error_bps": float(np.mean(arr)),
        "median_tick_crossing_price_error_bps": float(np.median(arr_mt)),
        "liquidity_exact_match_rate": round(liq_exact_matches / max(n_evaluated, 1), 6),
        "gate_g0_median_lt_0_05_bps_pass": bool(np.median(arr) < 0.05),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
