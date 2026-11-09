"""Heavy Colab Pro experiment suite for evm-intervene.

Three flagship experiments with incremental JSON checkpointing after every sub-run:

  1. rollout     Multi-step autoregressive counterfactual rollouts (H = 1, 2, 3, 5, 8, 12
                 consecutive swaps) across 62 held-out Mainnet checkpoints x 4 stress
                 scenarios (2,976 sequential transitions). Tests whether single-step
                 boundary errors compound catastrophically over multi-step trajectories.

  2. planner     Downstream decision value: risk-constrained optimal execution sizing
                 across 62 checkpoints x 2 directions x 5 slippage budgets B in
                 {25, 50, 100, 200, 400} bps (620 execution tasks, 48 candidate sizes
                 each = 29,760 counterfactual evaluations). Measures budget violation
                 rate (%), excess slippage overshoot (bps), and sizing error vs oracle.

  3. llm_ladder  Full Foundation-Model Scaling Ladder (0.5B, 0.5B-random-init, 1.5B,
                 3B, 7B r=16, 7B r=64, and 14B QLoRA) evaluated on single-step regimes,
                 12-step autoregressive rollouts, and downstream execution planning.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import scripts.run_experiment as rx
import scripts.run_research_v2 as rv2
from evm_intervene.baselines import predict_copy
from evm_intervene.metrics import TargetScaler
from evm_intervene.models import (
    BatchTensors,
    default_torch_device,
    evaluate_model_predictions,
    train_world_model,
)
from evm_intervene.simulator import (
    BorrowerPosition,
    PoolState,
    TransitionOutcome,
    raw_sqrt_price_from_eth_usd,
    raw_sqrt_price_to_tick,
    simulate_intervention,
)
from evm_intervene.state import (
    EncodedTransition,
    encode_state_action,
)

OUT_DIR = Path("results/heavy_suite")
HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 8, 12)
MAX_H = max(HORIZONS)
BUDGETS_BPS: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0, 400.0)

MODEL_SPECS: list[tuple[str, str, dict[str, Any]]] = [
    ("conservation_wm_none", "conservation_wm", {"ablation": "none"}),
    ("conservation_wm_no_bin_net", "conservation_wm", {"ablation": "no_bin_net"}),
    ("mlp_M_1.5M", "mlp_direct", {"hidden_dim": 640, "depth": 4}),
    ("mlp_L_8M", "mlp_direct", {"hidden_dim": 1536, "depth": 6}),
    ("transformer_M_1.5M", "seq_transformer", {"d_model": 256, "nhead": 8, "num_layers": 3}),
    ("transformer_L_8M", "seq_transformer", {"d_model": 512, "nhead": 8, "num_layers": 4}),
    ("jepa_M_1.5M", "latent_jepa", {"latent_dim": 512, "depth": 4}),
    ("jepa_L_8M", "latent_jepa", {"latent_dim": 1024, "depth": 5}),
]


# --------------------------------------------------------------------------------------
# Checkpoint & Trajectory Builders
# --------------------------------------------------------------------------------------
def _extract_window_checkpoints(
    checkpoints: list[dict[str, Any]],
    swaps: list[dict[str, Any]],
    all_borrowers: list[BorrowerPosition] | None = None,
    *,
    min_block: int = 0,
    max_block: int = 10**12,
    stride: int = 1,
    window_tag: str = "",
) -> list[tuple[PoolState, np.ndarray, str]]:
    """Fast extraction of held-out (PoolState, hist_summary, window_tag) checkpoints."""
    from evm_intervene.simulator import eth_usd_from_raw_sqrt_price, raw_sqrt_price_from_x96
    from evm_intervene.state import summarize_swap_history

    swap_blocks = np.array([int(s["block_number"]) for s in swaps], dtype=np.int64)
    out: list[tuple[PoolState, np.ndarray, str]] = []
    running_min_eth = float("inf")
    kept_idx = 0

    for cp in checkpoints:
        b_num = int(cp["block_number"])
        p_spot = eth_usd_from_raw_sqrt_price(raw_sqrt_price_from_x96(int(cp["sqrt_price_x96"])))
        running_min_eth = min(running_min_eth, p_spot)
        if b_num < min_block or b_num > max_block:
            continue
        kept_idx += 1
        if (kept_idx - 1) % stride != 0:
            continue

        active_borrowers: tuple[BorrowerPosition, ...] = ()
        if all_borrowers:
            active_borrowers = tuple(
                b for b in all_borrowers if b.critical_eth_price < running_min_eth * 0.999
            )
        state = rx._checkpoint_to_pool_state(cp, borrowers=active_borrowers)

        end_idx = int(np.searchsorted(swap_blocks, b_num, side="left"))
        start_idx = max(1, end_idx - 16)
        h_act: list[float] = []
        h_dlp: list[float] = []
        for k in range(start_idx, end_idx):
            p_prev = eth_usd_from_raw_sqrt_price(raw_sqrt_price_from_x96(int(swaps[k - 1]["sqrt_price_x96"])))
            p_cur = eth_usd_from_raw_sqrt_price(raw_sqrt_price_from_x96(int(swaps[k]["sqrt_price_x96"])))
            a1 = int(swaps[k]["amount1_raw"])
            a0 = int(swaps[k]["amount0_raw"])
            if a1 > 0:
                h_act.append(a1 / 1e18)
            elif a0 > 0:
                h_act.append(-(a0 / 1e6) / p_prev)
            else:
                continue
            h_dlp.append(float(np.log(p_cur / p_prev)))
        hist_summary = summarize_swap_history(np.array(h_act), np.array(h_dlp))
        out.append((state, hist_summary, window_tag))
    return out


def load_held_out_checkpoints(splits: dict[str, Any]) -> list[tuple[PoolState, np.ndarray, str]]:
    """Extract the 62 held-out (PoolState, hist_summary, window_tag) checkpoints in < 0.05s."""
    borrowers = splits["borrowers"]
    cps = splits["cps"]
    swaps = splits["swaps"]

    c1 = _extract_window_checkpoints(
        cps["crash_aug5_2024"], swaps["crash_aug5_2024"], all_borrowers=borrowers,
        min_block=20_458_000, max_block=20_460_400, window_tag="crash_onset",
    )
    c2 = _extract_window_checkpoints(
        cps["crash_aug5_2024"], swaps["crash_aug5_2024"], all_borrowers=borrowers,
        min_block=20_465_001, max_block=20_468_000, window_tag="crash_late",
    )
    c3 = _extract_window_checkpoints(
        cps["autumn_oct_2024"], swaps["autumn_oct_2024"], stride=2, window_tag="oct",
    )
    return c1 + c2 + c3


def advance_exact_pool_state(state: PoolState, outcome: TransitionOutcome) -> PoolState:
    """Advance true PoolState after `simulate_intervention(state, a_t)`."""
    rem_borrowers: tuple[BorrowerPosition, ...] = state.borrowers
    if outcome.liquidated_eth > 0.0 and len(state.borrowers) > 0:
        # Remove or scale down borrowers that crossed below HF 1.0 at post_eth_usd
        kept: list[BorrowerPosition] = []
        for b in state.borrowers:
            if b.health_factor(outcome.post_eth_usd) >= 1.0:
                kept.append(b)
        rem_borrowers = tuple(kept)
    return PoolState(
        block_number=state.block_number,
        sqrt_price_raw=outcome.post_sqrt_price_raw,
        tick=outcome.post_tick,
        liquidity=outcome.post_liquidity,
        ticks=state.ticks,
        liquidity_net=state.liquidity_net,
        borrowers=rem_borrowers,
        fee_rate=state.fee_rate,
    )


def advance_predicted_pool_state(
    state_hat: PoolState,
    pred_dlogp: float,
    pred_dlogl: float,
) -> PoolState:
    """Autoregressively update a world model's estimated PoolState from its predicted (dlogP, dlogL)."""
    dlogp_clamped = float(np.clip(pred_dlogp, -1.2, 1.2))
    dlogl_clamped = float(np.clip(pred_dlogl, -6.0, 6.0))
    p_next = max(50.0, min(50_000.0, state_hat.eth_usd * math.exp(dlogp_clamped)))
    l_next = max(1.0, min(1e26, state_hat.liquidity * math.exp(dlogl_clamped)))
    q_next = raw_sqrt_price_from_eth_usd(p_next)
    tick_next = raw_sqrt_price_to_tick(q_next)
    return PoolState(
        block_number=state_hat.block_number,
        sqrt_price_raw=q_next,
        tick=tick_next,
        liquidity=l_next,
        ticks=state_hat.ticks,
        liquidity_net=state_hat.liquidity_net,
        borrowers=state_hat.borrowers,
        fee_rate=state_hat.fee_rate,
    )


def build_rollout_GroundTruth(
    checkpoints: list[tuple[PoolState, np.ndarray, str]],
) -> list[dict[str, Any]]:
    """Build 4 x 62 = 248 exact 12-step counterfactual trajectories."""
    dummy_outcome = TransitionOutcome(
        pre_eth_usd=3000.0, post_amm_eth_usd=3000.0, post_eth_usd=3000.0,
        pre_liquidity=1e19, post_liquidity=1e19, post_sqrt_price_raw=1.0,
        post_tick=0, action_eth_signed=0.0, usdc_delta=0.0, weth_delta=0.0,
        ticks_crossed=0, cascade_rounds=0, liquidated_eth=0.0, liquidated_accounts=0,
    )
    trajectories: list[dict[str, Any]] = []
    scenario_mults: dict[str, list[float]] = {
        "sell_burst": [0.65] * MAX_H,
        "buy_burst": [-0.65] * MAX_H,
        "stress_whipsaw": [1.6, 1.8, -1.4, 2.1, -1.6, 2.4, -1.8, 2.2, 1.5, -2.0, 2.5, -1.5][:MAX_H],
        "cascade_drill": [1.25] * MAX_H,
    }
    for cp_state, hist_summary, tag in checkpoints:
        sell_cap = min(max(cp_state.max_in_tick_eth_sell(), 25.0), 600.0)
        buy_cap = min(max(cp_state.max_in_tick_eth_buy_equiv(), 25.0), 600.0)
        for scen_name, mults in scenario_mults.items():
            actions = [float(m * sell_cap if m >= 0 else m * buy_cap) for m in mults]
            cur = cp_state
            p0 = cp_state.eth_usd
            l0 = max(cp_state.liquidity, 1.0)
            cum_dlogp: list[float] = []
            cum_dlogl: list[float] = []
            ticks_crossed_cum = 0
            for a_eth in actions:
                out = simulate_intervention(cur, a_eth, enable_liquidations=True)
                ticks_crossed_cum += out.ticks_crossed
                cum_dlogp.append(math.log(out.post_eth_usd / p0))
                cum_dlogl.append(math.log(max(out.post_liquidity, 1.0) / l0))
                cur = advance_exact_pool_state(cur, out)
            trajectories.append(
                {
                    "init_state": cp_state,
                    "hist_summary": hist_summary,
                    "scenario": scen_name,
                    "window_tag": tag,
                    "actions": actions,
                    "true_cum_dlogp": np.array(cum_dlogp, dtype=np.float64),
                    "true_cum_dlogl": np.array(cum_dlogl, dtype=np.float64),
                    "total_ticks_crossed": ticks_crossed_cum,
                    "dummy_outcome": dummy_outcome,
                }
            )
    return trajectories


@torch.no_grad()
def evaluate_autoregressive_rollout(
    predict_fn,
    trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    """Roll out a world model autoregressively for MAX_H=12 steps across all 248 trajectories."""
    n_traj = len(trajectories)
    cur_states = [t["init_state"] for t in trajectories]
    hists = [t["hist_summary"] for t in trajectories]
    dummy = trajectories[0]["dummy_outcome"]

    pred_cum_dlogp = np.zeros((n_traj, MAX_H), dtype=np.float64)
    pred_cum_dlogl = np.zeros((n_traj, MAX_H), dtype=np.float64)

    for step in range(MAX_H):
        step_samples: list[EncodedTransition] = [
            encode_state_action(
                cur_states[i],
                trajectories[i]["actions"][step],
                dummy,
                hists[i],
                regime="observational",
            )
            for i in range(n_traj)
        ]
        preds = predict_fn(step_samples)  # shape (n_traj, 3)
        for i in range(n_traj):
            dl_p = float(preds[i, 0])
            dl_l = float(preds[i, 1])
            prev_p = 0.0 if step == 0 else pred_cum_dlogp[i, step - 1]
            prev_l = 0.0 if step == 0 else pred_cum_dlogl[i, step - 1]
            pred_cum_dlogp[i, step] = prev_p + dl_p
            pred_cum_dlogl[i, step] = prev_l + dl_l
            cur_states[i] = advance_predicted_pool_state(cur_states[i], dl_p, dl_l)

    true_dlogp = np.stack([t["true_cum_dlogp"] for t in trajectories], axis=0)
    true_dlogl = np.stack([t["true_cum_dlogl"] for t in trajectories], axis=0)

    by_horizon: dict[str, dict[str, float]] = {}
    for h in HORIZONS:
        idx = h - 1
        err_p_bps = (pred_cum_dlogp[:, idx] - true_dlogp[:, idx]) * 1e4
        err_l = pred_cum_dlogl[:, idx] - true_dlogl[:, idx]
        by_horizon[f"h{h}"] = {
            "horizon": h,
            "cum_price_rmse_bps": round(float(np.sqrt(np.mean(err_p_bps**2))), 2),
            "cum_price_mae_bps": round(float(np.mean(np.abs(err_p_bps))), 2),
            "cum_price_p95_bps": round(float(np.percentile(np.abs(err_p_bps), 95)), 2),
            "cum_liquidity_rmse": round(float(np.sqrt(np.mean(err_l**2))), 4),
        }
    return by_horizon


# --------------------------------------------------------------------------------------
# Experiment 2: Risk-Constrained Execution Planner
# --------------------------------------------------------------------------------------
def build_planner_suite(
    checkpoints: list[tuple[PoolState, np.ndarray, str]],
    n_grid: int = 48,
) -> dict[str, Any]:
    """Precompute exact EVM price impact across 48 candidate order sizes for 62 checkpoints x 2 directions."""
    mult_grid = np.geomspace(0.15, 20.0, num=n_grid)
    samples: list[EncodedTransition] = []
    true_impact_bps = np.zeros((len(checkpoints), 2, n_grid), dtype=np.float64)
    cand_actions_eth = np.zeros((len(checkpoints), 2, n_grid), dtype=np.float64)
    notional_usd = np.zeros((len(checkpoints), 2, n_grid), dtype=np.float64)

    for c_idx, (state, hist, _) in enumerate(checkpoints):
        sell_cap = min(max(state.max_in_tick_eth_sell(), 25.0), 600.0)
        buy_cap = min(max(state.max_in_tick_eth_buy_equiv(), 25.0), 600.0)
        for d_idx, sign in enumerate((1.0, -1.0)):
            cap = sell_cap if sign > 0 else buy_cap
            for g_idx, m in enumerate(mult_grid):
                a_eth = float(sign * m * cap)
                out = simulate_intervention(state, a_eth, enable_liquidations=True)
                samples.append(encode_state_action(state, a_eth, out, hist, regime="observational"))
                true_impact_bps[c_idx, d_idx, g_idx] = abs(out.delta_log_price) * 1e4
                cand_actions_eth[c_idx, d_idx, g_idx] = abs(a_eth)
                notional_usd[c_idx, d_idx, g_idx] = abs(a_eth) * state.eth_usd

    return {
        "samples": samples,
        "true_impact_bps": true_impact_bps,
        "cand_actions_eth": cand_actions_eth,
        "notional_usd": notional_usd,
        "n_checkpoints": len(checkpoints),
        "n_grid": n_grid,
    }


def evaluate_planner(pred_Fn, planner_suite: dict[str, Any]) -> dict[str, Any]:
    """Evaluate world-model-guided order sizing across budgets B in {25, 50, 100, 200, 400} bps."""
    preds = pred_Fn(planner_suite["samples"])  # (N, 3)
    n_cp = planner_suite["n_checkpoints"]
    n_grid = planner_suite["n_grid"]
    pred_impact_bps = np.abs(preds[:, 0].reshape(n_cp, 2, n_grid)) * 1e4
    true_impact_bps = planner_suite["true_impact_bps"]
    cand_q = planner_suite["cand_actions_eth"]
    notional = planner_suite["notional_usd"]

    by_budget: dict[str, Any] = {}
    all_violations: list[bool] = []
    all_overshoots: list[float] = []
    all_rel_sizing_err: list[float] = []
    all_excess_usd: list[float] = []

    for B in BUDGETS_BPS:
        b_viol: list[bool] = []
        b_over: list[float] = []
        b_rel_q: list[float] = []
        b_usd: list[float] = []

        for c_idx in range(n_cp):
            for d_idx in range(2):
                p_curve = pred_impact_bps[c_idx, d_idx]
                t_curve = true_impact_bps[c_idx, d_idx]
                q_curve = cand_q[c_idx, d_idx]
                u_curve = notional[c_idx, d_idx]

                feasible_pred = np.where(p_curve <= B)[0]
                chosen_idx = int(feasible_pred[-1]) if len(feasible_pred) > 0 else 0

                feasible_true = np.where(t_curve <= B)[0]
                oracle_idx = int(feasible_true[-1]) if len(feasible_true) > 0 else 0

                realized_bps = float(t_curve[chosen_idx])
                overshoot = max(0.0, realized_bps - B)
                violated = realized_bps > (B * 1.05)

                q_hat = float(q_curve[chosen_idx])
                q_star = float(q_curve[oracle_idx])
                rel_q_err = abs(q_hat - q_star) / max(q_star, 1e-6)
                excess_usd = float(u_curve[chosen_idx]) * (overshoot * 1e-4)

                b_viol.append(violated)
                b_over.append(overshoot)
                b_rel_q.append(rel_q_err)
                b_usd.append(excess_usd)

        by_budget[f"B_{int(B)}bps"] = {
            "budget_bps": B,
            "violation_rate_pct": round(100.0 * float(np.mean(b_viol)), 2),
            "overshoot_rmse_bps": round(float(np.sqrt(np.mean(np.square(b_over)))), 2),
            "overshoot_p95_bps": round(float(np.percentile(b_over, 95)), 2),
            "sizing_rel_err_pct": round(100.0 * float(np.mean(b_rel_q)), 2),
            "mean_excess_slippage_usd": round(float(np.mean(b_usd)), 2),
        }
        all_violations.extend(b_viol)
        all_overshoots.extend(b_over)
        all_rel_sizing_err.extend(b_rel_q)
        all_excess_usd.extend(b_usd)

    return {
        "overall": {
            "violation_rate_pct": round(100.0 * float(np.mean(all_violations)), 2),
            "overshoot_rmse_bps": round(float(np.sqrt(np.mean(np.square(all_overshoots)))), 2),
            "overshoot_p95_bps": round(float(np.percentile(all_overshoots, 95)), 2),
            "sizing_rel_err_pct": round(100.0 * float(np.mean(all_rel_sizing_err)), 2),
            "mean_excess_slippage_usd": round(float(np.mean(all_excess_usd)), 2),
        },
        "by_budget": by_budget,
    }


# --------------------------------------------------------------------------------------
# Runner for Rollout + Planner across Scratch & Conservation Models
# --------------------------------------------------------------------------------------
def run_rollout_and_planner(
    splits: dict[str, Any],
    device: torch.device,
    seeds: tuple[int, ...],
    epochs: int,
    *,
    do_rollout: bool = True,
    do_planner: bool = True,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rollout_path = OUT_DIR / "rollout.json"
    planner_path = OUT_DIR / "planner.json"

    scaler = TargetScaler.from_targets(np.stack([s.target for s in splits["train"]], axis=0))
    train_b = BatchTensors.from_samples(splits["train"], device)
    val_b = BatchTensors.from_samples(splits["val"], device)
    test_b = BatchTensors.from_samples(splits["test"], device)

    checkpoints = load_held_out_checkpoints(splits)
    trajectories = build_rollout_GroundTruth(checkpoints) if do_rollout else []
    planner_suite = build_planner_suite(checkpoints) if do_planner else {}

    rollout_res: dict[str, Any] = (
        json.loads(rollout_path.read_text(encoding="utf-8"))
        if (do_rollout and rollout_path.exists())
        else {"seeds": list(seeds), "epochs": epochs, "n_trajectories": len(trajectories), "models": {}}
    )
    planner_res: dict[str, Any] = (
        json.loads(planner_path.read_text(encoding="utf-8"))
        if (do_planner and planner_path.exists())
        else {"seeds": list(seeds), "epochs": epochs, "n_checkpoints": len(checkpoints), "models": {}}
    )

    for spec_name, family, kwargs in MODEL_SPECS:
        done_r = (not do_rollout) or (spec_name in rollout_res.get("models", {}))
        done_p = (not do_planner) or (spec_name in planner_res.get("models", {}))
        if done_r and done_p:
            print(f"[suite] skipping already-completed {spec_name}", flush=True)
            continue
        r_runs: list[dict[str, Any]] = []
        p_runs: list[dict[str, Any]] = []
        for seed in seeds:
            t0 = time.time()
            bundle = train_world_model(
                family, train_b, val_b, scaler, seed=seed, epochs=epochs,
                batch_size=512, lr=1.5e-3, model_kwargs=kwargs,
            )
            model = bundle.best_model
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            preds_test = evaluate_model_predictions(model, test_b)
            h_single = rv2.headline(rv2.summarize(preds_test, splits["test"], scaler))

            def _pred_fn(sample_list: list[EncodedTransition], _m=model) -> np.ndarray:
                b = BatchTensors.from_samples(sample_list, device)
                return evaluate_model_predictions(_m, b)

            dt = round(time.time() - t0, 2)
            print(
                f"[suite] {spec_name:26s} seed={seed} params={n_params:,} "
                f"obs={bundle.best_val_loss:.4f} cliff={h_single['cliff_bps']:.2f}bps ({dt:.1f}s)",
                flush=True,
            )

            if do_rollout:
                roll = evaluate_autoregressive_rollout(_pred_fn, trajectories)
                r_runs.append(
                    {
                        "seed": seed,
                        "params": n_params,
                        "obs_val_std_mse": bundle.best_val_loss,
                        **h_single,
                        "horizons": roll,
                    }
                )
                print(
                    f"        -> rollout h1={roll['h1']['cum_price_rmse_bps']:.1f}bps "
                    f"h5={roll['h5']['cum_price_rmse_bps']:.1f}bps "
                    f"h12={roll['h12']['cum_price_rmse_bps']:.1f}bps",
                    flush=True,
                )

            if do_planner:
                plan = evaluate_planner(_pred_fn, planner_suite)
                p_runs.append(
                    {
                        "seed": seed,
                        "params": n_params,
                        "obs_val_std_mse": bundle.best_val_loss,
                        **h_single,
                        "planner": plan,
                    }
                )
                ov = plan["overall"]
                print(
                    f"        -> planner viol={ov['violation_rate_pct']:.1f}% "
                    f"overshoot_rmse={ov['overshoot_rmse_bps']:.1f}bps "
                    f"sizing_err={ov['sizing_rel_err_pct']:.1f}%",
                    flush=True,
                )

            del model, bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if do_rollout and r_runs:
            mean_h: dict[str, Any] = {}
            for h in HORIZONS:
                hk = f"h{h}"
                mean_h[hk] = {
                    "horizon": h,
                    "cum_price_rmse_bps": round(float(np.mean([r["horizons"][hk]["cum_price_rmse_bps"] for r in r_runs])), 2),
                    "cum_price_p95_bps": round(float(np.mean([r["horizons"][hk]["cum_price_p95_bps"] for r in r_runs])), 2),
                    "cum_liquidity_rmse": round(float(np.mean([r["horizons"][hk]["cum_liquidity_rmse"] for r in r_runs])), 4),
                }
            rollout_res["models"][spec_name] = {
                "family": family,
                "params": r_runs[0]["params"],
                "obs_val_std_mse_mean": round(float(np.mean([r["obs_val_std_mse"] for r in r_runs])), 4),
                "cliff_bps_mean": round(float(np.mean([r["cliff_bps"] for r in r_runs])), 2),
                "mean_horizons": mean_h,
                "runs": r_runs,
            }
            rollout_path.write_text(json.dumps(rollout_res, indent=2), encoding="utf-8")

        if do_planner and p_runs:
            mean_b: dict[str, Any] = {}
            for B in BUDGETS_BPS:
                bk = f"B_{int(B)}bps"
                mean_b[bk] = {
                    "budget_bps": B,
                    "violation_rate_pct": round(float(np.mean([r["planner"]["by_budget"][bk]["violation_rate_pct"] for r in p_runs])), 2),
                    "overshoot_rmse_bps": round(float(np.mean([r["planner"]["by_budget"][bk]["overshoot_rmse_bps"] for r in p_runs])), 2),
                    "overshoot_p95_bps": round(float(np.mean([r["planner"]["by_budget"][bk]["overshoot_p95_bps"] for r in p_runs])), 2),
                    "sizing_rel_err_pct": round(float(np.mean([r["planner"]["by_budget"][bk]["sizing_rel_err_pct"] for r in p_runs])), 2),
                }
            planner_res["models"][spec_name] = {
                "family": family,
                "params": p_runs[0]["params"],
                "obs_val_std_mse_mean": round(float(np.mean([r["obs_val_std_mse"] for r in p_runs])), 4),
                "cliff_bps_mean": round(float(np.mean([r["cliff_bps"] for r in p_runs])), 2),
                "overall_mean": {
                    "violation_rate_pct": round(float(np.mean([r["planner"]["overall"]["violation_rate_pct"] for r in p_runs])), 2),
                    "overshoot_rmse_bps": round(float(np.mean([r["planner"]["overall"]["overshoot_rmse_bps"] for r in p_runs])), 2),
                    "overshoot_p95_bps": round(float(np.mean([r["planner"]["overall"]["overshoot_p95_bps"] for r in p_runs])), 2),
                    "sizing_rel_err_pct": round(float(np.mean([r["planner"]["overall"]["sizing_rel_err_pct"] for r in p_runs])), 2),
                    "mean_excess_slippage_usd": round(float(np.mean([r["planner"]["overall"]["mean_excess_slippage_usd"] for r in p_runs])), 2),
                },
                "by_budget_mean": mean_b,
                "runs": p_runs,
            }
            planner_path.write_text(json.dumps(planner_res, indent=2), encoding="utf-8")

    return {"rollout": rollout_res, "planner": planner_res}


# --------------------------------------------------------------------------------------
# Experiment 3: Full Foundation-Model Scaling Ladder (0.5B -> 14B QLoRA)
# --------------------------------------------------------------------------------------
DEFAULT_LLM_LADDER: list[dict[str, Any]] = [
    {"tag": "Qwen2.5-0.5B_pretrained", "model_id": "Qwen/Qwen2.5-0.5B", "random_init": False, "load_4bit": False, "lora_r": 16},
    {"tag": "Qwen2.5-0.5B_random_init", "model_id": "Qwen/Qwen2.5-0.5B", "random_init": True, "load_4bit": False, "lora_r": 16},
    {"tag": "Qwen2.5-1.5B_pretrained", "model_id": "Qwen/Qwen2.5-1.5B", "random_init": False, "load_4bit": False, "lora_r": 16},
    {"tag": "Qwen2.5-3B_qlora_r16", "model_id": "Qwen/Qwen2.5-3B", "random_init": False, "load_4bit": True, "lora_r": 16},
    {"tag": "Qwen2.5-7B_qlora_r16", "model_id": "Qwen/Qwen2.5-7B", "random_init": False, "load_4bit": True, "lora_r": 16},
    {"tag": "Qwen2.5-7B_qlora_r64", "model_id": "Qwen/Qwen2.5-7B", "random_init": False, "load_4bit": True, "lora_r": 64},
    {"tag": "Qwen2.5-14B_qlora_r16", "model_id": "Qwen/Qwen2.5-14B", "random_init": False, "load_4bit": True, "lora_r": 16},
]


def run_llm_ladder(
    splits: dict[str, Any],
    device: torch.device,
    *,
    epochs: int = 4,
    batch_size: int = 32,
    seed: int = 101,
) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModel, BitsAndBytesConfig

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "llm_ladder.json"

    scaler = TargetScaler.from_targets(np.stack([s.target for s in splits["train"]], axis=0))
    tr = BatchTensors.from_samples(splits["train"], device)
    va = BatchTensors.from_samples(splits["val"], device)
    te = BatchTensors.from_samples(splits["test"], device)

    checkpoints = load_held_out_checkpoints(splits)
    trajectories = build_rollout_GroundTruth(checkpoints)
    planner_suite = build_planner_suite(checkpoints)

    val_y = va.y.cpu().numpy().astype(np.float64)
    copy_val_loss = float(
        np.mean(scaler.per_sample_loss(np.stack([predict_copy(s.action_eth) for s in splits["val"]]), val_y))
    )

    existing: dict[str, Any] = (
        json.loads(out_path.read_text(encoding="utf-8"))
        if out_path.exists()
        else {"copy_obs_val_std_mse": copy_val_loss, "runs": {}}
    )

    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    cdtype = torch.bfloat16 if bf16 else (torch.float16 if device.type == "cuda" else torch.float32)
    y_scale = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
    inp_stats = rv2._input_stats(tr)

    for spec in DEFAULT_LLM_LADDER:
        tag = spec["tag"]
        if tag in existing["runs"] and "error" not in existing["runs"][tag]:
            print(f"[llm_ladder] skipping already-completed {tag}", flush=True)
            continue

        model_id = spec["model_id"]
        random_init = spec["random_init"]
        load_4bit = spec["load_4bit"] and not random_init
        lora_r = spec["lora_r"]
        print(f"\n[llm_ladder] starting {tag} ({model_id}, 4bit={load_4bit}, r={lora_r})...", flush=True)
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)

            if random_init:
                cfg = AutoConfig.from_pretrained(model_id)
                backbone = AutoModel.from_config(cfg, torch_dtype=cdtype).to(device)
            elif load_4bit:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=cdtype,
                    bnb_4bit_use_double_quant=True,
                )
                backbone = AutoModel.from_pretrained(
                    model_id, quantization_config=bnb, torch_dtype=cdtype, device_map={"": 0}
                )
                backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
            else:
                backbone = AutoModel.from_pretrained(model_id, torch_dtype=cdtype).to(device)

            n_backbone = sum(p.numel() * (2 if (load_4bit and p.dtype == torch.uint8) else 1) for p in backbone.parameters())
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=2 * lora_r,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                ),
            )
            d_model = backbone.config.hidden_size
            model = rv2.LLMWorldModel(backbone, d_model, inp_stats, y_scale, cdtype)
            model.enc.to(device)
            model.head.to(device)

            lora_params = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
            opt = torch.optim.AdamW(
                [
                    {"params": lora_params, "lr": 2e-4},
                    {"params": list(model.enc.parameters()) + list(model.head.parameters()), "lr": 1e-3},
                ],
                weight_decay=1e-4,
            )
            n_tr = tr.action.shape[0]
            steps = epochs * math.ceil(n_tr / batch_size)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=[2e-4, 1e-3], total_steps=steps, pct_start=0.1
            )
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            def _predict_batch(b: BatchTensors) -> np.ndarray:
                model.eval()
                outs = []
                with torch.no_grad():
                    for i in range(0, b.action.shape[0], 256):
                        idx = torch.arange(i, min(i + 256, b.action.shape[0]), device=device)
                        with torch.autocast(device_type=device.type, dtype=cdtype, enabled=device.type == "cuda"):
                            outs.append(model(b.slice(idx)).float().cpu().numpy())
                return np.concatenate(outs).astype(np.float64)

            best_val, best_state, val_curve = float("inf"), None, []
            gen = torch.Generator().manual_seed(seed)
            t0 = time.time()
            step = 0
            for ep in range(epochs):
                model.train()
                perm = torch.randperm(n_tr, generator=gen).to(device)
                run_loss = 0.0
                for k, i in enumerate(range(0, n_tr, batch_size)):
                    sub = tr.slice(perm[i : i + batch_size])
                    with torch.autocast(device_type=device.type, dtype=cdtype, enabled=device.type == "cuda"):
                        pred = model(sub)
                    loss = torch.mean(((pred.float() - sub.y) / y_scale) ** 2)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    sched.step()
                    step += 1
                    run_loss += loss.item()
                    if k % 50 == 0:
                        sys.stdout.write(
                            f"\r    [{tag}] ep {ep+1}/{epochs} step {step}/{steps} loss={run_loss/(k+1):.4f} ({time.time()-t0:.0f}s)"
                        )
                        sys.stdout.flush()
                v = float(np.mean(scaler.per_sample_loss(_predict_batch(va), val_y)))
                val_curve.append(v)
                print(f"\n    [{tag}] ep {ep+1}: obs val std-MSE = {v:.4f} (copy={copy_val_loss:.4f})", flush=True)
                if v < best_val:
                    best_val = v
                    best_state = {n_: p.detach().cpu().clone() for n_, p in model.named_parameters() if p.requires_grad}

            if best_state is not None:
                with torch.no_grad():
                    for n_, p in model.named_parameters():
                        if n_ in best_state:
                            p.copy_(best_state[n_].to(p.device))

            def _llm_pred_samples(sample_list: list[EncodedTransition]) -> np.ndarray:
                b = BatchTensors.from_samples(sample_list, device)
                return _predict_batch(b)

            regimes = rv2.summarize(_predict_batch(te), splits["test"], scaler)
            h_single = rv2.headline(regimes)
            roll = evaluate_autoregressive_rollout(_llm_pred_samples, trajectories)
            plan = evaluate_planner(_llm_pred_samples, planner_suite)
            dt = round(time.time() - t0, 1)

            existing["runs"][tag] = {
                **spec,
                "seed": seed,
                "backbone_params": int(n_backbone),
                "trainable_params": int(n_trainable),
                "epochs": epochs,
                "obs_val_std_mse": best_val,
                "copy_obs_val_std_mse": copy_val_loss,
                "valid_beats_copy": bool(best_val < copy_val_loss),
                "val_curve": val_curve,
                "train_seconds": dt,
                **h_single,
                "rollout_horizons": roll,
                "planner_overall": plan["overall"],
                "planner_by_budget": plan["by_budget"],
            }
            out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            print(
                f"  [{tag}] DONE in {dt}s | obs={best_val:.4f} | cliff={h_single['cliff_bps']:.2f}bps "
                f"| rollout_h12={roll['h12']['cum_price_rmse_bps']:.1f}bps "
                f"| planner_viol={plan['overall']['violation_rate_pct']:.1f}%",
                flush=True,
            )
            del model, backbone, opt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:  # noqa: BLE001
            print(f"[llm_ladder] {tag} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            existing["runs"][tag] = {**spec, "error": f"{type(e).__name__}: {e}"}
            out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return existing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--which",
        choices=["rollout", "planner", "rollout_and_planner", "llm_ladder", "all"],
        default="all",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    ap.add_argument("--llm-epochs", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else default_torch_device()
    print(f"Using device: {device}", flush=True)
    splits = rv2.load_splits(args.data_dir)
    print(
        f"Loaded splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
        flush=True,
    )

    if args.which in ("rollout", "planner", "rollout_and_planner", "all"):
        run_rollout_and_planner(
            splits,
            device,
            tuple(args.seeds),
            args.epochs,
            do_rollout=args.which in ("rollout", "rollout_and_planner", "all"),
            do_planner=args.which in ("planner", "rollout_and_planner", "all"),
        )
    if args.which in ("llm_ladder", "all"):
        run_llm_ladder(splits, device, epochs=args.llm_epochs)


if __name__ == "__main__":
    main()
