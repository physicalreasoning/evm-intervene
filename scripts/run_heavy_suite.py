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

import os

OUT_DIR = Path(os.environ.get("EVM_INTERVENE_OUT_DIR", "results/heavy_suite"))
LOCAL_MIRROR_DIR = Path("results/heavy_suite")


def _write_checkpoint(filename: str, payload: dict[str, Any]) -> None:
    """Write JSON atomically to OUT_DIR (e.g. Google Drive) and mirror to local results/heavy_suite."""
    text = json.dumps(payload, indent=2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / filename
    tmp = OUT_DIR / f"{filename}.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    if OUT_DIR.resolve() != LOCAL_MIRROR_DIR.resolve():
        LOCAL_MIRROR_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_MIRROR_DIR / filename).write_text(text, encoding="utf-8")
HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 8, 12)
MAX_H = max(HORIZONS)
REAL_FLOW_HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 35, 50)
MAX_REAL_H = max(REAL_FLOW_HORIZONS)
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
    do_forecast: bool = True,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rollout_path = OUT_DIR / "rollout.json"
    planner_path = OUT_DIR / "planner.json"
    forecast_path = OUT_DIR / "forecast.json"

    scaler = TargetScaler.from_targets(np.stack([s.target for s in splits["train"]], axis=0))
    train_b = BatchTensors.from_samples(splits["train"], device)
    val_b = BatchTensors.from_samples(splits["val"], device)
    test_b = BatchTensors.from_samples(splits["test"], device)

    checkpoints = load_held_out_checkpoints(splits)
    trajectories = build_rollout_GroundTruth(checkpoints) if do_rollout else []
    planner_suite = build_planner_suite(checkpoints) if do_planner else {}
    real_trajs = build_real_flow_trajectories(splits, horizon=MAX_REAL_H, stride=60) if do_forecast else []

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
    forecast_res: dict[str, Any] = (
        json.loads(forecast_path.read_text(encoding="utf-8"))
        if (do_forecast and forecast_path.exists())
        else {
            "seeds": list(seeds),
            "epochs": epochs,
            "n_real_trajectories": len(real_trajs),
            "real_flow_rollout_models": {},
            "unconditional_volatility_forecast": {},
        }
    )

    if do_forecast and not forecast_res.get("unconditional_volatility_forecast"):
        print("[forecast] Running unconditional 20-swap volatility & air-pocket forecasters...", flush=True)
        uncond = run_unconditional_volatility_forecast(splits, device, seeds, horizon_k=20, epochs=40)
        forecast_res["unconditional_volatility_forecast"] = uncond
        _write_checkpoint("forecast.json", forecast_res)
        print(
            f"  [uncond] EWMA spearman_exc={uncond['ewma_persistence']['spearman_excursion']:.3f} | "
            f"History-Only={uncond['history_only_forecaster_mean']['spearman_excursion']:.3f} | "
            f"Spatial-TickLens={uncond['spatial_ticklens_forecaster_mean']['spearman_excursion']:.3f} "
            f"(liq_spearman={uncond['spatial_ticklens_forecaster_mean']['spearman_liq_depletion']:.3f})",
            flush=True,
        )

    for spec_name, family, kwargs in MODEL_SPECS:
        done_r = (not do_rollout) or (spec_name in rollout_res.get("models", {}))
        done_p = (not do_planner) or (spec_name in planner_res.get("models", {}))
        done_f = (not do_forecast) or (spec_name in forecast_res.get("real_flow_rollout_models", {}))
        if done_r and done_p and done_f:
            print(f"[suite] skipping already-completed {spec_name}", flush=True)
            continue
        r_runs: list[dict[str, Any]] = []
        p_runs: list[dict[str, Any]] = []
        f_runs: list[dict[str, Any]] = []
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

            if do_forecast:
                rf = evaluate_real_flow_rollout(_pred_fn, real_trajs)
                f_runs.append(
                    {
                        "seed": seed,
                        "params": n_params,
                        "obs_val_std_mse": bundle.best_val_loss,
                        **h_single,
                        "real_flow": rf,
                    }
                )
                print(
                    f"        -> forecast 50-swap real flow: all={rf['all_trajectories']['h50']['price_rmse_bps']:.1f}bps "
                    f"cliff={rf['high_vol_cliff_trajectories']['h50']['price_rmse_bps']:.1f}bps "
                    f"dir_acc={rf['all_trajectories']['h50']['directional_acc_pct']:.1f}%",
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
            _write_checkpoint("rollout.json", rollout_res)

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
            _write_checkpoint("planner.json", planner_res)

        if do_forecast and f_runs:
            def _avg_horizon_dict(subset_key: str, _fr=f_runs) -> dict[str, Any]:
                out_h: dict[str, Any] = {}
                for h in REAL_FLOW_HORIZONS:
                    hk = f"h{h}"
                    out_h[hk] = {
                        "horizon_swaps": h,
                        "price_rmse_bps": round(float(np.mean([r["real_flow"][subset_key][hk]["price_rmse_bps"] for r in _fr])), 2),
                        "liquidity_rmse": round(float(np.mean([r["real_flow"][subset_key][hk]["liquidity_rmse"] for r in _fr])), 4),
                        "directional_acc_pct": round(float(np.mean([r["real_flow"][subset_key][hk]["directional_acc_pct"] for r in _fr])), 2),
                    }
                return out_h

            forecast_res["real_flow_rollout_models"][spec_name] = {
                "family": family,
                "params": f_runs[0]["params"],
                "obs_val_std_mse_mean": round(float(np.mean([r["obs_val_std_mse"] for r in f_runs])), 4),
                "cliff_bps_mean": round(float(np.mean([r["cliff_bps"] for r in f_runs])), 2),
                "all_trajectories_mean": _avg_horizon_dict("all_trajectories"),
                "high_vol_cliff_trajectories_mean": _avg_horizon_dict("high_vol_cliff_trajectories"),
                "runs": f_runs,
            }
            _write_checkpoint("forecast.json", forecast_res)

    return {"rollout": rollout_res, "planner": planner_res, "forecast": forecast_res}


# --------------------------------------------------------------------------------------
# Experiment 4: Real Historical Mainnet Forecasting (Open-Loop 50-Swap Rollout +
#               Unconditional Forward Volatility & Air-Pocket Forecasting)
# --------------------------------------------------------------------------------------
def _build_real_stream_with_states(
    swaps: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    all_borrowers: list[BorrowerPosition] | None = None,
    *,
    min_block: int = 0,
    max_block: int = 10**12,
    window_tag: str = "",
) -> list[dict[str, Any]]:
    """Build contiguous real Mainnet swap transitions preserving exact PoolState and history buffers."""
    from evm_intervene.constants import USDC_DECIMALS, WETH_DECIMALS
    from evm_intervene.simulator import eth_usd_from_raw_sqrt_price, raw_sqrt_price_from_x96
    from evm_intervene.state import summarize_swap_history

    cp_blocks = [int(c["block_number"]) for c in checkpoints]
    records: list[dict[str, Any]] = []
    hist_actions: list[float] = []
    hist_dlogp: list[float] = []
    running_min_eth = float("inf")

    for i in range(1, len(swaps)):
        prev_s = swaps[i - 1]
        cur_s = swaps[i]
        b_cur = int(cur_s["block_number"])

        q_prev = raw_sqrt_price_from_x96(int(prev_s["sqrt_price_x96"]))
        q_cur = raw_sqrt_price_from_x96(int(cur_s["sqrt_price_x96"]))
        p_prev = eth_usd_from_raw_sqrt_price(q_prev)
        p_cur = eth_usd_from_raw_sqrt_price(q_cur)
        running_min_eth = min(running_min_eth, p_prev)

        amt0 = int(cur_s["amount0_raw"])
        amt1 = int(cur_s["amount1_raw"])
        if amt1 > 0:
            action_eth = amt1 / (10.0**WETH_DECIMALS)
            usdc_delta = -amt0 / (10.0**USDC_DECIMALS)
            weth_delta = action_eth
        elif amt0 > 0:
            usdc_in = amt0 / (10.0**USDC_DECIMALS)
            action_eth = -(usdc_in / p_prev)
            usdc_delta = -usdc_in
            weth_delta = amt1 / (10.0**WETH_DECIMALS)
        else:
            continue

        dlogp = float(np.log(p_cur / p_prev))
        act_buf = np.array(hist_actions[-16:], dtype=np.float64)
        dlp_buf = np.array(hist_dlogp[-16:], dtype=np.float64)
        hist_actions.append(action_eth)
        hist_dlogp.append(dlogp)

        if b_cur < min_block or b_cur > max_block:
            continue
        if b_cur - int(prev_s["block_number"]) > 25:
            continue
        prev_tick = int(prev_s["tick"])
        cur_tick = int(cur_s["tick"])
        prev_liq = float(int(prev_s["liquidity"]))
        cur_liq = float(int(cur_s["liquidity"]))
        if prev_tick == cur_tick and prev_liq != cur_liq:
            continue
        if abs(action_eth) < 1e-4:
            continue

        cp_idx = max(0, int(np.searchsorted(cp_blocks, b_cur, side="right")) - 1)
        cp = checkpoints[cp_idx]
        active_borrowers: tuple[BorrowerPosition, ...] = ()
        if all_borrowers:
            active_borrowers = tuple(
                b for b in all_borrowers if b.critical_eth_price < running_min_eth * 0.999
            )

        state = PoolState(
            block_number=b_cur,
            sqrt_price_raw=q_prev,
            tick=prev_tick,
            liquidity=prev_liq,
            ticks=tuple(int(t) for t in cp["ticks"]),
            liquidity_net=tuple(float(int(ln)) for ln in cp["liquidity_net"]),
            borrowers=active_borrowers,
        )
        records.append(
            {
                "state": state,
                "action_eth": action_eth,
                "dlogp": dlogp,
                "dlogl": math.log(max(cur_liq, 1.0) / max(prev_liq, 1.0)),
                "post_eth_usd": p_cur,
                "post_liquidity": max(cur_liq, 1.0),
                "ticks_crossed": abs(cur_tick - prev_tick) // 10,
                "act_buf": act_buf,
                "dlp_buf": dlp_buf,
                "window_tag": window_tag,
            }
        )
    return records


def build_real_flow_trajectories(
    splits: dict[str, Any],
    horizon: int = MAX_REAL_H,
    stride: int = 60,
) -> list[dict[str, Any]]:
    """Extract held-out 50-swap real Mainnet trajectories from Aug 5 crash and Oct 2024."""
    swaps = splits["swaps"]
    cps = splits["cps"]
    borrowers = splits["borrowers"]

    streams = [
        _build_real_stream_with_states(
            swaps["crash_aug5_2024"], cps["crash_aug5_2024"], borrowers,
            min_block=20_458_000, max_block=20_460_400, window_tag="crash_onset",
        ),
        _build_real_stream_with_states(
            swaps["crash_aug5_2024"], cps["crash_aug5_2024"], borrowers,
            min_block=20_465_001, max_block=20_468_000, window_tag="crash_late",
        ),
        _build_real_stream_with_states(
            swaps["autumn_oct_2024"], cps["autumn_oct_2024"], None,
            window_tag="oct_2024",
        ),
    ]
    dummy_outcome = TransitionOutcome(
        pre_eth_usd=3000.0, post_amm_eth_usd=3000.0, post_eth_usd=3000.0,
        pre_liquidity=1e19, post_liquidity=1e19, post_sqrt_price_raw=1.0,
        post_tick=0, action_eth_signed=0.0, usdc_delta=0.0, weth_delta=0.0,
        ticks_crossed=0, cascade_rounds=0, liquidated_eth=0.0, liquidated_accounts=0,
    )
    trajs: list[dict[str, Any]] = []
    for stream in streams:
        for start in range(0, len(stream) - horizon, stride):
            seg = stream[start : start + horizon]
            p0 = seg[0]["state"].eth_usd
            l0 = max(seg[0]["state"].liquidity, 1.0)
            true_cum_dlogp = np.array([math.log(r["post_eth_usd"] / p0) for r in seg], dtype=np.float64)
            true_cum_dlogl = np.array([math.log(r["post_liquidity"] / l0) for r in seg], dtype=np.float64)
            total_crossed = sum(int(r["ticks_crossed"]) for r in seg)
            trajs.append(
                {
                    "init_state": seg[0]["state"],
                    "init_act_buf": list(seg[0]["act_buf"]),
                    "init_dlp_buf": list(seg[0]["dlp_buf"]),
                    "actions": [float(r["action_eth"]) for r in seg],
                    "true_cum_dlogp": true_cum_dlogp,
                    "true_cum_dlogl": true_cum_dlogl,
                    "total_ticks_crossed": total_crossed,
                    "is_high_vol_cliff": bool(total_crossed >= 8 or np.max(np.abs(true_cum_dlogp)) >= 0.004),
                    "window_tag": seg[0]["window_tag"],
                    "dummy_outcome": dummy_outcome,
                }
            )
    return trajs


@torch.no_grad()
def evaluate_real_flow_rollout(
    predict_fn,
    trajs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Open-loop 50-step forecast along real held-out Mainnet swap sequences."""
    from evm_intervene.state import summarize_swap_history

    n_traj = len(trajs)
    cur_states = [t["init_state"] for t in trajs]
    act_bufs = [list(t["init_act_buf"]) for t in trajs]
    dlp_bufs = [list(t["init_dlp_buf"]) for t in trajs]
    dummy = trajs[0]["dummy_outcome"]

    pred_cum_dlogp = np.zeros((n_traj, MAX_REAL_H), dtype=np.float64)
    pred_cum_dlogl = np.zeros((n_traj, MAX_REAL_H), dtype=np.float64)

    for step in range(MAX_REAL_H):
        step_samples: list[EncodedTransition] = []
        for i in range(n_traj):
            h_sum = summarize_swap_history(
                np.array(act_bufs[i][-16:], dtype=np.float64),
                np.array(dlp_bufs[i][-16:], dtype=np.float64),
            )
            a_t = trajs[i]["actions"][step]
            step_samples.append(
                encode_state_action(cur_states[i], a_t, dummy, h_sum, regime="observational")
            )
        preds = predict_fn(step_samples)
        for i in range(n_traj):
            dl_p = float(np.clip(preds[i, 0], -0.25, 0.25))
            dl_l = float(np.clip(preds[i, 1], -4.0, 4.0))
            prev_p = 0.0 if step == 0 else pred_cum_dlogp[i, step - 1]
            prev_l = 0.0 if step == 0 else pred_cum_dlogl[i, step - 1]
            pred_cum_dlogp[i, step] = prev_p + dl_p
            pred_cum_dlogl[i, step] = prev_l + dl_l
            cur_states[i] = advance_predicted_pool_state(cur_states[i], dl_p, dl_l)
            act_bufs[i].append(trajs[i]["actions"][step])
            dlp_bufs[i].append(dl_p)

    true_dlogp = np.stack([t["true_cum_dlogp"] for t in trajs], axis=0)
    true_dlogl = np.stack([t["true_cum_dlogl"] for t in trajs], axis=0)
    cliff_mask = np.array([t["is_high_vol_cliff"] for t in trajs], dtype=bool)

    def _metrics_for_mask(mask: np.ndarray) -> dict[str, Any]:
        out_h: dict[str, Any] = {}
        for h in REAL_FLOW_HORIZONS:
            idx = h - 1
            p_pred = pred_cum_dlogp[mask, idx]
            p_true = true_dlogp[mask, idx]
            l_pred = pred_cum_dlogl[mask, idx]
            l_true = true_dlogl[mask, idx]
            err_bps = (p_pred - p_true) * 1e4
            nz = np.abs(p_true) > 1e-5
            dir_acc = float(np.mean(np.sign(p_pred[nz]) == np.sign(p_true[nz])) * 100.0) if np.any(nz) else 50.0
            out_h[f"h{h}"] = {
                "horizon_swaps": h,
                "price_rmse_bps": round(float(np.sqrt(np.mean(err_bps**2))), 2),
                "price_mae_bps": round(float(np.mean(np.abs(err_bps))), 2),
                "liquidity_rmse": round(float(np.sqrt(np.mean((l_pred - l_true) ** 2))), 4),
                "directional_acc_pct": round(dir_acc, 2),
            }
        return out_h

    return {
        "n_trajectories_all": int(n_traj),
        "n_trajectories_high_vol_cliff": int(np.sum(cliff_mask)),
        "all_trajectories": _metrics_for_mask(np.ones(n_traj, dtype=bool)),
        "high_vol_cliff_trajectories": _metrics_for_mask(cliff_mask),
    }


def run_unconditional_volatility_forecast(
    splits: dict[str, Any],
    device: torch.device,
    seeds: tuple[int, ...],
    horizon_k: int = 20,
    epochs: int = 40,
) -> dict[str, Any]:
    """Forecast next-20-swap realized volatility, max price excursion, and liquidity cliff depletion
    WITHOUT seeing future actions: compares EWMA persistence, History-Only MLP, and Spatial TickLens MLP."""
    from scipy.stats import spearmanr

    def _make_xy(samples: list[EncodedTransition]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_spatial: list[np.ndarray] = []
        x_hist_only: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        ewma_list: list[np.ndarray] = []

        dlogps = np.array([s.target[0] for s in samples], dtype=np.float64)
        dlogls = np.array([s.target[1] for s in samples], dtype=np.float64)

        for t in range(len(samples) - horizon_k):
            s = samples[t]
            fut_p = dlogps[t + 1 : t + 1 + horizon_k]
            fut_l = dlogls[t + 1 : t + 1 + horizon_k]
            cum_p = np.cumsum(fut_p)
            cum_l = np.cumsum(fut_l)

            realized_vol_bps = float(np.sqrt(np.sum(fut_p**2)) * 1e4)
            max_excursion_bps = float(np.max(np.abs(cum_p)) * 1e4)
            max_liq_shift = float(np.max(np.abs(cum_l)))

            # History-only features: spot log-price, log-liquidity, and 6-D swap history H_t (8-D, blind to TickLens)
            feat_hist = np.concatenate([s.global_scalars[:2], s.global_scalars[6:12]])

            # Spatial TickLens + Aave summary features extending history-only with compact boundary depth (18-D)
            rl = np.log(np.clip(s.bin_rel_liquidity, 1e-4, 1e4))
            ec = np.log1p(s.bin_eth_capacity)
            feat_spatial = np.concatenate(
                [
                    feat_hist,
                    s.global_scalars[2:6],  # in-tick sell/buy caps, near Aave liq supply, min downside rel_liq
                    np.array(
                        [
                            float(np.mean(rl[:4])),    # downside 0..50bps rel liquidity
                            float(np.mean(rl[4:10])),  # downside 50..300bps rel liquidity
                            float(np.mean(rl[16:20])), # upside 0..50bps rel liquidity
                            float(np.mean(rl[20:26])), # upside 50..300bps rel liquidity
                            float(np.mean(ec[:6])),    # downside near ETH capacity
                            float(np.mean(ec[16:22])), # upside near ETH capacity
                        ],
                        dtype=np.float64,
                    ),
                ]
            )

            # EWMA persistence baseline from recent 16-swap realized vol (stored in global_scalars[8] = std * 1e3)
            past_step_std = s.global_scalars[8] * 1e-3
            ewma_vol_bps = float(past_step_std * math.sqrt(horizon_k) * 1e4)
            ewma_exc_bps = float(max(ewma_vol_bps, s.global_scalars[9] * 10.0))

            x_spatial.append(feat_spatial)
            x_hist_only.append(feat_hist)
            y_list.append(np.array([realized_vol_bps, max_excursion_bps, max_liq_shift], dtype=np.float64))
            ewma_list.append(np.array([ewma_vol_bps, ewma_exc_bps, 0.0], dtype=np.float64))

        return (
            np.stack(x_spatial, axis=0),
            np.stack(x_hist_only, axis=0),
            np.stack(y_list, axis=0),
            np.stack(ewma_list, axis=0),
        )

    swaps = splits["swaps"]
    cps = splits["cps"]
    borrowers = splits["borrowers"]
    obs = rx.build_real_observational_transitions
    oot_test = obs(
        swaps["crash_aug5_2024"], cps["crash_aug5_2024"], all_borrowers=borrowers,
        min_block=20_458_000, max_block=20_460_400,
    ) + obs(
        swaps["crash_aug5_2024"], cps["crash_aug5_2024"], all_borrowers=borrowers,
        min_block=20_465_001, max_block=20_468_000,
    ) + obs(
        swaps["autumn_oct_2024"], cps["autumn_oct_2024"],
    )

    xs_tr, xh_tr, y_tr, _ = _make_xy(splits["train"])
    xs_va, xh_va, y_va, _ = _make_xy(splits["val"])
    xs_te, xh_te, y_te, ewma_te = _make_xy(oot_test)

    y_log_tr = np.log1p(y_tr)
    y_log_va = np.log1p(y_va)
    y_mu = y_log_tr.mean(axis=0)
    y_sd = y_log_tr.std(axis=0) + 1e-6

    def _score_preds(pred_raw: np.ndarray) -> dict[str, float]:
        vol_rmse = float(np.sqrt(np.mean((pred_raw[:, 0] - y_te[:, 0]) ** 2)))
        exc_rmse = float(np.sqrt(np.mean((pred_raw[:, 1] - y_te[:, 1]) ** 2)))
        liq_rmse = float(np.sqrt(np.mean((pred_raw[:, 2] - y_te[:, 2]) ** 2)))
        rho_vol, _ = spearmanr(pred_raw[:, 0], y_te[:, 0])
        rho_exc, _ = spearmanr(pred_raw[:, 1], y_te[:, 1])
        rho_liq, _ = spearmanr(pred_raw[:, 2], y_te[:, 2])
        # Top-10% tail air-pocket recall
        q90_true = np.percentile(y_te[:, 1], 90)
        q90_pred = np.percentile(pred_raw[:, 1], 90)
        tail_recall = float(np.mean(pred_raw[y_te[:, 1] >= q90_true, 1] >= q90_pred) * 100.0)
        return {
            "vol_rmse_bps": round(vol_rmse, 2),
            "excursion_rmse_bps": round(exc_rmse, 2),
            "liq_depletion_rmse": round(liq_rmse, 4),
            "spearman_vol": round(float(rho_vol), 4),
            "spearman_excursion": round(float(rho_exc), 4),
            "spearman_liq_depletion": round(float(0.0 if np.isnan(rho_liq) else rho_liq), 4),
            "top10pct_air_pocket_recall_pct": round(tail_recall, 2),
        }

    def _train_forecaster(x_tr: np.ndarray, x_va: np.ndarray, x_te: np.ndarray, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_mu = x_tr.mean(axis=0)
        x_sd = x_tr.std(axis=0) + 1e-5
        xt_tr = torch.tensor((x_tr - x_mu) / x_sd, dtype=torch.float32, device=device)
        xt_va = torch.tensor((x_va - x_mu) / x_sd, dtype=torch.float32, device=device)
        xt_te = torch.tensor((x_te - x_mu) / x_sd, dtype=torch.float32, device=device)
        yt_tr = torch.tensor((y_log_tr - y_mu) / y_sd, dtype=torch.float32, device=device)
        yt_va = torch.tensor((y_log_va - y_mu) / y_sd, dtype=torch.float32, device=device)

        net = nn.Sequential(
            nn.Linear(x_tr.shape[1], 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 3),
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        best_v = float("inf")
        best_w = None
        n = xt_tr.shape[0]
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, device=device)
            for s in range(0, n, 256):
                idx = perm[s : s + 256]
                loss = torch.mean((net(xt_tr[idx]) - yt_tr[idx]) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(torch.mean((net(xt_va) - yt_va) ** 2).item())
                if v < best_v:
                    best_v = v
                    best_w = {k: v_.detach().cpu().clone() for k, v_ in net.state_dict().items()}
        if best_w is not None:
            net.load_state_dict(best_w)
        net.eval()
        with torch.no_grad():
            pred_std = net(xt_te).cpu().numpy()
        pred_log = pred_std * y_sd + y_mu
        return np.expm1(np.clip(pred_log, 0.0, 12.0))

    hist_runs = [_score_preds(_train_forecaster(xh_tr, xh_va, xh_te, s)) for s in seeds]
    spat_runs = [_score_preds(_train_forecaster(xs_tr, xs_va, xs_te, s)) for s in seeds]

    def _avg(runs: list[dict[str, float]]) -> dict[str, float]:
        return {k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0]}

    return {
        "horizon_swaps": horizon_k,
        "n_oot_samples": int(len(y_te)),
        "ewma_persistence": _score_preds(ewma_te),
        "history_only_forecaster_mean": _avg(hist_runs),
        "spatial_ticklens_forecaster_mean": _avg(spat_runs),
        "history_only_runs": hist_runs,
        "spatial_ticklens_runs": spat_runs,
    }


def run_forecast_suite(
    splits: dict[str, Any],
    device: torch.device,
    seeds: tuple[int, ...],
    epochs: int,
) -> dict[str, Any]:
    """Run both 50-step real order-flow trajectory forecasting and 20-step unconditional volatility forecasting."""
    forecast_path = OUT_DIR / "forecast.json"
    scaler = TargetScaler.from_targets(np.stack([s.target for s in splits["train"]], axis=0))
    train_b = BatchTensors.from_samples(splits["train"], device)
    val_b = BatchTensors.from_samples(splits["val"], device)

    real_trajs = build_real_flow_trajectories(splits, horizon=MAX_REAL_H, stride=60)
    print(
        f"[forecast] Built {len(real_trajs)} real {MAX_REAL_H}-swap held-out Mainnet trajectories "
        f"({sum(t['is_high_vol_cliff'] for t in real_trajs)} high-vol cliff trajectories)",
        flush=True,
    )

    res: dict[str, Any] = (
        json.loads(forecast_path.read_text(encoding="utf-8"))
        if forecast_path.exists()
        else {
            "seeds": list(seeds),
            "epochs": epochs,
            "n_real_trajectories": len(real_trajs),
            "real_flow_rollout_models": {},
            "unconditional_volatility_forecast": {},
        }
    )

    if not res.get("unconditional_volatility_forecast"):
        print("[forecast] Running unconditional 20-swap volatility & air-pocket forecasters...", flush=True)
        uncond = run_unconditional_volatility_forecast(splits, device, seeds, horizon_k=20, epochs=40)
        res["unconditional_volatility_forecast"] = uncond
        _write_checkpoint("forecast.json", res)
        print(
            f"  [uncond] EWMA spearman_exc={uncond['ewma_persistence']['spearman_excursion']:.3f} | "
            f"History-Only={uncond['history_only_forecaster_mean']['spearman_excursion']:.3f} | "
            f"Spatial-TickLens={uncond['spatial_ticklens_forecaster_mean']['spearman_excursion']:.3f} "
            f"(liq_spearman={uncond['spatial_ticklens_forecaster_mean']['spearman_liq_depletion']:.3f})",
            flush=True,
        )

    for spec_name, family, kwargs in MODEL_SPECS:
        if spec_name in res.get("real_flow_rollout_models", {}):
            print(f"[forecast] skipping already-completed {spec_name}", flush=True)
            continue

        runs: list[dict[str, Any]] = []
        for seed in seeds:
            t0 = time.time()
            bundle = train_world_model(
                family, train_b, val_b, scaler, seed=seed, epochs=epochs,
                batch_size=512, lr=1.5e-3, model_kwargs=kwargs,
            )
            model = bundle.best_model
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            def _pred_fn(sample_list: list[EncodedTransition], _m=model) -> np.ndarray:
                b = BatchTensors.from_samples(sample_list, device)
                return evaluate_model_predictions(_m, b)

            rf = evaluate_real_flow_rollout(_pred_fn, real_trajs)
            dt = round(time.time() - t0, 1)
            runs.append(
                {
                    "seed": seed,
                    "params": n_params,
                    "obs_val_std_mse": bundle.best_val_loss,
                    "real_flow": rf,
                }
            )
            print(
                f"[forecast] {spec_name:26s} seed={seed} ({dt:.1f}s) | "
                f"real 50-swap RMSE: all={rf['all_trajectories']['h50']['price_rmse_bps']:.2f}bps "
                f"cliff={rf['high_vol_cliff_trajectories']['h50']['price_rmse_bps']:.2f}bps "
                f"dir_acc={rf['all_trajectories']['h50']['directional_acc_pct']:.1f}%",
                flush=True,
            )
            del model, bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _avg_horizon_dict(subset_key: str) -> dict[str, Any]:
            out_h: dict[str, Any] = {}
            for h in REAL_FLOW_HORIZONS:
                hk = f"h{h}"
                out_h[hk] = {
                    "horizon_swaps": h,
                    "price_rmse_bps": round(float(np.mean([r["real_flow"][subset_key][hk]["price_rmse_bps"] for r in runs])), 2),
                    "liquidity_rmse": round(float(np.mean([r["real_flow"][subset_key][hk]["liquidity_rmse"] for r in runs])), 4),
                    "directional_acc_pct": round(float(np.mean([r["real_flow"][subset_key][hk]["directional_acc_pct"] for r in runs])), 2),
                }
            return out_h

        res["real_flow_rollout_models"][spec_name] = {
            "family": family,
            "params": runs[0]["params"],
            "obs_val_std_mse_mean": round(float(np.mean([r["obs_val_std_mse"] for r in runs])), 4),
            "all_trajectories_mean": _avg_horizon_dict("all_trajectories"),
            "high_vol_cliff_trajectories_mean": _avg_horizon_dict("high_vol_cliff_trajectories"),
            "runs": runs,
        }
        _write_checkpoint("forecast.json", res)

    return res


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
    real_trajs = build_real_flow_trajectories(splits, horizon=MAX_REAL_H, stride=60)

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
            real_flow = evaluate_real_flow_rollout(_llm_pred_samples, real_trajs)
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
                "real_flow_rollout": real_flow,
            }
            _write_checkpoint("llm_ladder.json", existing)
            print(
                f"  [{tag}] DONE in {dt}s | obs={best_val:.4f} | cliff={h_single['cliff_bps']:.2f}bps "
                f"| rollout_h12={roll['h12']['cum_price_rmse_bps']:.1f}bps "
                f"| real_h50={real_flow['all_trajectories']['h50']['price_rmse_bps']:.1f}bps "
                f"| planner_viol={plan['overall']['violation_rate_pct']:.1f}% -> saved to {OUT_DIR}",
                flush=True,
            )
            del model, backbone, opt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:  # noqa: BLE001
            print(f"[llm_ladder] {tag} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            existing["runs"][tag] = {**spec, "error": f"{type(e).__name__}: {e}"}
            _write_checkpoint("llm_ladder.json", existing)

    return existing


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--which",
        choices=["rollout", "planner", "forecast", "rollout_and_planner", "llm_ladder", "all"],
        default="all",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    ap.add_argument("--llm-epochs", type=int, default=4)
    args = ap.parse_args()

    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints & results will be saved to: {OUT_DIR.resolve()}", flush=True)

    device = torch.device(args.device) if args.device else default_torch_device()
    print(f"Using device: {device}", flush=True)
    splits = rv2.load_splits(args.data_dir)
    print(
        f"Loaded splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
        flush=True,
    )

    if args.which in ("rollout", "planner", "forecast", "rollout_and_planner", "all"):
        run_rollout_and_planner(
            splits,
            device,
            tuple(args.seeds),
            args.epochs,
            do_rollout=args.which in ("rollout", "rollout_and_planner", "all"),
            do_planner=args.which in ("planner", "rollout_and_planner", "all"),
            do_forecast=args.which in ("forecast", "rollout_and_planner", "all"),
        )
    if args.which in ("llm_ladder", "all"):
        run_llm_ladder(splits, device, epochs=args.llm_epochs)


if __name__ == "__main__":
    main()
