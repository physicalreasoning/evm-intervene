"""Main pre-registered experiment for `evm-intervene` (`docs/DESIGN.md`).

Trains 4 world-model families x 5 random seeds strictly on historical Ethereum Mainnet
observational swaps, evaluates both 5%-matched validation checkpoints (Gate G1) and
best-validation checkpoints (Gate G2) on counterfactual interventions across held-out
future archive states (Aug 5, 2024 crash + Oct 2024 autumn), and runs the crash-exposed
training sensitivity ablation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evm_intervene.baselines import (
    RandomEncoderRidge,
    predict_amm_only_exact,
    predict_copy,
    predict_local_linear,
    predict_single_tick_xyk,
)
from evm_intervene.constants import (
    BOOTSTRAP_REPLICATES,
    MATCHED_LOSS_REL_TOL,
    RANDOM_SEEDS,
    USDC_DECIMALS,
    WETH_DECIMALS,
)
from evm_intervene.metrics import (
    TargetScaler,
    cluster_bootstrap_mean_ci,
    cluster_bootstrap_paired_gap,
    holm_bonferroni,
    liquidity_rmse,
    passes_matched_loss_screen,
    price_rmse_bps,
    spearman_rank_ci,
)
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
    eth_usd_from_raw_sqrt_price,
    raw_sqrt_price_from_x96,
    simulate_intervention,
)
from evm_intervene.state import (
    EncodedTransition,
    classify_intervention_regime,
    encode_state_action,
    summarize_swap_history,
)

MODEL_FAMILIES: tuple[str, ...] = (
    "mlp_direct",
    "seq_transformer",
    "latent_jepa",
    "conservation_wm",
)

# 24 counterfactual action multipliers (in multiples of local in-tick capacity + ETH stress grid)
# spanning Regime I (in-tick smooth), Regime II (multi-tick cliffs), and Regime III (Aave cascades).
COUNTERFACTUAL_MULTIPLIERS: tuple[float, ...] = (
    -6.0,
    -4.0,
    -2.5,
    -1.5,
    -0.80,
    -0.45,
    -0.20,
    0.15,
    0.35,
    0.60,
    0.80,
    1.15,
    1.60,
    2.20,
    3.00,
    4.20,
    5.80,
    8.00,
)
FIXED_ETH_SHOCKS: tuple[float, ...] = (-1500.0, -600.0, 400.0, 900.0, 1800.0, 3200.0)


def _checkpoint_to_pool_state(
    cp: dict[str, Any],
    borrowers: tuple[BorrowerPosition, ...] = (),
) -> PoolState:
    return PoolState(
        block_number=int(cp["block_number"]),
        sqrt_price_raw=raw_sqrt_price_from_x96(int(cp["sqrt_price_x96"])),
        tick=int(cp["tick"]),
        liquidity=float(int(cp["liquidity"])),
        ticks=tuple(int(t) for t in cp["ticks"]),
        liquidity_net=tuple(float(int(ln)) for ln in cp["liquidity_net"]),
        borrowers=borrowers,
    )


def _load_aave_borrowers(aave_raw: list[dict[str, Any]]) -> list[BorrowerPosition]:
    return [
        BorrowerPosition(
            collateral_eth=float(b["collateral_eth"]),
            debt_usdc=float(b["debt_usdc"]),
            liquidation_threshold=float(b["liquidation_threshold"]),
        )
        for b in aave_raw
    ]


def build_real_observational_transitions(
    swaps: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    all_borrowers: list[BorrowerPosition] | None = None,
    *,
    min_block: int = 0,
    max_block: int = 10**12,
    max_samples: int | None = None,
) -> list[EncodedTransition]:
    """Encode real on-chain Mainnet Swap events with strictly causal TickLens & history context."""
    cp_blocks = [int(c["block_number"]) for c in checkpoints]
    samples: list[EncodedTransition] = []

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

        # Keep causal rolling history updated before encoding.
        hist_summary = summarize_swap_history(
            np.array(hist_actions[-16:], dtype=np.float64),
            np.array(hist_dlogp[-16:], dtype=np.float64),
        )
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

        real_outcome = TransitionOutcome(
            pre_eth_usd=p_prev,
            post_amm_eth_usd=p_cur,
            post_eth_usd=p_cur,
            pre_liquidity=prev_liq,
            post_liquidity=cur_liq,
            post_sqrt_price_raw=q_cur,
            post_tick=cur_tick,
            action_eth_signed=action_eth,
            usdc_delta=usdc_delta,
            weth_delta=weth_delta,
            ticks_crossed=abs(cur_tick - prev_tick) // 10,
            cascade_rounds=0,
            liquidated_eth=0.0,
            liquidated_accounts=0,
        )
        samples.append(
            encode_state_action(state, action_eth, real_outcome, hist_summary, regime="observational")
        )

    if max_samples is not None and len(samples) > max_samples:
        # Deterministic uniform stride across time to preserve full temporal span.
        stride_idx = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
        samples = [samples[int(k)] for k in stride_idx]

    return samples


def build_counterfactual_suite(
    checkpoints: list[dict[str, Any]],
    swaps: list[dict[str, Any]],
    all_borrowers: list[BorrowerPosition] | None = None,
    *,
    min_block: int = 0,
    max_block: int = 10**12,
    stride: int = 1,
    window_tag: str = "",
) -> tuple[list[EncodedTransition], list[PoolState], list[str]]:
    """Reset archive checkpoints and execute 24 exact counterfactual actions per state."""
    swap_blocks = np.array([int(s["block_number"]) for s in swaps], dtype=np.int64)
    samples: list[EncodedTransition] = []
    states_ref: list[PoolState] = []
    tags: list[str] = []

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

        state = _checkpoint_to_pool_state(cp, borrowers=active_borrowers)

        # Causal swap history right before checkpoint block b_num.
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

        sell_cap = min(max(state.max_in_tick_eth_sell(), 25.0), 800.0)
        buy_cap = min(max(state.max_in_tick_eth_buy_equiv(), 25.0), 800.0)

        actions: list[float] = []
        for m in COUNTERFACTUAL_MULTIPLIERS:
            actions.append(m * sell_cap if m >= 0 else m * buy_cap)
        actions.extend(FIXED_ETH_SHOCKS)

        for a_eth in actions:
            outcome = simulate_intervention(state, a_eth, enable_liquidations=True)
            reg = classify_intervention_regime(state, outcome)
            samples.append(encode_state_action(state, a_eth, outcome, hist_summary, regime=reg))
            states_ref.append(state)
            tags.append(window_tag)

    return samples, states_ref, tags


def _summarize_predictions(
    preds: np.ndarray,
    y_true: np.ndarray,
    regimes: np.ndarray,
    clusters: np.ndarray,
    scaler: TargetScaler,
) -> dict[str, Any]:
    per_loss = scaler.per_sample_loss(preds, y_true)
    out: dict[str, Any] = {}
    masks = {
        "all": np.ones(len(y_true), dtype=bool),
        "regime_1_smooth": regimes == "regime_1_smooth",
        "regime_2_cliff": regimes == "regime_2_cliff",
        "regime_3_cascade": regimes == "regime_3_cascade",
        "cliff_and_cascade": np.isin(regimes, ["regime_2_cliff", "regime_3_cascade"]),
    }
    for key, m in masks.items():
        if not np.any(m):
            continue
        mean_l, lo_l, hi_l = cluster_bootstrap_mean_ci(per_loss[m], clusters[m], n_boot=2000)
        out[key] = {
            "n": int(np.sum(m)),
            "std_mse": round(mean_l, 4),
            "std_mse_ci95": [round(lo_l, 4), round(hi_l, 4)],
            "price_rmse_bps": round(price_rmse_bps(preds[m], y_true[m]), 2),
            "liquidity_rmse": round(liquidity_rmse(preds[m], y_true[m]), 4),
        }
    return out


def run_arm(
    arm_name: str,
    train_obs: list[EncodedTransition],
    val_obs: list[EncodedTransition],
    oot_obs: list[EncodedTransition],
    int_samples: list[EncodedTransition],
    int_states: list[PoolState],
    *,
    device: torch.device,
    epochs: int = 50,
) -> dict[str, Any]:
    print(
        f"\n=== Running Arm: {arm_name} | train={len(train_obs)} val={len(val_obs)} "
        f"oot_obs={len(oot_obs)} interventions={len(int_samples)} on {device} ===",
        flush=True,
    )
    scaler = TargetScaler.from_targets(np.stack([s.target for s in train_obs], axis=0))

    train_b = BatchTensors.from_samples(train_obs, device)
    val_b = BatchTensors.from_samples(val_obs, device)
    oot_b = BatchTensors.from_samples(oot_obs, device)
    int_b = BatchTensors.from_samples(int_samples, device)

    val_y = np.stack([s.target for s in val_obs], axis=0)
    oot_y = np.stack([s.target for s in oot_obs], axis=0)
    int_y = np.stack([s.target for s in int_samples], axis=0)
    regimes = np.array([s.regime for s in int_samples])
    clusters = np.array([s.block_number for s in int_samples])
    val_clusters = np.array([s.block_number for s in val_obs])

    # 1. Analytical & Untrained Baselines.
    baselines_out: dict[str, Any] = {}
    baseline_preds_int: dict[str, np.ndarray] = {
        "copy": np.stack([predict_copy(s.action_eth) for s in int_samples], axis=0),
        "local_linear": np.stack(
            [predict_local_linear(s.pre_eth_usd, s.pre_liquidity, s.action_eth) for s in int_samples],
            axis=0,
        ),
        "single_tick_xyk": np.stack(
            [
                predict_single_tick_xyk(st, s.action_eth)
                for st, s in zip(int_states, int_samples, strict=True)
            ],
            axis=0,
        ),
        "amm_only_exact": np.stack(
            [predict_amm_only_exact(st, s.action_eth) for st, s in zip(int_states, int_samples, strict=True)],
            axis=0,
        ),
    }
    baseline_preds_val: dict[str, np.ndarray] = {
        "copy": np.stack([predict_copy(s.action_eth) for s in val_obs], axis=0),
        "local_linear": np.stack(
            [predict_local_linear(s.pre_eth_usd, s.pre_liquidity, s.action_eth) for s in val_obs],
            axis=0,
        ),
    }

    ridge = RandomEncoderRidge(hidden_dim=512, ridge_alpha=15.0, seed=11).fit(train_obs)
    baseline_preds_val["random_encoder_ridge"] = ridge.predict(val_obs)
    baseline_preds_int["random_encoder_ridge"] = ridge.predict(int_samples)

    for b_name, p_int in baseline_preds_int.items():
        obs_val_l = (
            round(float(np.mean(scaler.per_sample_loss(baseline_preds_val[b_name], val_y))), 5)
            if b_name in baseline_preds_val
            else None
        )
        obs_val_bps = (
            round(price_rmse_bps(baseline_preds_val[b_name], val_y), 3)
            if b_name in baseline_preds_val
            else None
        )
        baselines_out[b_name] = {
            "obs_val_std_mse": obs_val_l,
            "obs_val_price_rmse_bps": obs_val_bps,
            "interventions": _summarize_predictions(p_int, int_y, regimes, clusters, scaler),
        }

    copy_obs_loss = float(np.mean(scaler.per_sample_loss(baseline_preds_val["copy"], val_y)))

    # 2. First train conservation_wm across all 5 seeds to determine its median/mean obs val loss
    #    as the matched-loss anchor, then train all other families with target_match_loss!
    cons_bundles = []
    for seed in RANDOM_SEEDS:
        b = train_world_model("conservation_wm", train_b, val_b, scaler, seed=seed, epochs=epochs)
        cons_bundles.append(b)
    anchor_obs_loss = float(np.mean([b.best_val_loss for b in cons_bundles]))
    print(
        f"  [conservation_wm] 5-seed mean obs val loss anchor = {anchor_obs_loss:.5f} "
        f"(copy = {copy_obs_loss:.5f})",
        flush=True,
    )

    bundles_by_family: dict[str, list] = {"conservation_wm": cons_bundles}
    for fam in ("mlp_direct", "seq_transformer", "latent_jepa"):
        fam_list = []
        for seed in RANDOM_SEEDS:
            b = train_world_model(
                fam,
                train_b,
                val_b,
                scaler,
                seed=seed,
                epochs=epochs,
                target_match_loss=anchor_obs_loss,
            )
            fam_list.append(b)
        bundles_by_family[fam] = fam_list
        mean_best = float(np.mean([x.best_val_loss for x in fam_list]))
        mean_match = float(np.mean([x.matched_val_loss for x in fam_list]))
        print(
            f"  [{fam}] 5-seed obs val loss: best={mean_best:.5f}, matched={mean_match:.5f}",
            flush=True,
        )

    # 3. Evaluate all 20 checkpoints (both Matched-Loss [Gate G1] and Best-Val [Gate G2]).
    models_summary: dict[str, Any] = {}
    seed_records: list[dict[str, Any]] = []
    matched_cliff_per_sample: dict[str, np.ndarray] = {}
    best_cliff_per_sample: dict[str, np.ndarray] = {}

    cliff_mask = np.isin(regimes, ["regime_2_cliff", "regime_3_cascade"])
    cliff_clusters = clusters[cliff_mask]

    for fam in MODEL_FAMILIES:
        fam_bundles = bundles_by_family[fam]

        # Seed-averaged predictions & per-seed metrics.
        matched_preds_seeds = []
        best_preds_seeds = []
        oot_preds_seeds = []
        matched_val_losses = []
        best_val_losses = []

        for b in fam_bundles:
            p_int_match = evaluate_model_predictions(b.matched_model, int_b)
            p_int_best = evaluate_model_predictions(b.best_model, int_b)
            p_oot_best = evaluate_model_predictions(b.best_model, oot_b)

            matched_preds_seeds.append(p_int_match)
            best_preds_seeds.append(p_int_best)
            oot_preds_seeds.append(p_oot_best)
            matched_val_losses.append(b.matched_val_loss)
            best_val_losses.append(b.best_val_loss)

            per_best_cliff = scaler.per_sample_loss(p_int_best[cliff_mask], int_y[cliff_mask])
            per_match_cliff = scaler.per_sample_loss(p_int_match[cliff_mask], int_y[cliff_mask])

            seed_records.append(
                {
                    "family": fam,
                    "seed": b.seed,
                    "best_obs_val_loss": round(b.best_val_loss, 6),
                    "matched_obs_val_loss": round(b.matched_val_loss, 6),
                    "best_cliff_std_mse": round(float(np.mean(per_best_cliff)), 4),
                    "matched_cliff_std_mse": round(float(np.mean(per_match_cliff)), 4),
                    "best_cliff_price_rmse_bps": round(
                        price_rmse_bps(p_int_best[cliff_mask], int_y[cliff_mask]), 2
                    ),
                    "matched_cliff_price_rmse_bps": round(
                        price_rmse_bps(p_int_match[cliff_mask], int_y[cliff_mask]), 2
                    ),
                }
            )

        # Average per-sample loss across the 5 seeds for paired cluster bootstrap across blocks.
        per_sample_match = np.mean([scaler.per_sample_loss(p, int_y) for p in matched_preds_seeds], axis=0)
        per_sample_best = np.mean([scaler.per_sample_loss(p, int_y) for p in best_preds_seeds], axis=0)
        matched_cliff_per_sample[fam] = per_sample_match[cliff_mask]
        best_cliff_per_sample[fam] = per_sample_best[cliff_mask]

        mean_p_match = np.mean(matched_preds_seeds, axis=0)
        mean_p_best = np.mean(best_preds_seeds, axis=0)
        mean_p_oot = np.mean(oot_preds_seeds, axis=0)

        val_p_match = np.mean(
            [evaluate_model_predictions(b.matched_model, val_b) for b in fam_bundles], axis=0
        )
        _, val_ci_lo, val_ci_hi = cluster_bootstrap_mean_ci(
            np.mean(
                [
                    scaler.per_sample_loss(evaluate_model_predictions(b.matched_model, val_b), val_y)
                    for b in fam_bundles
                ],
                axis=0,
            ),
            val_clusters,
            n_boot=2000,
        )

        models_summary[fam] = {
            "n_seeds": len(fam_bundles),
            "matched_obs_val_std_mse_mean": round(float(np.mean(matched_val_losses)), 5),
            "matched_obs_val_std_mse_sd": round(float(np.std(matched_val_losses, ddof=1)), 5),
            "matched_obs_val_ci95": [round(val_ci_lo, 5), round(val_ci_hi, 5)],
            "matched_obs_val_price_rmse_bps": round(price_rmse_bps(val_p_match, val_y), 3),
            "best_obs_val_std_mse_mean": round(float(np.mean(best_val_losses)), 5),
            "best_obs_val_std_mse_sd": round(float(np.std(best_val_losses, ddof=1)), 5),
            "oot_obs_oct_std_mse": round(float(np.mean(scaler.per_sample_loss(mean_p_oot, oot_y))), 5),
            "oot_obs_oct_price_rmse_bps": round(price_rmse_bps(mean_p_oot, oot_y), 3),
            "beats_copy_on_obs": bool(np.mean(best_val_losses) < copy_obs_loss),
            "passes_5pct_match_with_conservation": passes_matched_loss_screen(
                float(np.mean(matched_val_losses)), anchor_obs_loss, rel_tol=MATCHED_LOSS_REL_TOL
            ),
            "matched_checkpoint_interventions": _summarize_predictions(
                mean_p_match, int_y, regimes, clusters, scaler
            ),
            "best_checkpoint_interventions": _summarize_predictions(
                mean_p_best, int_y, regimes, clusters, scaler
            ),
        }

    # 4. Gate G1: Pairwise cluster-bootstrapped intervention gaps vs `conservation_wm` + Holm correction.
    raw_pvals: list[float] = []
    pairwise_tests: list[dict[str, Any]] = []
    for challenger in ("mlp_direct", "seq_transformer", "latent_jepa"):
        gap_match = cluster_bootstrap_paired_gap(
            matched_cliff_per_sample[challenger],
            matched_cliff_per_sample["conservation_wm"],
            cliff_clusters,
            n_boot=BOOTSTRAP_REPLICATES,
        )
        gap_best = cluster_bootstrap_paired_gap(
            best_cliff_per_sample[challenger],
            best_cliff_per_sample["conservation_wm"],
            cliff_clusters,
            n_boot=BOOTSTRAP_REPLICATES,
        )
        obs_m_chal = models_summary[challenger]["matched_obs_val_std_mse_mean"]
        obs_m_cons = models_summary["conservation_wm"]["matched_obs_val_std_mse_mean"]
        matched_ok = passes_matched_loss_screen(obs_m_chal, obs_m_cons, rel_tol=MATCHED_LOSS_REL_TOL)

        raw_pvals.append(gap_match["p_two_sided"])
        pairwise_tests.append(
            {
                "challenger": challenger,
                "reference": "conservation_wm",
                "matched_within_5pct": matched_ok,
                "obs_val_rel_diff": round(abs(obs_m_chal - obs_m_cons) / min(obs_m_chal, obs_m_cons), 5),
                "matched_cliff_gap": gap_match,
                "best_cliff_gap": gap_best,
            }
        )

    holm_p = holm_bonferroni(raw_pvals)
    for item, hp in zip(pairwise_tests, holm_p, strict=True):
        item["matched_cliff_gap"]["holm_p"] = round(hp, 6)

    # 5. Gate G2: Rank correlation between observational validation loss and Regime II+III error
    #    across all 20 (family x seed) best-validation checkpoints.
    obs_arr = np.array([r["best_obs_val_loss"] for r in seed_records], dtype=np.float64)
    int_arr = np.array([r["best_cliff_std_mse"] for r in seed_records], dtype=np.float64)
    int_bps_arr = np.array([r["best_cliff_price_rmse_bps"] for r in seed_records], dtype=np.float64)

    spearman_mse = spearman_rank_ci(obs_arr, int_arr, n_boot=BOOTSTRAP_REPLICATES)
    spearman_bps = spearman_rank_ci(obs_arr, int_bps_arr, n_boot=BOOTSTRAP_REPLICATES)

    gate_g1_pass = any(
        t["matched_within_5pct"]
        and t["matched_cliff_gap"]["ci_low"] > 0.0
        and t["matched_cliff_gap"]["rel_gap"] >= 0.20
        and t["matched_cliff_gap"]["holm_p"] < 0.05
        for t in pairwise_tests
    )
    gate_g2_decoupled = bool(spearman_mse["rho"] < 0.50)

    return {
        "arm_name": arm_name,
        "counts": {
            "n_train_obs": len(train_obs),
            "n_val_obs": len(val_obs),
            "n_oot_obs": len(oot_obs),
            "n_interventions_total": len(int_samples),
            "n_held_out_checkpoints": int(len(np.unique(clusters))),
            "regime_counts": {
                r: int(np.sum(regimes == r))
                for r in ("regime_1_smooth", "regime_2_cliff", "regime_3_cascade")
            },
        },
        "target_scaler_std": [round(float(x), 6) for x in scaler.scale],
        "copy_obs_val_std_mse": round(copy_obs_loss, 5),
        "baselines": baselines_out,
        "models": models_summary,
        "pairwise_gate_g1": pairwise_tests,
        "rank_correlation_gate_g2": {
            "n_checkpoints": len(seed_records),
            "spearman_obs_vs_cliff_std_mse": spearman_mse,
            "spearman_obs_vs_cliff_price_bps": spearman_bps,
        },
        "gates": {
            "gate_g1_matched_separation_pass": gate_g1_pass,
            "gate_g2_obs_rank_decoupled": gate_g2_decoupled,
        },
        "seed_checkpoints": seed_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/mainnet_intervention_results.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    t0 = time.perf_counter()
    swaps_by_win = json.loads((args.data_dir / "mainnet_swaps.json").read_text(encoding="utf-8"))
    cps_by_win = json.loads((args.data_dir / "mainnet_checkpoints.json").read_text(encoding="utf-8"))
    aave_raw = json.loads((args.data_dir / "aave_v3_borrowers.json").read_text(encoding="utf-8"))
    aave_borrowers = _load_aave_borrowers(aave_raw)

    device = default_torch_device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Using compute device: {device} ({gpu_name})", flush=True)

    # --- Primary Chronologically Sealed Dataset ---
    # Train: early June 2024 (20,000,000 .. 20,006,400)
    # Val:   late June 2024  (20,006,401 .. 20,008,000)
    # OOT Obs: Oct 2024      (21,000,000 .. 21,008,000)
    train_june = build_real_observational_transitions(
        swaps_by_win["calm_june_2024"],
        cps_by_win["calm_june_2024"],
        min_block=20_000_000,
        max_block=20_006_400,
    )
    val_june = build_real_observational_transitions(
        swaps_by_win["calm_june_2024"],
        cps_by_win["calm_june_2024"],
        min_block=20_006_401,
        max_block=20_008_000,
    )
    oot_oct_obs = build_real_observational_transitions(
        swaps_by_win["autumn_oct_2024"],
        cps_by_win["autumn_oct_2024"],
    )

    # Held-out future counterfactual intervention suite:
    # 1. Crash onset & cascade checkpoints (Aug 5, 2024: 20,458,000 .. 20,460,500 + every 2nd recovery cp)
    # 2. Out-of-time October 2024 checkpoints (every 2nd checkpoint across 21,000,000 .. 21,008,000)
    int_crash_early, st_crash_early, tag_c1 = build_counterfactual_suite(
        cps_by_win["crash_aug5_2024"],
        swaps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_458_000,
        max_block=20_460_400,
        window_tag="crash_aug5_onset",
    )
    int_crash_late, st_crash_late, tag_c2 = build_counterfactual_suite(
        cps_by_win["crash_aug5_2024"],
        swaps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_465_001,
        max_block=20_468_000,
        stride=2,
        window_tag="crash_aug5_late",
    )
    int_oct, st_oct, tag_oct = build_counterfactual_suite(
        cps_by_win["autumn_oct_2024"],
        swaps_by_win["autumn_oct_2024"],
        all_borrowers=None,
        stride=2,
        window_tag="autumn_oct_oot",
    )

    primary_int_samples = int_crash_early + int_crash_late + int_oct
    primary_int_states = st_crash_early + st_crash_late + st_oct

    primary_res = run_arm(
        "primary_chronological_sealed",
        train_june,
        val_june,
        oot_oct_obs,
        primary_int_samples,
        primary_int_states,
        device=device,
        epochs=args.epochs,
    )

    # --- Sensitivity Arm: Crash-Exposed Observational Training ---
    # Add 4,500 real high-volatility swaps from mid-crash (20,460,500 .. 20,465,000) to Train,
    # keeping held-out intervention states strictly disjoint in block space!
    crash_mid_train = build_real_observational_transitions(
        swaps_by_win["crash_aug5_2024"],
        cps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_460_500,
        max_block=20_464_200,
        max_samples=4500,
    )
    crash_mid_val = build_real_observational_transitions(
        swaps_by_win["crash_aug5_2024"],
        cps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_464_201,
        max_block=20_465_000,
        max_samples=1200,
    )

    disjoint_int_samples = int_crash_late + int_oct
    disjoint_int_states = st_crash_late + st_oct

    sensitivity_res = run_arm(
        "sensitivity_crash_exposed_training",
        train_june + crash_mid_train,
        val_june + crash_mid_val,
        oot_oct_obs,
        disjoint_int_samples,
        disjoint_int_states,
        device=device,
        epochs=args.epochs,
    )

    elapsed = time.perf_counter() - t0
    full_output = {
        "benchmark": "evm-intervene",
        "device": str(device),
        "seeds": list(RANDOM_SEEDS),
        "epochs": args.epochs,
        "wall_seconds": round(elapsed, 2),
        "primary": primary_res,
        "sensitivity_crash_exposed": sensitivity_res,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    print(f"\nCompleted full Mainnet intervention benchmark in {elapsed:.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
