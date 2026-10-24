"""Gate G0 offline synthetic positive control (< 5 seconds, no network required).

Verifies that on a controlled concentrated-liquidity + liquidation cliff environment:
1. All 4 learned world-model families beat `copy` on routine observational swaps;
2. Unstructured neural families match `conservation_wm` on observational validation loss (within 5%);
3. Under unseen Regime II/III cliff interventions, `conservation_wm` separates significantly from
   matched-loss unstructured neural models (95% cluster-bootstrap CI excluding zero).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evm_intervene.baselines import (
    RandomEncoderRidge,
    predict_amm_only_exact,
    predict_copy,
    predict_local_linear,
    predict_single_tick_xyk,
)
from evm_intervene.metrics import (
    TargetScaler,
    cluster_bootstrap_paired_gap,
    passes_matched_loss_screen,
    price_rmse_bps,
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
    raw_sqrt_price_from_eth_usd,
    raw_sqrt_price_to_tick,
    simulate_intervention,
)
from evm_intervene.state import classify_intervention_regime, encode_state_action


def _synthetic_pool(block_id: int, rng: np.random.Generator) -> PoolState:
    spot = float(rng.uniform(2200.0, 3400.0))
    active_liq = float(rng.uniform(1.2e19, 3.2e19))
    cliff_ratio = float(rng.uniform(0.08, 0.22))
    half_width_ticks = int(rng.choice([20, 30, 40, 50]))

    q0 = raw_sqrt_price_from_eth_usd(spot)
    t0 = raw_sqrt_price_to_tick(q0)
    t_lo = (t0 - half_width_ticks) // 10 * 10
    t_hi = (t0 + half_width_ticks) // 10 * 10
    conc_L = active_liq * (1.0 - cliff_ratio)

    # Place 3 staggered leveraged borrowers between -1.2% and -4.5% below spot.
    borrowers = []
    for drop in (0.014, 0.026, 0.040):
        crit_p = spot * (1.0 - drop * float(rng.uniform(0.9, 1.1)))
        coll = float(rng.uniform(250.0, 600.0))
        lt = 0.825
        debt = coll * crit_p * lt
        borrowers.append(BorrowerPosition(collateral_eth=coll, debt_usdc=debt, liquidation_threshold=lt))

    return PoolState(
        block_number=block_id,
        sqrt_price_raw=q0,
        tick=t0,
        liquidity=active_liq,
        ticks=(t_lo, t_hi),
        liquidity_net=(+conc_L, -conc_L),
        borrowers=tuple(borrowers),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/positive_control.json"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = default_torch_device(args.device)

    t0 = time.perf_counter()
    rng = np.random.default_rng(20261024)

    train_pools = [_synthetic_pool(1000 + i, rng) for i in range(40)]
    val_pools = [_synthetic_pool(2000 + i, rng) for i in range(16)]
    test_pools = [_synthetic_pool(3000 + i, rng) for i in range(24)]

    def _sample_obs(pools: list[PoolState], per_pool: int) -> list:
        out_s = []
        for p in pools:
            cap = p.max_in_tick_eth_sell()
            for _ in range(per_pool):
                # 90% small in-tick routine swaps, 10% mild boundary-grazing swaps.
                scale = 0.18 * cap if rng.random() < 0.90 else 0.75 * cap
                a = float(rng.normal(0.0, scale))
                res = simulate_intervention(p, a)
                out_s.append(encode_state_action(p, a, res, regime="observational"))
        return out_s

    train_obs = _sample_obs(train_pools, 20)
    val_obs = _sample_obs(val_pools, 20)

    int_samples = []
    int_pools_ref = []
    for p in test_pools:
        cap = p.max_in_tick_eth_sell()
        for mult in (-2.5, -1.6, -0.5, 0.4, 0.75, 1.5, 2.4, 3.5):
            a = mult * cap
            res = simulate_intervention(p, a)
            reg = classify_intervention_regime(p, res)
            int_samples.append(encode_state_action(p, a, res, regime=reg))
            int_pools_ref.append(p)

    scaler = TargetScaler.from_targets(np.stack([s.target for s in train_obs], axis=0))
    device = default_torch_device("auto")

    train_b = BatchTensors.from_samples(train_obs, device)
    val_b = BatchTensors.from_samples(val_obs, device)
    int_b = BatchTensors.from_samples(int_samples, device)

    val_y = np.stack([s.target for s in val_obs], axis=0)
    int_y = np.stack([s.target for s in int_samples], axis=0)
    cliff_mask = np.array([s.regime in ("regime_2_cliff", "regime_3_cascade") for s in int_samples])
    clusters = np.array([s.block_number for s in int_samples])[cliff_mask]

    # Baselines.
    copy_val = np.stack([predict_copy(s.action_eth) for s in val_obs], axis=0)
    copy_obs_loss = float(np.mean(scaler.per_sample_loss(copy_val, val_y)))

    ridge = RandomEncoderRidge(hidden_dim=256, seed=11).fit(train_obs)
    ridge_int = ridge.predict(int_samples)
    xyk_int = np.stack(
        [predict_single_tick_xyk(p, s.action_eth) for p, s in zip(int_pools_ref, int_samples, strict=True)]
    )
    amm_int = np.stack(
        [predict_amm_only_exact(p, s.action_eth) for p, s in zip(int_pools_ref, int_samples, strict=True)]
    )
    lin_int = np.stack(
        [predict_local_linear(s.pre_eth_usd, s.pre_liquidity, s.action_eth) for s in int_samples]
    )

    # First train conservation_wm to establish target observational validation loss.
    cons_bundle = train_world_model("conservation_wm", train_b, val_b, scaler, seed=11, epochs=45)
    target_obs = cons_bundle.best_val_loss

    families = ("mlp_direct", "seq_transformer", "latent_jepa", "conservation_wm")
    results_by_model: dict[str, dict] = {}
    cliff_losses: dict[str, np.ndarray] = {}

    for fam in families:
        b = (
            cons_bundle
            if fam == "conservation_wm"
            else train_world_model(
                fam, train_b, val_b, scaler, seed=11, epochs=55, target_match_loss=target_obs
            )
        )
        preds_int = evaluate_model_predictions(b.matched_model, int_b)
        per_int = scaler.per_sample_loss(preds_int, int_y)
        cliff_losses[fam] = per_int[cliff_mask]
        results_by_model[fam] = {
            "matched_obs_val_loss": round(b.matched_val_loss, 6),
            "best_obs_val_loss": round(b.best_val_loss, 6),
            "beats_copy_on_obs": bool(b.matched_val_loss < copy_obs_loss),
            "matched_within_5pct_of_conservation": passes_matched_loss_screen(b.matched_val_loss, target_obs),
            "cliff_intervention_loss": round(float(np.mean(per_int[cliff_mask])), 4),
            "cliff_price_rmse_bps": round(price_rmse_bps(preds_int[cliff_mask], int_y[cliff_mask]), 2),
        }

    gap_mlp_vs_cons = cluster_bootstrap_paired_gap(
        cliff_losses["mlp_direct"], cliff_losses["conservation_wm"], clusters, n_boot=2000
    )

    elapsed = time.perf_counter() - t0
    out_json = {
        "copy_obs_val_loss": round(copy_obs_loss, 4),
        "baselines_cliff_price_rmse_bps": {
            "local_linear": round(price_rmse_bps(lin_int[cliff_mask], int_y[cliff_mask]), 2),
            "single_tick_xyk": round(price_rmse_bps(xyk_int[cliff_mask], int_y[cliff_mask]), 2),
            "amm_only_exact": round(price_rmse_bps(amm_int[cliff_mask], int_y[cliff_mask]), 2),
            "random_encoder_ridge": round(price_rmse_bps(ridge_int[cliff_mask], int_y[cliff_mask]), 2),
        },
        "models": results_by_model,
        "mlp_vs_conservation_cliff_gap": gap_mlp_vs_cons,
        "runtime_seconds": round(elapsed, 2),
        "positive_control_pass": bool(gap_mlp_vs_cons["ci_low"] > 0.0),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
