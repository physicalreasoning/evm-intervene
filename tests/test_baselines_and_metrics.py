"""Unit tests for baselines, state encoding, matched-loss screen, bootstrap CIs, and Holm."""

from __future__ import annotations

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
    holm_bonferroni,
    passes_matched_loss_screen,
    price_rmse_bps,
    spearman_rank_ci,
)
from evm_intervene.simulator import BorrowerPosition, simulate_intervention
from evm_intervene.state import classify_intervention_regime, encode_state_action
from tests.test_simulator import _make_cliff_pool


def test_local_linear_matches_exact_amm_for_infinitesimal_swap() -> None:
    pool = _make_cliff_pool(spot_eth_usd=3000.0, active_liq=2.0e19)
    a = 1.0  # tiny 1 ETH swap (~0.05 bps)
    out = simulate_intervention(pool, a, enable_liquidations=False)
    lin = predict_local_linear(pool.eth_usd, pool.liquidity, a)
    xyk = predict_single_tick_xyk(pool, a)

    assert np.isclose(lin[0], out.delta_log_price, rtol=1e-3)
    assert np.isclose(xyk[0], out.delta_log_price, rtol=1e-7)


def test_regime_classification_and_amm_only_separation() -> None:
    b1 = BorrowerPosition(collateral_eth=500.0, debt_usdc=1_221_000.0, liquidation_threshold=0.825)
    pool = _make_cliff_pool(spot_eth_usd=3000.0, active_liq=2.0e19, cliff_drop_ratio=0.15, borrowers=(b1,))

    r1 = simulate_intervention(pool, 50.0)
    assert classify_intervention_regime(pool, r1) == "regime_1_smooth"

    r2 = simulate_intervention(pool, -700.0)
    assert classify_intervention_regime(pool, r2) == "regime_2_cliff"

    r3 = simulate_intervention(pool, +750.0)
    assert classify_intervention_regime(pool, r3) == "regime_3_cascade"
    amm_pred = predict_amm_only_exact(pool, +750.0)
    assert r3.delta_log_price < amm_pred[0]


def test_random_encoder_ridge_and_metrics() -> None:
    pool = _make_cliff_pool()
    rng = np.random.default_rng(7)
    samples = []
    for _ in range(80):
        a = float(rng.normal(0.0, 20.0))
        out = simulate_intervention(pool, a)
        samples.append(encode_state_action(pool, a, out))

    ridge = RandomEncoderRidge(hidden_dim=64, seed=3).fit(samples[:60])
    preds = ridge.predict(samples[60:])
    y_true = np.stack([s.target for s in samples[60:]], axis=0)
    copy_preds = np.stack([predict_copy(s.action_eth) for s in samples[60:]], axis=0)

    assert price_rmse_bps(preds, y_true) < price_rmse_bps(copy_preds, y_true)

    scaler = TargetScaler.from_targets(np.stack([s.target for s in samples[:60]], axis=0))
    l_ridge = scaler.per_sample_loss(preds, y_true)
    l_copy = scaler.per_sample_loss(copy_preds, y_true)
    clusters = np.arange(len(y_true)) // 2

    gap = cluster_bootstrap_paired_gap(l_copy, l_ridge, clusters, n_boot=500)
    assert gap["gap"] > 0.0
    assert gap["ci_low"] > 0.0

    assert passes_matched_loss_screen(0.100, 0.104, rel_tol=0.05)
    assert not passes_matched_loss_screen(0.100, 0.110, rel_tol=0.05)

    adj = holm_bonferroni([0.04, 0.01, 0.20])
    assert adj[1] == 0.03
    assert adj[0] == 0.08
    assert adj[2] == 0.20

    sp = spearman_rank_ci(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 1.9, 3.2, 4.5]), n_boot=200)
    assert sp["rho"] > 0.9
