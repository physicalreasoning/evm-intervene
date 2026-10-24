"""Unit tests for all four PyTorch world-model families."""

from __future__ import annotations

import numpy as np
import torch

from evm_intervene.metrics import TargetScaler
from evm_intervene.models import (
    BatchTensors,
    evaluate_model_predictions,
    train_world_model,
)
from evm_intervene.simulator import BorrowerPosition, simulate_intervention
from evm_intervene.state import encode_state_action
from tests.test_simulator import _make_cliff_pool


def test_all_four_model_families_train_and_beat_zero() -> None:
    b1 = BorrowerPosition(collateral_eth=300.0, debt_usdc=720_000.0, liquidation_threshold=0.825)
    pool = _make_cliff_pool(spot_eth_usd=3000.0, active_liq=2.0e19, cliff_drop_ratio=0.20, borrowers=(b1,))
    rng = np.random.default_rng(42)

    samples = []
    for _ in range(96):
        a = float(rng.normal(0.0, 40.0))
        out = simulate_intervention(pool, a)
        samples.append(encode_state_action(pool, a, out))

    train_s = samples[:72]
    val_s = samples[72:]
    scaler = TargetScaler.from_targets(np.stack([s.target for s in train_s], axis=0))

    device = torch.device("cpu")
    train_b = BatchTensors.from_samples(train_s, device)
    val_b = BatchTensors.from_samples(val_s, device)
    val_y = np.stack([s.target for s in val_s], axis=0)
    zero_loss = float(np.mean(scaler.per_sample_loss(np.zeros_like(val_y), val_y)))

    for family in ("mlp_direct", "seq_transformer", "latent_jepa", "conservation_wm"):
        bundle = train_world_model(
            family,
            train_b,
            val_b,
            scaler,
            seed=11,
            epochs=18,
            batch_size=32,
            lr=2e-3,
            target_match_loss=0.25,
        )
        preds = evaluate_model_predictions(bundle.best_model, val_b)
        assert preds.shape == val_y.shape
        assert np.all(np.isfinite(preds))
        assert bundle.best_val_loss < zero_loss
