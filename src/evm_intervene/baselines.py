"""Analytical, partial-physics, and untrained random-projection baselines."""

from __future__ import annotations

import math

import numpy as np

from evm_intervene.constants import POOL_FEE_RATE
from evm_intervene.simulator import PoolState, raw_sqrt_price_from_eth_usd, simulate_intervention
from evm_intervene.state import EncodedTransition, flatten_features, target_from_outcome


def predict_copy(action_eth: float, fee_rate: float = POOL_FEE_RATE) -> np.ndarray:
    """Zero-impact martingale baseline: price and liquidity unchanged."""
    eff = (1.0 - fee_rate) if abs(action_eth) > 1e-12 else 1.0
    return np.array([0.0, 0.0, math.log(eff)], dtype=np.float64)


def predict_local_linear(
    pre_eth_usd: float,
    pre_liquidity: float,
    action_eth: float,
    fee_rate: float = POOL_FEE_RATE,
) -> np.ndarray:
    """Exact first-order Kyle's lambda tangent around current state s_t."""
    if abs(action_eth) <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    q0 = raw_sqrt_price_from_eth_usd(pre_eth_usd)
    # Exact local slope d(log P)/da at a=0: -2 * (1 - fee) * 1e18 / (L_t * q_t)
    slope = -2.0 * (1.0 - fee_rate) * 1e18 / (max(pre_liquidity, 1.0) * q0)
    dlogp = float(np.clip(slope * action_eth, -2.0, 2.0))
    # Average execution price moves by ~0.5 * |dlogp| against trader plus fee.
    log_eff = math.log(1.0 - fee_rate) - 0.5 * abs(dlogp)
    return np.array([dlogp, 0.0, log_eff], dtype=np.float64)


def predict_single_tick_xyk(state: PoolState, action_eth: float) -> np.ndarray:
    """Constant-liquidity x*y=k curve ignoring tick boundaries and liquidations."""
    infinite_tick_state = PoolState(
        block_number=state.block_number,
        sqrt_price_raw=state.sqrt_price_raw,
        tick=state.tick,
        liquidity=state.liquidity,
        ticks=(),
        liquidity_net=(),
        borrowers=(),
        fee_rate=state.fee_rate,
    )
    out = simulate_intervention(infinite_tick_state, action_eth, enable_liquidations=False)
    return target_from_outcome(out)


def predict_amm_only_exact(state: PoolState, action_eth: float) -> np.ndarray:
    """Exact multi-tick Uniswap v3 physics without Stage 2 Aave liquidations."""
    out = simulate_intervention(state, action_eth, enable_liquidations=False)
    return target_from_outcome(out)


class RandomEncoderRidge:
    """Untrained 2-layer random neural feature projection + Ridge readout (`causal-jepa` control)."""

    def __init__(self, hidden_dim: int = 512, ridge_alpha: float = 10.0, seed: int = 0) -> None:
        self.hidden_dim = hidden_dim
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._w1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._w2: np.ndarray | None = None
        self._b2: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None

    def _project(self, x: np.ndarray) -> np.ndarray:
        assert self._x_mean is not None and self._x_std is not None
        assert self._w1 is not None and self._b1 is not None
        assert self._w2 is not None and self._b2 is not None
        z = (x - self._x_mean) / self._x_std
        h1 = np.tanh(z @ self._w1 + self._b1)
        h2 = np.maximum(0.0, h1 @ self._w2 + self._b2)
        return np.concatenate([z, h2], axis=1)

    def fit(self, samples: list[EncodedTransition]) -> RandomEncoderRidge:
        x = np.stack([flatten_features(s) for s in samples], axis=0)
        y = np.stack([s.target for s in samples], axis=0)
        self._x_mean = np.mean(x, axis=0, keepdims=True)
        self._x_std = np.maximum(np.std(x, axis=0, keepdims=True), 1e-6)

        rng = np.random.default_rng(self.seed)
        d = x.shape[1]
        self._w1 = rng.normal(0.0, 1.0 / math.sqrt(d), size=(d, self.hidden_dim))
        self._b1 = rng.uniform(-0.5, 0.5, size=(1, self.hidden_dim))
        self._w2 = rng.normal(0.0, 1.0 / math.sqrt(self.hidden_dim), size=(self.hidden_dim, self.hidden_dim))
        self._b2 = rng.uniform(-0.5, 0.5, size=(1, self.hidden_dim))

        phi = self._project(x)
        phi_mean = np.mean(phi, axis=0, keepdims=True)
        y_mean = np.mean(y, axis=0, keepdims=True)
        phi_c = phi - phi_mean
        y_c = y - y_mean

        gram = phi_c.T @ phi_c + self.ridge_alpha * np.eye(phi.shape[1])
        self._coef = np.linalg.solve(gram, phi_c.T @ y_c)
        self._intercept = y_mean - phi_mean @ self._coef
        return self

    def predict(self, samples: list[EncodedTransition]) -> np.ndarray:
        assert self._coef is not None and self._intercept is not None
        x = np.stack([flatten_features(s) for s in samples], axis=0)
        phi = self._project(x)
        return phi @ self._coef + self._intercept
