"""Fair, spatial state-action encoding shared identically across all world models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from evm_intervene.constants import (
    AAVE_FULL_LIQ_HF_CUTOFF,
    AAVE_LIQUIDATION_BONUS,
    POOL_FEE_RATE,
    USDC_DECIMALS,
    WETH_DECIMALS,
)
from evm_intervene.simulator import (
    PoolState,
    TransitionOutcome,
    raw_sqrt_price_from_eth_usd,
    raw_sqrt_price_to_tick,
)

# 16 symmetric log-price offsets from spot P_t (in log-ETH/USD coordinates), spanning +-17%.
POSITIVE_LOG_EDGES: tuple[float, ...] = (
    0.0005,
    0.0010,
    0.0020,
    0.0035,
    0.0050,
    0.0075,
    0.0100,
    0.0150,
    0.0200,
    0.0300,
    0.0400,
    0.0550,
    0.0750,
    0.1000,
    0.1300,
    0.1700,
)

# Downside bins: b = 0..15 cover [0 -> -0.0005], [-0.0005 -> -0.0010], ..., [-0.13 -> -0.17].
# Upside bins:   b = 16..31 cover [0 -> +0.0005], [+0.0005 -> +0.0010], ..., [+0.13 -> +0.17].
NUM_SIDE_BINS = len(POSITIVE_LOG_EDGES)  # 16
NUM_TOTAL_BINS = 2 * NUM_SIDE_BINS  # 32
HISTORY_DIM = 6
GLOBAL_SCALAR_DIM = 6 + HISTORY_DIM  # 12


@dataclass(frozen=True)
class EncodedTransition:
    """Model-ready representation of (s_t, a_t) -> y_{t+1} with full spatial observability."""

    # Spatial profile arrays of shape (32,): 0..15 = downside intervals, 16..31 = upside intervals.
    bin_rel_liquidity: np.ndarray  # L_b / L_t
    bin_eth_capacity: np.ndarray  # Exact fee-free ETH-equiv capacity of bin b at L_b (if ratio=1 -> base_cap)
    bin_base_unit_capacity: np.ndarray  # ETH-equiv capacity of bin b if L_b were equal to L_t
    bin_log_delta: np.ndarray  # Signed log-price change across bin b (<0 for downside, >0 for upside)
    bin_liq_eth_supply: np.ndarray  # Forced WETH liquidation supply triggered inside bin b (ETH)
    global_scalars: np.ndarray  # Shape (12,): pool scalars + recent swap history summary
    action_eth: float  # Signed action a_t (>0 sell WETH, <0 buy WETH equiv)
    target: np.ndarray  # Shape (3,): [delta_log_price, delta_log_liq, log_exec_eff]
    pre_eth_usd: float
    pre_liquidity: float
    regime: str  # "observational", "regime_1_smooth", "regime_2_cliff", "regime_3_cascade"
    block_number: int


def _active_liquidity_at_eth_price(state: PoolState, target_eth_usd: float) -> float:
    """Compute exact Uniswap v3 active liquidity L if spot moved to `target_eth_usd`."""
    q_target = raw_sqrt_price_from_eth_usd(target_eth_usd)
    target_tick = raw_sqrt_price_to_tick(q_target)
    if target_tick == state.tick or len(state.ticks) == 0:
        return state.liquidity

    L = state.liquidity
    ticks = state.ticks
    liq_net = state.liquidity_net

    if target_tick > state.tick:
        # Moving down in ETH/USD -> moving UP in raw Uniswap tick.
        i_start = int(np.searchsorted(ticks, state.tick, side="right"))
        i_end = int(np.searchsorted(ticks, target_tick, side="right"))
        for idx in range(i_start, i_end):
            L = max(1.0, L + liq_net[idx])
    else:
        # Moving up in ETH/USD -> moving DOWN in raw Uniswap tick.
        i_end = int(np.searchsorted(ticks, state.tick, side="right")) - 1
        i_start = int(np.searchsorted(ticks, target_tick, side="right"))
        for idx in range(i_end, i_start - 1, -1):
            L = max(1.0, L - liq_net[idx])
    return L


def summarize_swap_history(recent_actions_eth: np.ndarray, recent_dlogp: np.ndarray) -> np.ndarray:
    """Compute causal 6-dim summary H_t from preceding observational swaps (strictly < t)."""
    if len(recent_actions_eth) == 0:
        return np.zeros(HISTORY_DIM, dtype=np.float64)
    abs_a = np.abs(recent_actions_eth)
    mean_signed = float(np.mean(recent_actions_eth))
    mean_abs = float(np.mean(abs_a))
    realized_vol = float(np.std(recent_dlogp)) if len(recent_dlogp) > 1 else 0.0
    max_abs_dlogp = float(np.max(np.abs(recent_dlogp)))
    imbalance = float(np.sum(recent_actions_eth) / (np.sum(abs_a) + 1e-8))
    # Empirical local impact ratio |dlogp| / (|a| + 0.1 ETH)
    impact_ratio = float(np.mean(np.abs(recent_dlogp) / (abs_a + 0.1)))
    return np.array(
        [
            math.asinh(mean_signed),
            math.log1p(mean_abs),
            realized_vol * 1e3,
            max_abs_dlogp * 1e3,
            imbalance,
            impact_ratio * 1e3,
        ],
        dtype=np.float64,
    )


def target_from_outcome(outcome: TransitionOutcome) -> np.ndarray:
    """Construct 3D target y_{t+1} = [delta_log_price, delta_log_liquidity, log_exec_eff]."""
    eff = max(outcome.execution_efficiency, 1e-4)
    return np.array(
        [
            outcome.delta_log_price,
            outcome.delta_log_liquidity,
            math.log(eff),
        ],
        dtype=np.float64,
    )


def classify_intervention_regime(state: PoolState, outcome: TransitionOutcome) -> str:
    """Classify an intervention transition into Regimes I, II, or III (docs/DESIGN.md)."""
    if outcome.liquidated_eth > 1e-6 or outcome.cascade_rounds > 0:
        return "regime_3_cascade"
    if outcome.ticks_crossed > 0 or abs(outcome.delta_log_liquidity) > 1e-4:
        return "regime_2_cliff"
    return "regime_1_smooth"


def encode_state_action(
    state: PoolState,
    action_eth: float,
    outcome: TransitionOutcome,
    history_summary: np.ndarray | None = None,
    *,
    regime: str = "observational",
) -> EncodedTransition:
    """Build identical spatial + global features for (s_t, a_t) across all model families."""
    p0 = state.eth_usd
    L0 = max(state.liquidity, 1.0)
    hist = history_summary if history_summary is not None else np.zeros(HISTORY_DIM, dtype=np.float64)

    rel_liq = np.ones(NUM_TOTAL_BINS, dtype=np.float64)
    eth_cap = np.zeros(NUM_TOTAL_BINS, dtype=np.float64)
    unit_cap = np.zeros(NUM_TOTAL_BINS, dtype=np.float64)
    log_delta = np.zeros(NUM_TOTAL_BINS, dtype=np.float64)
    liq_supply = np.zeros(NUM_TOTAL_BINS, dtype=np.float64)

    # Precompute borrower critical prices for downside bins.
    b_crit = [b.critical_eth_price for b in state.borrowers]

    # Downside bins b = 0..15: moving from u_hi <= 0 down to u_lo < u_hi (WETH sell direction).
    prev_u = 0.0
    for b, edge in enumerate(POSITIVE_LOG_EDGES):
        u_hi = -prev_u
        u_lo = -edge
        u_mid = 0.5 * (u_hi + u_lo)
        p_hi = p0 * math.exp(u_hi)
        p_lo = p0 * math.exp(u_lo)
        p_mid = p0 * math.exp(u_mid)

        L_mid = _active_liquidity_at_eth_price(state, p_mid)
        ratio = L_mid / L0
        rel_liq[b] = ratio

        q_hi = raw_sqrt_price_from_eth_usd(p_hi)  # smaller q
        q_lo = raw_sqrt_price_from_eth_usd(p_lo)  # larger q
        base_c = (L0 * (q_lo - q_hi)) / ((10.0**WETH_DECIMALS) * (1.0 - POOL_FEE_RATE))
        unit_cap[b] = base_c
        eth_cap[b] = base_c * ratio
        log_delta[b] = u_lo - u_hi  # negative

        # Sum forced WETH liquidation supply for positions crossing HF < 1 inside (p_lo, p_hi].
        if len(state.borrowers) > 0:
            forced_b = 0.0
            for idx_m, pos in enumerate(state.borrowers):
                pc = b_crit[idx_m]
                if p_lo <= pc <= p_hi and pos.collateral_eth > 0.0 and pos.debt_usdc > 0.0:
                    hf_at_lo = pos.health_factor(p_lo)
                    cf = 1.0 if hf_at_lo < AAVE_FULL_LIQ_HF_CUTOFF else 0.5
                    seize = min(
                        pos.collateral_eth,
                        (cf * pos.debt_usdc * (1.0 + AAVE_LIQUIDATION_BONUS)) / p_mid,
                    )
                    forced_b += seize
            liq_supply[b] = forced_b

        prev_u = edge

    # Upside bins b = 16..31: moving from u_lo >= 0 up to u_hi > u_lo (WETH buy direction).
    prev_u = 0.0
    for j, edge in enumerate(POSITIVE_LOG_EDGES):
        b = NUM_SIDE_BINS + j
        u_lo = prev_u
        u_hi = edge
        u_mid = 0.5 * (u_lo + u_hi)
        p_lo = p0 * math.exp(u_lo)
        p_hi = p0 * math.exp(u_hi)
        p_mid = p0 * math.exp(u_mid)

        L_mid = _active_liquidity_at_eth_price(state, p_mid)
        ratio = L_mid / L0
        rel_liq[b] = ratio

        q_lo = raw_sqrt_price_from_eth_usd(p_lo)  # larger q
        q_hi = raw_sqrt_price_from_eth_usd(p_hi)  # smaller q
        base_c = (L0 * (1.0 / q_hi - 1.0 / q_lo)) / ((10.0**USDC_DECIMALS) * p0 * (1.0 - POOL_FEE_RATE))
        unit_cap[b] = base_c
        eth_cap[b] = base_c * ratio
        log_delta[b] = u_hi - u_lo  # positive
        liq_supply[b] = 0.0

        prev_u = edge

    max_sell_tick = min(state.max_in_tick_eth_sell(), 1e6)
    max_buy_tick = min(state.max_in_tick_eth_buy_equiv(), 1e6)
    total_liq_near = float(np.sum(liq_supply[:NUM_SIDE_BINS]))
    min_rel_liq_down = float(np.min(rel_liq[:NUM_SIDE_BINS]))

    global_scalars = np.concatenate(
        [
            np.array(
                [
                    math.log(p0 / 3000.0),
                    math.log(L0 / 1e19),
                    math.log1p(max_sell_tick),
                    math.log1p(max_buy_tick),
                    math.log1p(total_liq_near),
                    math.log(max(min_rel_liq_down, 1e-4)),
                ],
                dtype=np.float64,
            ),
            hist.astype(np.float64),
        ]
    )

    return EncodedTransition(
        bin_rel_liquidity=rel_liq,
        bin_eth_capacity=eth_cap,
        bin_base_unit_capacity=unit_cap,
        bin_log_delta=log_delta,
        bin_liq_eth_supply=liq_supply,
        global_scalars=global_scalars,
        action_eth=float(action_eth),
        target=target_from_outcome(outcome),
        pre_eth_usd=p0,
        pre_liquidity=L0,
        regime=regime,
        block_number=state.block_number,
    )


def flatten_features(sample: EncodedTransition) -> np.ndarray:
    """Flatten EncodedTransition into a fixed-length vector for tabular/MLP/random-feature models."""
    a = sample.action_eth
    act_feats = np.array(
        [
            a / 100.0,
            math.asinh(a),
            math.asinh(a / 10.0),
            1.0 if a >= 0.0 else -1.0,
            math.log1p(abs(a)),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            np.log(np.clip(sample.bin_rel_liquidity, 1e-4, 1e4)),
            np.log1p(sample.bin_eth_capacity),
            np.log1p(sample.bin_liq_eth_supply),
            sample.global_scalars,
            act_feats,
        ]
    )
