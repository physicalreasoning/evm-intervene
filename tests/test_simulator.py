"""Unit tests for exact Uniswap v3 concentrated-liquidity and Aave v3 cascade physics."""

from __future__ import annotations

import math

import pytest

from evm_intervene.simulator import (
    BorrowerPosition,
    PoolState,
    eth_usd_from_raw_sqrt_price,
    raw_sqrt_price_from_eth_usd,
    raw_sqrt_price_to_tick,
    simulate_intervention,
    tick_to_raw_sqrt_price,
)


def _make_cliff_pool(
    spot_eth_usd: float = 3000.0,
    active_liq: float = 2.0e19,
    cliff_drop_ratio: float = 0.10,
    borrowers: tuple[BorrowerPosition, ...] = (),
) -> PoolState:
    """Create a realistic USDC/WETH 0.05% pool with a tight +-0.30% concentrated-liquidity range."""
    q0 = raw_sqrt_price_from_eth_usd(spot_eth_usd)
    tick0 = raw_sqrt_price_to_tick(q0)

    # Tight LP position spanning +-30 ticks (~+-0.30% around spot); outside, only 10% remains.
    t_low = (tick0 - 30) // 10 * 10
    t_high = (tick0 + 30) // 10 * 10
    concentrated_L = active_liq * (1.0 - cliff_drop_ratio)

    return PoolState(
        block_number=20_000_000,
        sqrt_price_raw=q0,
        tick=tick0,
        liquidity=active_liq,
        ticks=(t_low, t_high),
        liquidity_net=(+concentrated_L, -concentrated_L),
        borrowers=borrowers,
    )


def test_price_and_tick_roundtrip() -> None:
    for price in (1500.0, 2500.0, 3500.0):
        q = raw_sqrt_price_from_eth_usd(price)
        assert math.isclose(eth_usd_from_raw_sqrt_price(q), price, rel_tol=1e-12)
        t = raw_sqrt_price_to_tick(q)
        assert tick_to_raw_sqrt_price(t) <= q < tick_to_raw_sqrt_price(t + 1)


def test_zero_action_is_identity() -> None:
    pool = _make_cliff_pool()
    out = simulate_intervention(pool, 0.0)
    assert out.post_eth_usd == pool.eth_usd
    assert out.post_liquidity == pool.liquidity
    assert out.ticks_crossed == 0
    assert out.cascade_rounds == 0


def test_directional_monotonicity_and_fee_bound() -> None:
    pool = _make_cliff_pool()
    sell = simulate_intervention(pool, +25.0)
    buy = simulate_intervention(pool, -25.0)

    assert sell.post_eth_usd < pool.eth_usd
    assert sell.usdc_delta > 0.0
    assert sell.execution_efficiency < 1.0

    assert buy.post_eth_usd > pool.eth_usd
    assert buy.usdc_delta < 0.0
    assert buy.execution_efficiency < 1.0


def test_concentrated_liquidity_tick_cliff_amplifies_impact() -> None:
    pool = _make_cliff_pool(spot_eth_usd=3000.0, active_liq=2.0e19, cliff_drop_ratio=0.10)
    max_in_tick = pool.max_in_tick_eth_sell()
    assert 100.0 < max_in_tick < 1000.0

    # Small swap inside active tick: zero tick crossings, liquidity unchanged.
    small = simulate_intervention(pool, 0.5 * max_in_tick)
    assert small.ticks_crossed == 0
    assert math.isclose(small.post_liquidity, pool.liquidity, rel_tol=1e-9)

    # Swap 2x the active-tick capacity crosses the cliff -> active liquidity drops to 10%!
    large = simulate_intervention(pool, 2.0 * max_in_tick)
    assert large.ticks_crossed >= 1
    assert math.isclose(large.post_liquidity, 0.10 * pool.liquidity, rel_tol=1e-6)
    # Marginal impact beyond the cliff must be dramatically steeper than 4x the half-tick trade.
    assert abs(large.delta_log_price) > 6.0 * abs(small.delta_log_price)


def test_aave_liquidation_cascade_feedback_loop() -> None:
    # Two staggered leveraged borrowers sitting at $2965 and $2950.
    b1 = BorrowerPosition(collateral_eth=400.0, debt_usdc=978_450.0, liquidation_threshold=0.825)
    b2 = BorrowerPosition(collateral_eth=500.0, debt_usdc=1_216_875.0, liquidation_threshold=0.825)
    assert pytest.approx(b1.critical_eth_price, rel=1e-4) == 2965.0
    assert pytest.approx(b2.critical_eth_price, rel=1e-4) == 2950.0

    pool = _make_cliff_pool(spot_eth_usd=3000.0, active_liq=2.0e19, cliff_drop_ratio=0.15, borrowers=(b1, b2))
    # Swap pushes post-AMM price to ~$2961.8 (< $2965, > $2950), triggering b1 in round 1,
    # whose forced sale pushes price to ~$2943 (< $2950), triggering b2 in round 2!
    amm_only = simulate_intervention(pool, 720.0, enable_liquidations=False)
    with_cas = simulate_intervention(pool, 720.0, enable_liquidations=True)

    assert 2950.0 < amm_only.post_eth_usd < 2965.0
    assert with_cas.cascade_rounds >= 2
    assert with_cas.liquidated_accounts == 2
    assert with_cas.liquidated_eth > 350.0
    assert with_cas.post_eth_usd < amm_only.post_eth_usd
