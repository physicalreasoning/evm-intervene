"""Deterministic Uniswap v3 concentrated-liquidity + Aave v3 liquidation cascade state machine.

Implements the exact two-stage transition operator s_{t+1} = (F_cas o F_AMM)(s_t, a_t)
for the Mainnet USDC/WETH 0.05% pool (token0 = USDC [6 dec], token1 = WETH [18 dec]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from evm_intervene.constants import (
    AAVE_FULL_LIQ_HF_CUTOFF,
    AAVE_LIQUIDATION_BONUS,
    DECIMAL_SCALE,
    POOL_FEE_RATE,
    Q96,
    USDC_DECIMALS,
    WETH_DECIMALS,
)

LN_1_0001 = math.log(1.0001)


def raw_sqrt_price_from_x96(sqrt_price_x96: int | float) -> float:
    """Convert Uniswap v3 Q64.96 sqrtPriceX96 to raw sqrt(token1_raw / token0_raw)."""
    return float(sqrt_price_x96) / float(Q96)


def x96_from_raw_sqrt_price(q_raw: float) -> int:
    """Convert raw sqrt(token1_raw / token0_raw) back to integer sqrtPriceX96."""
    return max(1, int(round(q_raw * float(Q96))))


def eth_usd_from_raw_sqrt_price(q_raw: float) -> float:
    """Convert raw sqrt(token1/token0) to human ETH price in USDC: 1e12 / q_raw^2."""
    return float(DECIMAL_SCALE) / (q_raw * q_raw)


def raw_sqrt_price_from_eth_usd(eth_usd: float) -> float:
    """Convert human ETH price in USDC to raw sqrt(token1/token0): sqrt(1e12 / P)."""
    return math.sqrt(float(DECIMAL_SCALE) / eth_usd)


def tick_to_raw_sqrt_price(tick: int) -> float:
    """Exact sqrt(1.0001^tick) in raw pool coordinates."""
    return math.exp(0.5 * tick * LN_1_0001)


def raw_sqrt_price_to_tick(q_raw: float) -> int:
    """Uniswap v3 floor(log_{1.0001}(q_raw^2))."""
    return int(math.floor(2.0 * math.log(q_raw) / LN_1_0001 + 1e-12))


@dataclass(frozen=True)
class BorrowerPosition:
    """Leveraged WETH-collateral, USDC-debt position on Aave v3."""

    collateral_eth: float
    debt_usdc: float
    liquidation_threshold: float  # e.g., 0.825

    @property
    def critical_eth_price(self) -> float:
        """ETH/USDC price below which Health Factor < 1.0."""
        denom = self.collateral_eth * self.liquidation_threshold
        if denom <= 0.0:
            return 0.0
        return self.debt_usdc / denom

    def health_factor(self, eth_usd: float) -> float:
        if self.debt_usdc <= 0.0:
            return float("inf")
        return (self.collateral_eth * eth_usd * self.liquidation_threshold) / self.debt_usdc


@dataclass(frozen=True)
class PoolState:
    """Complete deterministic snapshot of AMM pool + leveraged borrower book at block t."""

    block_number: int
    sqrt_price_raw: float  # q_t = sqrtPriceX96 / 2^96
    tick: int
    liquidity: float  # active uint128 liquidity L_t
    ticks: tuple[int, ...]  # sorted initialized tick indices
    liquidity_net: tuple[float, ...]  # aligned int128 liquidityNet at each initialized tick
    borrowers: tuple[BorrowerPosition, ...] = ()
    fee_rate: float = POOL_FEE_RATE

    @property
    def eth_usd(self) -> float:
        return eth_usd_from_raw_sqrt_price(self.sqrt_price_raw)

    def max_in_tick_eth_sell(self) -> float:
        """Maximum WETH sell input (in ETH) before crossing the next upward raw tick."""
        idx = int(np.searchsorted(self.ticks, self.tick, side="right"))
        if idx >= len(self.ticks):
            return float("inf")
        q_target = tick_to_raw_sqrt_price(self.ticks[idx])
        if q_target <= self.sqrt_price_raw:
            return 0.0
        dy_raw = self.liquidity * (q_target - self.sqrt_price_raw) / (1.0 - self.fee_rate)
        return dy_raw / (10.0**WETH_DECIMALS)

    def max_in_tick_eth_buy_equiv(self) -> float:
        """Maximum WETH-equivalent buy input (USDC / P_t) before crossing next downward raw tick."""
        idx = int(np.searchsorted(self.ticks, self.tick, side="right")) - 1
        if idx < 0:
            return float("inf")
        q_target = tick_to_raw_sqrt_price(self.ticks[idx])
        if q_target >= self.sqrt_price_raw:
            idx -= 1
            if idx < 0:
                return float("inf")
            q_target = tick_to_raw_sqrt_price(self.ticks[idx])
        dx_raw = self.liquidity * (1.0 / q_target - 1.0 / self.sqrt_price_raw) / (1.0 - self.fee_rate)
        usdc_in = dx_raw / (10.0**USDC_DECIMALS)
        return usdc_in / self.eth_usd


@dataclass(frozen=True)
class TransitionOutcome:
    """Exact result of s_{t+1} = T(s_t, a_t)."""

    pre_eth_usd: float
    post_amm_eth_usd: float
    post_eth_usd: float
    pre_liquidity: float
    post_liquidity: float
    post_sqrt_price_raw: float
    post_tick: int
    action_eth_signed: float  # >0: sell WETH, <0: buy WETH (USDC in / P_0)
    usdc_delta: float  # >0: USDC paid out to seller, <0: USDC paid in by buyer
    weth_delta: float  # >0: WETH paid in by seller, <0: WETH paid out to buyer
    ticks_crossed: int
    cascade_rounds: int
    liquidated_eth: float
    liquidated_accounts: int

    @property
    def delta_log_price(self) -> float:
        return math.log(self.post_eth_usd / self.pre_eth_usd)

    @property
    def delta_log_liquidity(self) -> float:
        return math.log(max(self.post_liquidity, 1.0) / max(self.pre_liquidity, 1.0))

    @property
    def execution_efficiency(self) -> float:
        """Realized USD value ratio |Delta USDC| / (P_0 * |action_eth|); <= 1 under slippage."""
        notional = self.pre_eth_usd * abs(self.action_eth_signed)
        if notional <= 1e-12:
            return 1.0
        if self.action_eth_signed >= 0:
            return abs(self.usdc_delta) / notional
        return (abs(self.weth_delta) * self.pre_eth_usd) / notional


def simulate_amm_swap_raw(
    state: PoolState,
    *,
    zero_for_one: bool,
    amount_in_raw: float,
) -> tuple[float, int, float, float, float, int]:
    """Execute Stage 1 (F_AMM) exact input swap in raw Uniswap v3 coordinates.

    Parameters
    ----------
    zero_for_one:
        True  -> exact input token0 (USDC raw), q decreases, ETH/USD price increases.
        False -> exact input token1 (WETH raw), q increases, ETH/USD price decreases.
    amount_in_raw:
        Positive input amount in raw token units (1e6 for USDC, 1e18 for WETH).

    Returns
    -------
    (q_post, tick_post, liquidity_post, amount0_signed_raw, amount1_signed_raw, ticks_crossed)
    """
    if amount_in_raw <= 0.0:
        return (
            state.sqrt_price_raw,
            state.tick,
            state.liquidity,
            0.0,
            0.0,
            0,
        )

    q = state.sqrt_price_raw
    tick = state.tick
    L = state.liquidity
    fee = state.fee_rate
    ticks = state.ticks
    liq_net = state.liquidity_net
    n_ticks = len(ticks)

    rem = float(amount_in_raw)
    in_used = 0.0
    out_acc = 0.0
    crossed = 0

    if not zero_for_one:
        # Selling token1 (WETH) -> q moves UP toward higher initialized ticks.
        idx = int(np.searchsorted(ticks, tick, side="right"))
        while rem > 0.0 and L > 0.0:
            if idx < n_ticks:
                next_tick = ticks[idx]
                q_target = tick_to_raw_sqrt_price(next_tick)
            else:
                next_tick = None
                q_target = tick_to_raw_sqrt_price(887272)

            if q_target <= q:
                if next_tick is not None:
                    L = max(1.0, L + liq_net[idx])
                    tick = next_tick
                    idx += 1
                    crossed += 1
                    continue
                break

            avail_less_fee = rem * (1.0 - fee)
            dy_max = L * (q_target - q)

            if avail_less_fee < dy_max or next_tick is None:
                q_next = q + avail_less_fee / L
                dx_out = L * (1.0 / q - 1.0 / q_next)
                in_used += rem
                out_acc += dx_out
                q = q_next
                tick = raw_sqrt_price_to_tick(q)
                rem = 0.0
                break
            else:
                consumed = dy_max / (1.0 - fee)
                dx_out = L * (1.0 / q - 1.0 / q_target)
                rem = max(0.0, rem - consumed)
                in_used += consumed
                out_acc += dx_out
                q = q_target
                tick = next_tick
                L = max(1.0, L + liq_net[idx])
                idx += 1
                crossed += 1

        # Pool received +in_used token1 (WETH) and paid out -out_acc token0 (USDC).
        return q, tick, L, -out_acc, +in_used, crossed

    else:
        # Spending token0 (USDC) -> q moves DOWN toward lower initialized ticks.
        idx = int(np.searchsorted(ticks, tick, side="right")) - 1
        while idx >= 0 and tick_to_raw_sqrt_price(ticks[idx]) >= q:
            if tick_to_raw_sqrt_price(ticks[idx]) == q:
                # Standing right on an initialized tick boundary moving left: cross immediately.
                L = max(1.0, L - liq_net[idx])
                tick = ticks[idx] - 1
                crossed += 1
            idx -= 1

        while rem > 0.0 and L > 0.0:
            if idx >= 0:
                next_tick = ticks[idx]
                q_target = tick_to_raw_sqrt_price(next_tick)
            else:
                next_tick = None
                q_target = tick_to_raw_sqrt_price(-887272)

            avail_less_fee = rem * (1.0 - fee)
            dx_max = L * (1.0 / q_target - 1.0 / q)

            if avail_less_fee < dx_max or next_tick is None:
                inv_q_next = (1.0 / q) + (avail_less_fee / L)
                q_next = 1.0 / inv_q_next
                dy_out = L * (q - q_next)
                in_used += rem
                out_acc += dy_out
                q = q_next
                tick = raw_sqrt_price_to_tick(q)
                rem = 0.0
                break
            else:
                consumed = dx_max / (1.0 - fee)
                dy_out = L * (q - q_target)
                rem = max(0.0, rem - consumed)
                in_used += consumed
                out_acc += dy_out
                q = q_target
                tick = next_tick - 1
                L = max(1.0, L - liq_net[idx])
                idx -= 1
                crossed += 1

        # Pool received +in_used token0 (USDC) and paid out -out_acc token1 (WETH).
        return q, tick, L, +in_used, -out_acc, crossed


def simulate_intervention(
    state: PoolState,
    action_eth_signed: float,
    *,
    enable_liquidations: bool = True,
    liquidation_bonus: float = AAVE_LIQUIDATION_BONUS,
    max_cascade_rounds: int = 16,
) -> TransitionOutcome:
    """Execute complete two-stage transition s_{t+1} = (F_cas o F_AMM)(s_t, a_t).

    Parameters
    ----------
    state:
        Frozen snapshot of AMM pool + leveraged borrower positions.
    action_eth_signed:
        > 0: Sell `action_eth_signed` WETH into the pool (pushes ETH/USD down).
        < 0: Buy WETH by spending `|action_eth_signed| * P_t` USDC (pushes ETH/USD up).
    enable_liquidations:
        If True, executes Stage 2 endogenous Aave v3 collateral liquidations until convergence.
    """
    pre_eth_usd = state.eth_usd
    pre_liq = state.liquidity

    if abs(action_eth_signed) <= 1e-15:
        return TransitionOutcome(
            pre_eth_usd=pre_eth_usd,
            post_amm_eth_usd=pre_eth_usd,
            post_eth_usd=pre_eth_usd,
            pre_liquidity=pre_liq,
            post_liquidity=pre_liq,
            post_sqrt_price_raw=state.sqrt_price_raw,
            post_tick=state.tick,
            action_eth_signed=0.0,
            usdc_delta=0.0,
            weth_delta=0.0,
            ticks_crossed=0,
            cascade_rounds=0,
            liquidated_eth=0.0,
            liquidated_accounts=0,
        )

    if action_eth_signed > 0.0:
        amount_in_raw = action_eth_signed * (10.0**WETH_DECIMALS)
        q, tick, L, amt0_raw, amt1_raw, crossed = simulate_amm_swap_raw(
            state, zero_for_one=False, amount_in_raw=amount_in_raw
        )
        usdc_delta = -amt0_raw / (10.0**USDC_DECIMALS)  # positive USDC received by trader
        weth_delta = amt1_raw / (10.0**WETH_DECIMALS)
    else:
        usdc_in = abs(action_eth_signed) * pre_eth_usd
        amount_in_raw = usdc_in * (10.0**USDC_DECIMALS)
        q, tick, L, amt0_raw, amt1_raw, crossed = simulate_amm_swap_raw(
            state, zero_for_one=True, amount_in_raw=amount_in_raw
        )
        usdc_delta = -amt0_raw / (10.0**USDC_DECIMALS)  # negative (USDC spent)
        weth_delta = amt1_raw / (10.0**WETH_DECIMALS)  # negative (WETH received)

    post_amm_eth_usd = eth_usd_from_raw_sqrt_price(q)
    cur_eth_usd = post_amm_eth_usd

    cascade_rounds = 0
    total_liq_eth = 0.0
    liquidated_ids: set[int] = set()

    if enable_liquidations and len(state.borrowers) > 0 and cur_eth_usd < pre_eth_usd:
        # Mutable copies of borrower collateral & debt for fixed-point cascade iteration.
        coll = [b.collateral_eth for b in state.borrowers]
        debt = [b.debt_usdc for b in state.borrowers]
        lt = [b.liquidation_threshold for b in state.borrowers]

        cur_pool = PoolState(
            block_number=state.block_number,
            sqrt_price_raw=q,
            tick=tick,
            liquidity=L,
            ticks=state.ticks,
            liquidity_net=state.liquidity_net,
            borrowers=(),
            fee_rate=state.fee_rate,
        )

        for _ in range(max_cascade_rounds):
            forced_eth = 0.0
            for idx_b in range(len(coll)):
                if coll[idx_b] <= 1e-9 or debt[idx_b] <= 1e-6:
                    continue
                hf = (coll[idx_b] * cur_eth_usd * lt[idx_b]) / debt[idx_b]
                if hf < 1.0:
                    close_factor = 1.0 if hf < AAVE_FULL_LIQ_HF_CUTOFF else 0.5
                    repay_usdc = close_factor * debt[idx_b]
                    seize_eth = min(
                        coll[idx_b],
                        (repay_usdc * (1.0 + liquidation_bonus)) / cur_eth_usd,
                    )
                    if seize_eth > 1e-9:
                        coll[idx_b] -= seize_eth
                        debt[idx_b] -= repay_usdc
                        forced_eth += seize_eth
                        liquidated_ids.add(idx_b)

            if forced_eth <= 1e-9:
                break

            cascade_rounds += 1
            total_liq_eth += forced_eth
            q, tick, L, _, _, extra_crossed = simulate_amm_swap_raw(
                cur_pool,
                zero_for_one=False,
                amount_in_raw=forced_eth * (10.0**WETH_DECIMALS),
            )
            crossed += extra_crossed
            cur_eth_usd = eth_usd_from_raw_sqrt_price(q)
            cur_pool = PoolState(
                block_number=state.block_number,
                sqrt_price_raw=q,
                tick=tick,
                liquidity=L,
                ticks=state.ticks,
                liquidity_net=state.liquidity_net,
                borrowers=(),
                fee_rate=state.fee_rate,
            )

    return TransitionOutcome(
        pre_eth_usd=pre_eth_usd,
        post_amm_eth_usd=post_amm_eth_usd,
        post_eth_usd=cur_eth_usd,
        pre_liquidity=pre_liq,
        post_liquidity=L,
        post_sqrt_price_raw=q,
        post_tick=tick,
        action_eth_signed=action_eth_signed,
        usdc_delta=usdc_delta,
        weth_delta=weth_delta,
        ticks_crossed=crossed,
        cascade_rounds=cascade_rounds,
        liquidated_eth=total_liq_eth,
        liquidated_accounts=len(liquidated_ids),
    )
