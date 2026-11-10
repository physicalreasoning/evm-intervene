# ruff: noqa: E501
"""Vectorized multi-horizon ETH price, order-flow, and crash-regime forecasting models on Ethereum Mainnet.

Couples multi-scale historical order-flow/microstructure sequences with live Uniswap V3
`TickLens` spatial liquidity profiles and Aave V3 liquidation walls to forecast:
  1. Future net signed order flow (ETH) over the next H in {10, 25, 50, 100} swaps (~2.5-25 min)
  2. Future ETH/USD price path ($ USD and log-return) at H in {10, 25, 50, 100} swaps
  3. Downside tail-move / air-pocket crash probability (P(drop > 50 bps over next 50 swaps))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from evm_intervene.constants import POOL_FEE_RATE, USDC_DECIMALS, WETH_DECIMALS
from evm_intervene.simulator import (
    BorrowerPosition,
    PoolState,
    eth_usd_from_raw_sqrt_price,
    raw_sqrt_price_from_x96,
)
from evm_intervene.state import (
    NUM_SIDE_BINS,
    NUM_TOTAL_BINS,
    POSITIVE_LOG_EDGES,
    _active_liquidity_at_eth_price,
    raw_sqrt_price_from_eth_usd,
)

FORECAST_HORIZONS: tuple[int, ...] = (10, 25, 50, 100)
SEQ_LOOKBACK: int = 32


@dataclass(frozen=True)
class SwapEventRecord:
    """Parsed chronological Mainnet swap record with exact pool state."""

    block_number: int
    tx_index: int
    eth_usd: float
    sqrt_price_raw: float
    liquidity: float
    tick: int
    signed_eth: float
    dlogp: float
    block_gap: int
    ticks_crossed: int
    window_name: str


@dataclass(frozen=True)
class ForecastSample:
    """Causal forecasting sample at swap index t predicting horizons H in {10, 25, 50, 100}."""

    block_number: int
    window_name: str
    spot_eth_usd: float
    active_liquidity: float
    # Causal multi-scale order-flow & microstructure feature vector (dim = 38)
    micro_features: np.ndarray
    # Causal sequence of last SEQ_LOOKBACK=32 swaps: shape (32, 6)
    seq_tokens: np.ndarray
    # Spatial TickLens 32-bin relative liquidity, ETH capacity, log_delta, raw ETH cap: shape (32, 4)
    bin_profile: np.ndarray
    # Ground-truth future targets across H in {10, 25, 50, 100}:
    future_dlogp: np.ndarray
    future_eth_usd: np.ndarray
    future_net_flow_eth: np.ndarray
    future_min_dlogp_50: float
    realized_dlogp_10: float
    realized_dlogp_25: float
    realized_dlogp_50: float


def parse_swap_stream(swaps_raw: list[dict], window_name: str) -> list[SwapEventRecord]:
    """Convert raw Mainnet swap dicts into clean chronological SwapEventRecords."""
    records: list[SwapEventRecord] = []
    for i in range(1, len(swaps_raw)):
        prev_s = swaps_raw[i - 1]
        cur_s = swaps_raw[i]
        q_prev = raw_sqrt_price_from_x96(int(prev_s["sqrt_price_x96"]))
        q_cur = raw_sqrt_price_from_x96(int(cur_s["sqrt_price_x96"]))
        p_prev = eth_usd_from_raw_sqrt_price(q_prev)
        p_cur = eth_usd_from_raw_sqrt_price(q_cur)

        amt0 = int(cur_s["amount0_raw"])
        amt1 = int(cur_s["amount1_raw"])
        if amt1 > 0:
            signed_eth = amt1 / (10.0**WETH_DECIMALS)
        elif amt0 > 0:
            usdc_in = amt0 / (10.0**USDC_DECIMALS)
            signed_eth = -(usdc_in / p_prev)
        else:
            continue

        b_cur = int(cur_s["block_number"])
        b_prev = int(prev_s["block_number"])
        dlogp = float(math.log(p_cur / p_prev))
        ticks_crossed = abs(int(cur_s["tick"]) - int(prev_s["tick"])) // 10
        records.append(
            SwapEventRecord(
                block_number=b_cur,
                tx_index=int(cur_s.get("tx_index", 0)),
                eth_usd=p_cur,
                sqrt_price_raw=q_cur,
                liquidity=float(int(cur_s["liquidity"])),
                tick=int(cur_s["tick"]),
                signed_eth=signed_eth,
                dlogp=dlogp,
                block_gap=max(0, b_cur - b_prev),
                ticks_crossed=ticks_crossed,
                window_name=window_name,
            )
        )
    return records


def extract_spatial_bin_profile(
    spot_eth_usd: float,
    sqrt_price_raw: float,
    tick: int,
    liquidity: float,
    block_number: int,
    cp: dict,
    borrowers: tuple[BorrowerPosition, ...] = (),
) -> np.ndarray:
    """Compute (32, 4) spatial TickLens profile around current spot price."""
    state = PoolState(
        block_number=block_number,
        sqrt_price_raw=sqrt_price_raw,
        tick=tick,
        liquidity=max(liquidity, 1.0),
        ticks=tuple(int(t) for t in cp["ticks"]),
        liquidity_net=tuple(float(int(ln)) for ln in cp["liquidity_net"]),
        borrowers=borrowers,
    )
    L0 = max(liquidity, 1.0)
    profile = np.zeros((NUM_TOTAL_BINS, 4), dtype=np.float32)

    prev_u = 0.0
    for b, edge in enumerate(POSITIVE_LOG_EDGES):
        u_hi = -prev_u
        u_lo = -edge
        u_mid = 0.5 * (u_hi + u_lo)
        p_hi = spot_eth_usd * math.exp(u_hi)
        p_lo = spot_eth_usd * math.exp(u_lo)
        p_mid = spot_eth_usd * math.exp(u_mid)

        L_mid = _active_liquidity_at_eth_price(state, p_mid)
        ratio = L_mid / L0
        q_hi = raw_sqrt_price_from_eth_usd(p_hi)
        q_lo = raw_sqrt_price_from_eth_usd(p_lo)
        base_c = (L0 * (q_lo - q_hi)) / ((10.0**WETH_DECIMALS) * (1.0 - POOL_FEE_RATE))
        eth_cap = base_c * ratio
        profile[b, 0] = float(math.log(max(ratio, 1e-4)))
        profile[b, 1] = float(math.log1p(max(eth_cap, 0.0)))
        profile[b, 2] = float(u_lo - u_hi)
        profile[b, 3] = float(eth_cap)
        prev_u = edge

    prev_u = 0.0
    for j, edge in enumerate(POSITIVE_LOG_EDGES):
        b = NUM_SIDE_BINS + j
        u_lo = prev_u
        u_hi = edge
        u_mid = 0.5 * (u_lo + u_hi)
        p_lo = spot_eth_usd * math.exp(u_lo)
        p_hi = spot_eth_usd * math.exp(u_hi)
        p_mid = spot_eth_usd * math.exp(u_mid)

        L_mid = _active_liquidity_at_eth_price(state, p_mid)
        ratio = L_mid / L0
        q_lo = raw_sqrt_price_from_eth_usd(p_lo)
        q_hi = raw_sqrt_price_from_eth_usd(p_hi)
        base_c = (L0 * (1.0 / q_hi - 1.0 / q_lo)) / (
            (10.0**USDC_DECIMALS) * spot_eth_usd * (1.0 - POOL_FEE_RATE)
        )
        eth_cap = base_c * ratio
        profile[b, 0] = float(math.log(max(ratio, 1e-4)))
        profile[b, 1] = float(math.log1p(max(eth_cap, 0.0)))
        profile[b, 2] = float(u_hi - u_lo)
        profile[b, 3] = float(eth_cap)
        prev_u = edge

    return profile


def build_forecast_dataset(
    swaps_by_window: dict[str, list[dict]],
    checkpoints: dict[str, list[dict]] | list[dict],
    borrowers: tuple[BorrowerPosition, ...] = (),
) -> list[ForecastSample]:
    """Build strictly causal multi-horizon forecasting samples across all Mainnet windows."""
    if isinstance(checkpoints, dict):
        all_cps = [c for w in ("calm_june_2024", "crash_aug5_2024", "autumn_oct_2024") for c in checkpoints.get(w, [])]
    else:
        all_cps = checkpoints
    all_cps = sorted(all_cps, key=lambda c: int(c["block_number"]))
    cp_blocks = [int(c["block_number"]) for c in all_cps]
    max_h = max(FORECAST_HORIZONS)
    samples: list[ForecastSample] = []

    for w_name in ("calm_june_2024", "crash_aug5_2024", "autumn_oct_2024"):
        if w_name not in swaps_by_window:
            continue
        recs = parse_swap_stream(swaps_by_window[w_name], w_name)
        n = len(recs)
        if n <= 150 + max_h + 5:
            continue

        signed_arr = np.array([r.signed_eth for r in recs], dtype=np.float64)
        abs_signed = np.abs(signed_arr)
        dlogp_arr = np.array([r.dlogp for r in recs], dtype=np.float64)
        price_arr = np.array([r.eth_usd for r in recs], dtype=np.float64)
        liq_arr = np.array([r.liquidity for r in recs], dtype=np.float64)
        gap_arr = np.array([r.block_gap for r in recs], dtype=np.float64)
        cross_arr = np.array([r.ticks_crossed for r in recs], dtype=np.float64)

        # Precompute full sequence token matrix once per window (vectorized!)
        seq_full = np.zeros((n, 6), dtype=np.float32)
        seq_full[:, 0] = np.arcsinh(signed_arr / 10.0)
        seq_full[:, 1] = dlogp_arr * 1000.0
        seq_full[:, 2] = np.log1p(abs_signed)
        seq_full[:, 3] = np.log1p(gap_arr)
        seq_full[:, 4] = np.log(np.maximum(liq_arr, 1.0) / 1e19)
        seq_full[:, 5] = cross_arr

        # Cache spatial profiles by (cp_idx, tick // 30) so each distinct bucket is computed once
        profile_cache: dict[tuple[int, int], np.ndarray] = {}

        for t in range(150, n - max_h):
            cur = recs[t]
            cp_idx = max(0, int(np.searchsorted(cp_blocks, cur.block_number, side="right")) - 1)
            cache_key = (cp_idx, cur.tick // 30)
            cached_profile = profile_cache.get(cache_key)
            if cached_profile is None:
                cached_profile = extract_spatial_bin_profile(
                    cur.eth_usd,
                    cur.sqrt_price_raw,
                    cur.tick,
                    cur.liquidity,
                    cur.block_number,
                    all_cps[cp_idx],
                    borrowers,
                )
                profile_cache[cache_key] = cached_profile

            feats: list[float] = [
                math.log(cur.eth_usd / 3000.0),
                math.log(max(cur.liquidity, 1.0) / 1e19),
            ]
            for w in (5, 15, 50, 150):
                s_win = signed_arr[t - w + 1 : t + 1]
                a_win = abs_signed[t - w + 1 : t + 1]
                r_win = dlogp_arr[t - w + 1 : t + 1]
                g_win = gap_arr[t - w + 1 : t + 1]
                c_win = cross_arr[t - w + 1 : t + 1]
                net_f = float(np.sum(s_win))
                gross_f = float(np.sum(a_win)) + 1e-6
                cum_ret = float(np.sum(r_win))
                r_vol = float(np.std(r_win))
                mean_gap = float(np.mean(g_win))
                whale_share = float(np.max(a_win) / gross_f)
                cliff_rate = float(np.mean(c_win > 0))
                feats.extend(
                    [
                        math.asinh(net_f / 50.0),
                        math.log1p(gross_f / 50.0),
                        net_f / gross_f,
                        cum_ret * 100.0,
                        r_vol * 1000.0,
                        math.log1p(mean_gap),
                        whale_share,
                        cliff_rate,
                    ]
                )

            down_cap_50bps = float(np.sum(cached_profile[:6, 3]))
            up_cap_50bps = float(np.sum(cached_profile[NUM_SIDE_BINS : NUM_SIDE_BINS + 6, 3]))
            down_cap_300bps = float(np.sum(cached_profile[:10, 3]))
            up_cap_300bps = float(np.sum(cached_profile[NUM_SIDE_BINS : NUM_SIDE_BINS + 10, 3]))
            book_imb_near = (up_cap_50bps - down_cap_50bps) / (up_cap_50bps + down_cap_50bps + 1.0)
            book_imb_deep = (up_cap_300bps - down_cap_300bps) / (up_cap_300bps + down_cap_300bps + 1.0)
            feats.extend(
                [
                    math.log1p(down_cap_50bps / 100.0),
                    math.log1p(up_cap_50bps / 100.0),
                    book_imb_near,
                    book_imb_deep,
                ]
            )

            seq = seq_full[t - SEQ_LOOKBACK + 1 : t + 1].copy()
            seq[:, 4] = seq[:, 4] - float(math.log(max(cur.liquidity, 1.0) / 1e19))

            f_dlogp = np.zeros(len(FORECAST_HORIZONS), dtype=np.float32)
            f_usd = np.zeros(len(FORECAST_HORIZONS), dtype=np.float32)
            f_flow = np.zeros(len(FORECAST_HORIZONS), dtype=np.float32)
            for h_idx, h in enumerate(FORECAST_HORIZONS):
                p_fut = price_arr[t + h]
                f_usd[h_idx] = float(p_fut)
                f_dlogp[h_idx] = float(math.log(p_fut / cur.eth_usd))
                f_flow[h_idx] = float(np.sum(signed_arr[t + 1 : t + h + 1]))

            window_prices_50 = price_arr[t + 1 : t + 51]
            min_dlogp_50 = float(math.log(float(np.min(window_prices_50)) / cur.eth_usd))

            samples.append(
                ForecastSample(
                    block_number=cur.block_number,
                    window_name=w_name,
                    spot_eth_usd=cur.eth_usd,
                    active_liquidity=cur.liquidity,
                    micro_features=np.array(feats, dtype=np.float32),
                    seq_tokens=seq,
                    bin_profile=cached_profile,
                    future_dlogp=f_dlogp,
                    future_eth_usd=f_usd,
                    future_net_flow_eth=f_flow,
                    future_min_dlogp_50=min_dlogp_50,
                    realized_dlogp_10=float(f_dlogp[0]),
                    realized_dlogp_25=float(f_dlogp[1]),
                    realized_dlogp_50=float(f_dlogp[2]),
                )
            )
    return samples


class OrderFlowMLPForecaster(nn.Module):
    """Time-series MLP predicting future net order flow and log-price returns from history alone."""

    def __init__(self, in_dim: int = 38, hidden_dim: int = 192) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.flow_head = nn.Linear(hidden_dim // 2, len(FORECAST_HORIZONS))
        self.ret_head = nn.Linear(hidden_dim // 2, len(FORECAST_HORIZONS))
        self.crash_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self, micro: torch.Tensor, seq: torch.Tensor, bins: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(micro)
        return self.flow_head(h), self.ret_head(h), self.crash_head(h).squeeze(-1)


class OrderFlowTransformerForecaster(nn.Module):
    """Causal sequence Transformer attending over the last 32 trades + microstructure state."""

    def __init__(self, micro_dim: int = 38, d_model: int = 128, nhead: int = 4, num_layers: int = 3) -> None:
        super().__init__()
        self.seq_proj = nn.Linear(6, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LOOKBACK + 1, d_model) * 0.02)
        self.ctx_proj = nn.Linear(micro_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.flow_head = nn.Linear(d_model, len(FORECAST_HORIZONS))
        self.ret_head = nn.Linear(d_model, len(FORECAST_HORIZONS))
        self.crash_head = nn.Linear(d_model, 1)

    def forward(
        self, micro: torch.Tensor, seq: torch.Tensor, bins: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_tok = self.ctx_proj(micro).unsqueeze(1)
        s_tok = self.seq_proj(seq)
        tokens = torch.cat([ctx_tok, s_tok], dim=1) + self.pos_emb
        out = self.norm(self.encoder(tokens))
        pooled = 0.5 * (out[:, 0, :] + out[:, -1, :])
        return self.flow_head(pooled), self.ret_head(pooled), self.crash_head(pooled).squeeze(-1)


class LiquidityCoupledWorldForecaster(nn.Module):
    """Vectorized Coupled World-Model Forecaster: predicts future order flow AND integrates it across the TickLens book."""

    def __init__(self, micro_dim: int = 38, d_model: int = 128) -> None:
        super().__init__()
        self.seq_gru = nn.GRU(input_size=6, hidden_size=d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.bin_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
        )
        self.fusion = nn.Sequential(
            nn.Linear(micro_dim + d_model + 32, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.flow_head = nn.Linear(d_model, len(FORECAST_HORIZONS))
        self.flow_scale = nn.Parameter(torch.ones(len(FORECAST_HORIZONS)) * 1.1)
        self.phys_weight = nn.Parameter(torch.ones(len(FORECAST_HORIZONS)) * 0.75)
        self.residual_ret_head = nn.Linear(d_model + len(FORECAST_HORIZONS), len(FORECAST_HORIZONS))
        self.crash_head = nn.Linear(d_model + len(FORECAST_HORIZONS), 1)

    @staticmethod
    def integrate_flow_on_book(
        net_flow_eth: torch.Tensor,
        bins: torch.Tensor,
    ) -> torch.Tensor:
        """Fully vectorized integration of predicted net order flow (B, 4) across 32-bin TickLens book (B, 32, 4)."""
        down_cap = torch.clamp(bins[:, :NUM_SIDE_BINS, 3], min=1.0).unsqueeze(1)  # (B, 1, 16)
        down_dlogp = bins[:, :NUM_SIDE_BINS, 2].unsqueeze(1)  # (B, 1, 16)
        up_cap = torch.clamp(bins[:, NUM_SIDE_BINS:, 3], min=1.0).unsqueeze(1)  # (B, 1, 16)
        up_dlogp = bins[:, NUM_SIDE_BINS:, 2].unsqueeze(1)  # (B, 1, 16)

        cum_down_hi = torch.cumsum(down_cap, dim=2)  # (B, 1, 16)
        cum_down_lo = cum_down_hi - down_cap
        cum_up_hi = torch.cumsum(up_cap, dim=2)
        cum_up_lo = cum_up_hi - up_cap

        sell_flow = torch.clamp(net_flow_eth, min=0.0).unsqueeze(2)  # (B, 4, 1)
        buy_flow = torch.clamp(-net_flow_eth, min=0.0).unsqueeze(2)  # (B, 4, 1)

        used_down = torch.clamp(sell_flow - cum_down_lo, min=0.0, max=None)
        used_down = torch.minimum(used_down, down_cap)
        frac_down = used_down / down_cap
        dlogp_sell = torch.sum(frac_down * down_dlogp, dim=2)
        rem_sell = torch.clamp(sell_flow.squeeze(2) - cum_down_hi[:, 0, -1:], min=0.0)
        dlogp_sell = dlogp_sell - (rem_sell / (down_cap[:, 0, -1:] + 100.0)) * 0.02

        used_up = torch.clamp(buy_flow - cum_up_lo, min=0.0, max=None)
        used_up = torch.minimum(used_up, up_cap)
        frac_up = used_up / up_cap
        dlogp_buy = torch.sum(frac_up * up_dlogp, dim=2)
        rem_buy = torch.clamp(buy_flow.squeeze(2) - cum_up_hi[:, 0, -1:], min=0.0)
        dlogp_buy = dlogp_buy + (rem_buy / (up_cap[:, 0, -1:] + 100.0)) * 0.02

        return (dlogp_sell + dlogp_buy) * 1e4

    def forward(
        self, micro: torch.Tensor, seq: torch.Tensor, bins: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h_seq = self.seq_gru(seq)
        seq_emb = h_seq[-1]
        bin_emb = self.bin_encoder(bins[:, :, :3]).mean(dim=1)
        h = self.fusion(torch.cat([micro, seq_emb, bin_emb], dim=-1))

        pred_flow_asinh = self.flow_head(h)
        pred_flow_eth = torch.sinh(torch.clamp(pred_flow_asinh, -5.5, 5.5)) * 50.0 * self.flow_scale
        phys_bps = self.integrate_flow_on_book(pred_flow_eth, bins)

        cat_h = torch.cat([h, phys_bps / 100.0], dim=-1)
        res_bps = self.residual_ret_head(cat_h)
        total_ret_bps = self.phys_weight * phys_bps + res_bps
        crash_logit = self.crash_head(cat_h).squeeze(-1)
        return pred_flow_asinh, total_ret_bps, crash_logit
