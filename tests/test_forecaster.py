"""Unit tests for the multi-horizon ETH price, order-flow, and liquidity-book forecaster."""

from __future__ import annotations

import torch

from evm_intervene.forecaster import (
    FORECAST_HORIZONS,
    SEQ_LOOKBACK,
    LiquidityCoupledWorldForecaster,
    OrderFlowMLPForecaster,
    OrderFlowTransformerForecaster,
)
from evm_intervene.state import NUM_SIDE_BINS, NUM_TOTAL_BINS


def _dummy_batch(batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    micro = torch.randn(batch_size, 38)
    seq = torch.randn(batch_size, SEQ_LOOKBACK, 6)
    bins = torch.zeros(batch_size, NUM_TOTAL_BINS, 4)
    # Downside bins: negative log_delta, positive ETH capacity
    bins[:, :NUM_SIDE_BINS, 2] = -0.005
    bins[:, :NUM_SIDE_BINS, 3] = 120.0
    # Upside bins: positive log_delta, positive ETH capacity
    bins[:, NUM_SIDE_BINS:, 2] = 0.005
    bins[:, NUM_SIDE_BINS:, 3] = 120.0
    return micro, seq, bins


def test_vectorized_book_integration_monotonicity() -> None:
    """Selling more WETH (>0 flow) must monotonically lower predicted price impact (<0 bps)."""
    _, _, bins = _dummy_batch(batch_size=3)
    flow_eth = torch.tensor(
        [
            [50.0, 150.0, 400.0, 900.0],   # WETH sell across increasing sizes
            [-50.0, -150.0, -400.0, -900.0],  # WETH buy across increasing sizes
            [0.0, 0.0, 0.0, 0.0],          # Zero net order flow
        ],
        dtype=torch.float32,
    )
    impact_bps = LiquidityCoupledWorldForecaster.integrate_flow_on_book(flow_eth, bins)
    assert impact_bps.shape == (3, len(FORECAST_HORIZONS))

    # Sell flow produces negative price impact that grows in magnitude with flow
    assert torch.all(impact_bps[0] < 0.0)
    assert float(impact_bps[0, 3]) < float(impact_bps[0, 2]) < float(impact_bps[0, 1]) < float(impact_bps[0, 0])

    # Buy flow produces positive price impact that grows in magnitude with flow
    assert torch.all(impact_bps[1] > 0.0)
    assert float(impact_bps[1, 3]) > float(impact_bps[1, 2]) > float(impact_bps[1, 1]) > float(impact_bps[1, 0])

    # Zero net flow produces exact zero physical impact
    assert torch.allclose(impact_bps[2], torch.zeros(len(FORECAST_HORIZONS)), atol=1e-6)


def test_forecaster_architectures_forward_and_gradients() -> None:
    """All three forecasting architectures must output (flow, ret_bps, crash_logit) with valid gradients."""
    micro, seq, bins = _dummy_batch(batch_size=6)
    for model in (
        OrderFlowMLPForecaster(in_dim=38),
        OrderFlowTransformerForecaster(micro_dim=38),
        LiquidityCoupledWorldForecaster(micro_dim=38),
    ):
        pf, pr, pc = model(micro, seq, bins)
        assert pf.shape == (6, len(FORECAST_HORIZONS))
        assert pr.shape == (6, len(FORECAST_HORIZONS))
        assert pc.shape == (6,)
        loss = pf.pow(2).mean() + pr.pow(2).mean() + pc.pow(2).mean()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and all(g >= 0.0 for g in grad_norms)
