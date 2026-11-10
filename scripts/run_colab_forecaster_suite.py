# ruff: noqa: E501
"""Colab Pro Suite: Next-N-Minutes ETH Price, Order-Flow, and Crash-Warning Forecaster.

Runs 4 comprehensive forecasting & trading experiments on 31,283 real Ethereum Mainnet swaps
(including the August 5, 2024 ETH crash from $2,748 -> $2,092 and October 2024):
  1. Multi-Horizon Unconditional Order-Flow & ETH Price Forecasting (H = 10, 25, 50, 100 swaps)
  2. Flow-Conditioned ETH Price Path Forecasting (Order-Flow + Live TickLens Book Integration)
  3. Pretrained Qwen2.5 LoRA/QLoRA Foundation-Model Price & Order-Flow Forecasters (0.5B, 1.5B, 3B)
  4. Fee-Aware DeFi Trading Strategy, Smart Execution Timing, & Flash-Crash Early Warning Backtest

Every experiment saves atomically to `--out-dir` (mapped to Google Drive in Colab) after each model.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from evm_intervene.forecaster import (
    FORECAST_HORIZONS,
    ForecastSample,
    LiquidityCoupledWorldForecaster,
    OrderFlowMLPForecaster,
    OrderFlowTransformerForecaster,
    build_forecast_dataset,
)
from evm_intervene.simulator import BorrowerPosition


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_borrowers(path: Path) -> tuple[BorrowerPosition, ...]:
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        BorrowerPosition(
            collateral_eth=float(b["collateral_eth"]),
            debt_usdc=float(b["debt_usdc"]),
            liquidation_threshold=float(b["liquidation_threshold"]),
        )
        for b in raw
    )


def _pack_tensors(
    samples: list[ForecastSample],
    micro_mean: np.ndarray,
    micro_std: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    micro = np.stack([s.micro_features for s in samples], axis=0)
    micro_norm = (micro - micro_mean) / micro_std
    seq = np.stack([s.seq_tokens for s in samples], axis=0)
    bins = np.stack([s.bin_profile for s in samples], axis=0)

    flow_eth = np.stack([s.future_net_flow_eth for s in samples], axis=0)
    flow_asinh = np.arcsinh(flow_eth / 50.0).astype(np.float32)
    ret_bps = (np.stack([s.future_dlogp for s in samples], axis=0) * 1e4).astype(np.float32)
    crash_label = np.array(
        [1.0 if s.future_min_dlogp_50 * 1e4 < -50.0 else 0.0 for s in samples],
        dtype=np.float32,
    )

    return {
        "micro": torch.tensor(micro_norm, dtype=torch.float32, device=device),
        "seq": torch.tensor(seq, dtype=torch.float32, device=device),
        "bins": torch.tensor(bins, dtype=torch.float32, device=device),
        "flow_eth": torch.tensor(flow_eth, dtype=torch.float32, device=device),
        "flow_asinh": torch.tensor(flow_asinh, dtype=torch.float32, device=device),
        "ret_bps": torch.tensor(ret_bps, dtype=torch.float32, device=device),
        "crash_label": torch.tensor(crash_label, dtype=torch.float32, device=device),
    }


def _roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    pos_ranks = ranks[: len(pos)]
    auc = (np.sum(pos_ranks) - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def train_forecaster(
    model_kind: str,
    train_t: dict[str, torch.Tensor],
    val_t: dict[str, torch.Tensor],
    *,
    seed: int,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1.5e-3,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = train_t["micro"].device
    micro_dim = int(train_t["micro"].shape[1])

    if model_kind == "orderflow_mlp":
        model: nn.Module = OrderFlowMLPForecaster(in_dim=micro_dim).to(device)
    elif model_kind == "orderflow_transformer":
        model = OrderFlowTransformerForecaster(micro_dim=micro_dim).to(device)
    elif model_kind == "liquidity_coupled_wm":
        model = LiquidityCoupledWorldForecaster(micro_dim=micro_dim).to(device)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber_flow = nn.HuberLoss(delta=1.0)
    huber_ret = nn.HuberLoss(delta=25.0)
    pos_rate = float(train_t["crash_label"].mean().item())
    pos_weight = torch.tensor([max(1.0, (1.0 - pos_rate) / max(pos_rate, 0.05))], device=device)
    bce_crash = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_train = int(train_t["micro"].shape[0])
    best_val_loss = float("inf")
    best_state = None

    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred_flow, pred_ret, pred_crash = model(
                train_t["micro"][idx], train_t["seq"][idx], train_t["bins"][idx]
            )
            loss_f = huber_flow(pred_flow, train_t["flow_asinh"][idx])
            loss_r = huber_ret(pred_ret, train_t["ret_bps"][idx]) / 25.0
            cos_r = -torch.mean(torch.tanh(pred_ret / 15.0) * torch.tanh(train_t["ret_bps"][idx] / 15.0))
            loss_c = bce_crash(pred_crash, train_t["crash_label"][idx])
            loss = loss_f + loss_r + 0.35 * cos_r + 0.30 * loss_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vf, vr, vc = model(val_t["micro"], val_t["seq"], val_t["bins"])
            vloss = (
                huber_flow(vf, val_t["flow_asinh"])
                + huber_ret(vr, val_t["ret_bps"]) / 25.0
                + 0.25 * bce_crash(vc, val_t["crash_label"])
            ).item()
            if vloss < best_val_loss:
                best_val_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


class FlowConditionedPriceModel(nn.Module):
    """Predicts multi-horizon ETH price change given microstructure state + net order flow F_{t->t+H}."""

    def __init__(self, micro_dim: int, use_ticklens_physics: bool = True) -> None:
        super().__init__()
        self.use_ticklens_physics = use_ticklens_physics
        self.flow_scale = nn.Parameter(torch.ones(len(FORECAST_HORIZONS)) * 0.92)
        self.phys_weight = nn.Parameter(torch.ones(len(FORECAST_HORIZONS)) * 0.88)
        in_d = micro_dim + len(FORECAST_HORIZONS) * (2 if use_ticklens_physics else 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_d, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Linear(160, 96),
            nn.SiLU(),
            nn.Linear(96, len(FORECAST_HORIZONS)),
        )

    def forward(self, micro: torch.Tensor, flow_eth: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        flow_asinh = torch.arcsinh(flow_eth / 50.0)
        if not self.use_ticklens_physics:
            return self.mlp(torch.cat([micro, flow_asinh], dim=-1))
        phys_bps = LiquidityCoupledWorldForecaster.integrate_flow_on_book(
            flow_eth * self.flow_scale, bins
        )
        res_bps = self.mlp(torch.cat([micro, flow_asinh, phys_bps / 100.0], dim=-1))
        return self.phys_weight * phys_bps + res_bps


def evaluate_predictions(
    samples: list[ForecastSample],
    pred_flow_eth: np.ndarray,
    pred_ret_bps: np.ndarray,
    pred_crash_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive price, order-flow, crash-warning, and fee-aware trading metrics."""
    spot = np.array([s.spot_eth_usd for s in samples], dtype=np.float64)
    true_flow = np.stack([s.future_net_flow_eth for s in samples], axis=0)
    true_dlogp = np.stack([s.future_dlogp for s in samples], axis=0)
    true_ret_bps = true_dlogp * 1e4
    true_usd = np.stack([s.future_eth_usd for s in samples], axis=0)

    by_horizon: dict[str, Any] = {}
    for h_idx, h in enumerate(FORECAST_HORIZONS):
        pf = pred_flow_eth[:, h_idx]
        tf = true_flow[:, h_idx]
        pr_bps = pred_ret_bps[:, h_idx]
        tr_bps = true_ret_bps[:, h_idx]

        pred_usd_h = spot * np.exp(pr_bps * 1e-4)
        true_usd_h = true_usd[:, h_idx]

        flow_mae = float(np.mean(np.abs(pf - tf)))
        flow_sign_acc = float(np.mean((pf >= 0) == (tf >= 0)) * 100.0)
        active_flow_mask = np.abs(tf) > 25.0
        flow_sign_active = (
            float(np.mean((pf[active_flow_mask] >= 0) == (tf[active_flow_mask] >= 0)) * 100.0)
            if np.any(active_flow_mask)
            else flow_sign_acc
        )
        rho_flow, _ = spearmanr(pf, tf) if np.std(pf) > 1e-8 else (0.0, 1.0)
        rho_flow = 0.0 if np.isnan(rho_flow) else float(rho_flow)

        usd_mae = float(np.mean(np.abs(pred_usd_h - true_usd_h)))
        usd_rmse = float(np.sqrt(np.mean((pred_usd_h - true_usd_h) ** 2)))
        ret_rmse_bps = float(np.sqrt(np.mean((pr_bps - tr_bps) ** 2)))
        ret_mae_bps = float(np.mean(np.abs(pr_bps - tr_bps)))

        dir_acc_all = float(np.mean((pr_bps >= 0) == (tr_bps >= 0)) * 100.0)
        active_move_mask = np.abs(tr_bps) > 10.0
        dir_acc_active = (
            float(np.mean((pr_bps[active_move_mask] >= 0) == (tr_bps[active_move_mask] >= 0)) * 100.0)
            if np.any(active_move_mask)
            else dir_acc_all
        )
        confident_mask = np.abs(pr_bps) >= np.percentile(np.abs(pr_bps), 70)
        dir_acc_top30 = (
            float(np.mean((pr_bps[confident_mask] >= 0) == (tr_bps[confident_mask] >= 0)) * 100.0)
            if np.any(confident_mask)
            else dir_acc_all
        )
        rho_ret, _ = spearmanr(pr_bps, tr_bps) if np.std(pr_bps) > 1e-8 else (0.0, 1.0)
        rho_ret = 0.0 if np.isnan(rho_ret) else float(rho_ret)

        by_horizon[f"h{h}"] = {
            "horizon_swaps": h,
            "approx_minutes": round(h * 0.25, 1),
            "price_usd_mae": round(usd_mae, 2),
            "price_usd_rmse": round(usd_rmse, 2),
            "return_rmse_bps": round(ret_rmse_bps, 2),
            "return_mae_bps": round(ret_mae_bps, 2),
            "directional_acc_all_pct": round(dir_acc_all, 2),
            "directional_acc_active_pct": round(dir_acc_active, 2),
            "directional_acc_top30_confident_pct": round(dir_acc_top30, 2),
            "spearman_return_ic": round(rho_ret, 4),
            "orderflow_mae_eth": round(flow_mae, 2),
            "orderflow_sign_acc_pct": round(flow_sign_acc, 2),
            "orderflow_sign_active_pct": round(flow_sign_active, 2),
            "spearman_orderflow": round(rho_flow, 4),
        }

    # Crash / Air-pocket early warning evaluation (drop > 50 bps over next 50 swaps)
    true_crash = np.array([1 if s.future_min_dlogp_50 * 1e4 < -50.0 else 0 for s in samples])
    auc = _roc_auc_score_np(true_crash, pred_crash_prob)
    thresh = float(np.percentile(pred_crash_prob, 75))
    pred_pos = pred_crash_prob >= thresh
    tp = int(np.sum((pred_pos == 1) & (true_crash == 1)))
    fp = int(np.sum((pred_pos == 1) & (true_crash == 0)))
    fn = int(np.sum((pred_pos == 0) & (true_crash == 1)))
    prec = tp / max(tp + fp, 1) * 100.0
    rec = tp / max(tp + fn, 1) * 100.0

    # Fee-Aware High-Conviction Trading Strategy & Execution Timing ($100,000 Notional, 50-swap blocks)
    step = 50
    h50_idx = FORECAST_HORIZONS.index(50)
    notional_usd = 100_000.0
    pool_fee_bps = 5.0  # Uniswap V3 0.05% pool fee

    pr_50 = pred_ret_bps[::step, h50_idx]
    tr_50 = true_ret_bps[::step, h50_idx]
    cp_50 = pred_crash_prob[::step]

    # Only enter/flip positions when expected move exceeds the 10 bps round-trip pool fee hurdle
    fee_hurdle_bps = 11.0
    positions = np.zeros_like(pr_50)
    cur_pos = 0.0
    for k in range(len(pr_50)):
        if pr_50[k] >= fee_hurdle_bps and cp_50[k] < thresh:
            cur_pos = 1.0
        elif pr_50[k] <= -fee_hurdle_bps:
            cur_pos = -1.0
        elif abs(pr_50[k]) < 3.0:
            cur_pos = 0.0
        positions[k] = cur_pos

    gross_pnl_bps = positions * tr_50
    pos_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
    fee_bps = pos_changes * pool_fee_bps
    net_pnl_bps = gross_pnl_bps - fee_bps

    gross_pnl_usd = gross_pnl_bps * (notional_usd * 1e-4)
    net_pnl_usd = net_pnl_bps * (notional_usd * 1e-4)
    cum_net_pnl_usd = np.cumsum(net_pnl_usd)
    cum_gross_pnl_usd = np.cumsum(gross_pnl_usd)

    active_trades = positions != 0
    n_trades = int(np.sum(active_trades))
    win_rate = float(np.mean(gross_pnl_bps[active_trades] > 0) * 100.0) if n_trades > 0 else 0.0
    mean_ret = float(np.mean(net_pnl_bps))
    std_ret = float(np.std(net_pnl_bps)) + 1e-8
    sharpe = float((mean_ret / std_ret) * math.sqrt(120.0 * 365.0 / 10.0))

    # Smart Execution Timing Savings across all 25-swap windows:
    step25 = 25
    h25_idx = FORECAST_HORIZONS.index(25)
    pr_25 = pred_ret_bps[::step25, h25_idx]
    tr_25 = true_ret_bps[::step25, h25_idx]
    timed_exec_improvement_bps = np.where(pr_25 > 0, tr_25, 0.0) - 0.5 * tr_25
    mean_exec_savings_bps = float(np.mean(timed_exec_improvement_bps))
    total_exec_savings_usd = float(np.sum(timed_exec_improvement_bps) * (notional_usd * 1e-4))

    return {
        "by_horizon": by_horizon,
        "crash_early_warning_50swaps": {
            "base_crash_rate_pct": round(float(np.mean(true_crash) * 100.0), 2),
            "roc_auc": round(auc, 4),
            "top25pct_alert_precision_pct": round(prec, 2),
            "top25pct_alert_recall_pct": round(rec, 2),
        },
        "trading_backtest_h50": {
            "notional_usd": notional_usd,
            "n_intervals": len(pr_50),
            "n_active_intervals": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "gross_pnl_usd": round(float(np.sum(gross_pnl_usd)), 2),
            "net_pnl_usd": round(float(np.sum(net_pnl_usd)), 2),
            "annualized_sharpe": round(sharpe, 2),
            "execution_timing_mean_savings_bps": round(mean_exec_savings_bps, 2),
            "execution_timing_total_savings_usd": round(total_exec_savings_usd, 2),
            "cum_net_pnl_curve_usd": [round(float(x), 2) for x in cum_net_pnl_usd],
            "cum_gross_pnl_curve_usd": [round(float(x), 2) for x in cum_gross_pnl_usd],
        },
    }


def run_flow_conditioned_experiment(
    train_t: dict[str, torch.Tensor],
    val_t: dict[str, torch.Tensor],
    test_t: dict[str, torch.Tensor],
    test_samples: list[ForecastSample],
    pred_flows_by_model: dict[str, np.ndarray],
    out_path: Path,
) -> dict[str, Any]:
    """Experiment 2: Flow-Conditioned ETH Price Path Forecasting across H in {10, 25, 50, 100}."""
    device = train_t["micro"].device
    micro_dim = int(train_t["micro"].shape[1])
    spot = np.array([s.spot_eth_usd for s in test_samples], dtype=np.float64)
    true_usd = np.stack([s.future_eth_usd for s in test_samples], axis=0)
    true_ret_bps = np.stack([s.future_dlogp for s in test_samples], axis=0) * 1e4

    results: dict[str, Any] = {}
    for use_phys, name in [
        (False, "timeseries_mlp_flow_to_price"),
        (True, "liquidity_coupled_ticklens_flow_to_price"),
    ]:
        preds_real_flow = []
        preds_pred_flow = []
        for sd in (101, 202, 303):
            torch.manual_seed(sd)
            np.random.seed(sd)
            model = FlowConditionedPriceModel(micro_dim, use_ticklens_physics=use_phys).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
            huber = nn.HuberLoss(delta=20.0)
            n_tr = int(train_t["micro"].shape[0])
            best_v = float("inf")
            best_st = None
            for _ep in range(35):
                model.train()
                perm = torch.randperm(n_tr, device=device)
                for i in range(0, n_tr, 256):
                    idx = perm[i : i + 256]
                    opt.zero_grad()
                    pr = model(train_t["micro"][idx], train_t["flow_eth"][idx], train_t["bins"][idx])
                    loss = huber(pr, train_t["ret_bps"][idx])
                    loss.backward()
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    vr = model(val_t["micro"], val_t["flow_eth"], val_t["bins"])
                    vl = huber(vr, val_t["ret_bps"]).item()
                    if vl < best_v:
                        best_v = vl
                        best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if best_st is not None:
                model.load_state_dict(best_st)
            model.eval()
            with torch.no_grad():
                # 1. Conditioned on realized order flow
                pr_real = model(test_t["micro"], test_t["flow_eth"], test_t["bins"]).cpu().numpy()
                preds_real_flow.append(pr_real)
                # 2. Conditioned on predicted order flow from LiquidityCoupledWorldForecaster
                pf_tensor = torch.tensor(
                    pred_flows_by_model["liquidity_coupled_wm"], dtype=torch.float32, device=device
                )
                pr_pred = model(test_t["micro"], pf_tensor, test_t["bins"]).cpu().numpy()
                preds_pred_flow.append(pr_pred)

        mean_real = np.mean(preds_real_flow, axis=0)
        mean_pred = np.mean(preds_pred_flow, axis=0)

        by_h_real = {}
        by_h_pred = {}
        for h_idx, h in enumerate(FORECAST_HORIZONS):
            for _tag, arr, target_dict in [
                ("realized_flow", mean_real[:, h_idx], by_h_real),
                ("predicted_flow", mean_pred[:, h_idx], by_h_pred),
            ]:
                p_usd = spot * np.exp(arr * 1e-4)
                t_usd = true_usd[:, h_idx]
                tr_b = true_ret_bps[:, h_idx]
                active = np.abs(tr_b) > 10.0
                dir_acc = float(np.mean((arr[active] >= 0) == (tr_b[active] >= 0)) * 100.0)
                rho, _ = spearmanr(arr, tr_b)
                target_dict[f"h{h}"] = {
                    "horizon_swaps": h,
                    "price_usd_mae": round(float(np.mean(np.abs(p_usd - t_usd))), 2),
                    "price_usd_rmse": round(float(np.sqrt(np.mean((p_usd - t_usd) ** 2))), 2),
                    "return_rmse_bps": round(float(np.sqrt(np.mean((arr - tr_b) ** 2))), 2),
                    "directional_acc_active_pct": round(dir_acc, 2),
                    "spearman_ic": round(float(rho), 4),
                }

        results[name] = {
            "conditioned_on_realized_orderflow": by_h_real,
            "conditioned_on_predicted_orderflow": by_h_pred,
        }

    _atomic_save_json(out_path, results)
    return results


class QwenPriceOrderFlowForecaster(nn.Module):
    """Pretrained Qwen2.5 backbone with LoRA adapters predicting multi-horizon order flow & ETH price."""

    def __init__(self, backbone: nn.Module, micro_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.ctx_proj = nn.Linear(micro_dim, hidden_size)
        self.seq_proj = nn.Linear(6, hidden_size)
        self.bin_proj = nn.Linear(4, hidden_size)
        self.pos_emb = nn.Parameter(torch.randn(1, 1 + 32 + 32, hidden_size) * 0.02)
        self.norm = nn.LayerNorm(hidden_size)
        self.flow_head = nn.Linear(hidden_size, len(FORECAST_HORIZONS))
        self.flow_scale = nn.Parameter(torch.ones(len(FORECAST_HORIZONS)) * 1.05)
        self.phys_weight = nn.Parameter(torch.ones(len(FORECAST_HORIZONS)) * 0.70)
        self.ret_head = nn.Linear(hidden_size + len(FORECAST_HORIZONS), len(FORECAST_HORIZONS))
        self.crash_head = nn.Linear(hidden_size + len(FORECAST_HORIZONS), 1)

    def forward(
        self, micro: torch.Tensor, seq: torch.Tensor, bins: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_ctx = self.ctx_proj(micro).unsqueeze(1)
        t_seq = self.seq_proj(seq)
        t_bin = self.bin_proj(bins)
        tokens = torch.cat([t_ctx, t_seq, t_bin], dim=1) + self.pos_emb

        param_dtype = next(self.backbone.parameters()).dtype
        out = self.backbone(inputs_embeds=tokens.to(param_dtype), output_hidden_states=True)
        h_last = out.hidden_states[-1].float()
        pooled = self.norm(0.5 * (h_last[:, 0, :] + h_last[:, -1, :]))

        pred_flow_asinh = self.flow_head(pooled)
        pred_flow_eth = torch.sinh(torch.clamp(pred_flow_asinh, -5.5, 5.5)) * 50.0 * self.flow_scale
        phys_bps = LiquidityCoupledWorldForecaster.integrate_flow_on_book(pred_flow_eth, bins)

        cat_h = torch.cat([pooled, phys_bps / 100.0], dim=-1)
        total_ret_bps = self.phys_weight * phys_bps + self.ret_head(cat_h)
        crash_logit = self.crash_head(cat_h).squeeze(-1)
        return pred_flow_asinh, total_ret_bps, crash_logit


def run_llm_forecaster_ladder(
    train_t: dict[str, torch.Tensor],
    val_t: dict[str, torch.Tensor],
    test_t: dict[str, torch.Tensor],
    test_samples: list[ForecastSample],
    out_path: Path,
) -> dict[str, Any]:
    """Experiment 3: Pretrained Qwen2.5 LoRA/QLoRA Price & Order-Flow Forecasters (0.5B, 1.5B, 3B)."""
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        msg = {"error": f"transformers/peft not installed: {exc}"}
        _atomic_save_json(out_path, msg)
        return msg

    payload: dict[str, Any] = {"runs": {}}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    configs = [
        ("Qwen2.5-0.5B_LoRA", "Qwen/Qwen2.5-0.5B", False, 16),
        ("Qwen2.5-1.5B_QLoRA", "Qwen/Qwen2.5-1.5B", True, 16),
        ("Qwen2.5-3B_QLoRA", "Qwen/Qwen2.5-3B", True, 16),
    ]

    device = train_t["micro"].device
    micro_dim = int(train_t["micro"].shape[1])

    for tag, model_id, use_4bit, lora_r in configs:
        if tag in payload["runs"] and "error" not in payload["runs"][tag]:
            print(f"[LLM Forecaster] Skipping already completed {tag}")
            continue
        t_start = time.time()
        print(f"[LLM Forecaster] Training {tag} ({model_id})...")
        try:
            torch.manual_seed(101)
            np.random.seed(101)
            if use_4bit:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                base = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=bnb_cfg, device_map="auto"
                )
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16, device_map="auto"
                )
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            peft_backbone = get_peft_model(base.model, lora_cfg)
            hidden_size = int(base.config.hidden_size)
            model = QwenPriceOrderFlowForecaster(peft_backbone, micro_dim, hidden_size).to(device)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-3
            )
            epochs = 4
            batch_size = 64
            n_tr = int(train_t["micro"].shape[0])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            huber_flow = nn.HuberLoss(delta=1.0)
            huber_ret = nn.HuberLoss(delta=25.0)
            bce_crash = nn.BCEWithLogitsLoss()

            for _ep in range(epochs):
                model.train()
                perm = torch.randperm(n_tr, device=device)
                for i in range(0, n_tr, batch_size):
                    idx = perm[i : i + batch_size]
                    opt.zero_grad()
                    pf, pr, pc = model(train_t["micro"][idx], train_t["seq"][idx], train_t["bins"][idx])
                    loss_f = huber_flow(pf, train_t["flow_asinh"][idx])
                    loss_r = huber_ret(pr, train_t["ret_bps"][idx]) / 25.0
                    cos_r = -torch.mean(torch.tanh(pr / 15.0) * torch.tanh(train_t["ret_bps"][idx] / 15.0))
                    loss_c = bce_crash(pc, train_t["crash_label"][idx])
                    loss = loss_f + loss_r + 0.35 * cos_r + 0.25 * loss_c
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                sched.step()

            model.eval()
            pf_list, pr_list, pc_list = [], [], []
            n_te = int(test_t["micro"].shape[0])
            with torch.no_grad():
                for i in range(0, n_te, 128):
                    pf, pr, pc = model(
                        test_t["micro"][i : i + 128],
                        test_t["seq"][i : i + 128],
                        test_t["bins"][i : i + 128],
                    )
                    pf_list.append(torch.sinh(torch.clamp(pf, -6.0, 6.0)).cpu().numpy() * 50.0)
                    pr_list.append(pr.cpu().numpy())
                    pc_list.append(torch.sigmoid(pc).cpu().numpy())

            pf_np = np.concatenate(pf_list, axis=0)
            pr_np = np.concatenate(pr_list, axis=0)
            pc_np = np.concatenate(pc_list, axis=0)

            eval_res = evaluate_predictions(test_samples, pf_np, pr_np, pc_np)
            eval_res["model_id"] = model_id
            eval_res["total_params"] = total_params
            eval_res["trainable_params"] = trainable_params
            eval_res["train_seconds"] = round(time.time() - t_start, 2)
            payload["runs"][tag] = eval_res
            _atomic_save_json(out_path, payload)
            print(
                f"  -> {tag} done in {eval_res['train_seconds']}s | H=25 DirAcc="
                f"{eval_res['by_horizon']['h25']['directional_acc_active_pct']:.1f}%, "
                f"NetPnL=${eval_res['trading_backtest_h50']['net_pnl_usd']:,.2f}"
            )
            del model, peft_backbone, base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            payload["runs"][tag] = {"model_id": model_id, "error": str(exc)}
            _atomic_save_json(out_path, payload)
            print(f"  -> {tag} failed: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results/forecast_suite")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    swaps_by_window = json.loads(Path("data/mainnet_swaps.json").read_text(encoding="utf-8"))
    checkpoints = json.loads(Path("data/mainnet_checkpoints.json").read_text(encoding="utf-8"))
    borrowers = _load_borrowers(Path("data/aave_v3_borrowers.json"))

    print("Building vectorized multi-horizon ETH price & order-flow dataset...")
    all_samples = build_forecast_dataset(swaps_by_window, checkpoints, borrowers)

    june = [s for s in all_samples if s.window_name == "calm_june_2024"]
    crash = [s for s in all_samples if s.window_name == "crash_aug5_2024"]
    octo = [s for s in all_samples if s.window_name == "autumn_oct_2024"]

    n_crash = len(crash)
    i_tr = int(0.55 * n_crash)
    i_va = int(0.70 * n_crash)

    train_samples = june + crash[:i_tr]
    val_samples = crash[i_tr:i_va]
    test_samples = crash[i_va:] + octo
    n_crash_test = len(crash[i_va:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Dataset ready in {time.time() - t0:.1f}s | train={len(train_samples):,}, "
        f"val={len(val_samples):,}, test={len(test_samples):,} | device={device}"
    )

    micro_train = np.stack([s.micro_features for s in train_samples], axis=0)
    micro_mean = micro_train.mean(axis=0, keepdims=True)
    micro_std = micro_train.std(axis=0, keepdims=True) + 1e-6

    train_t = _pack_tensors(train_samples, micro_mean, micro_std, device)
    val_t = _pack_tensors(val_samples, micro_mean, micro_std, device)
    test_t = _pack_tensors(test_samples, micro_mean, micro_std, device)

    # Experiment 1: Multi-Horizon Unconditional Order-Flow & ETH Price Forecasting
    n_test = len(test_samples)
    zero_flow = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    zero_ret = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    vol_proxy = np.array([s.micro_features[14] for s in test_samples], dtype=np.float64)

    ewma_flow = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    ewma_ret = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    for i, s in enumerate(test_samples):
        net15_eth = math.sinh(s.micro_features[10]) * 50.0
        ret15_bps = s.micro_features[13] * 100.0
        for h_idx, h in enumerate(FORECAST_HORIZONS):
            scale = (h / 15.0) ** 0.55
            ewma_flow[i, h_idx] = -0.25 * net15_eth * scale
            ewma_ret[i, h_idx] = -0.20 * ret15_bps * scale

    results_models: dict[str, Any] = {
        "last_price_naive": evaluate_predictions(test_samples, zero_flow, zero_ret, vol_proxy),
        "momentum_reversion_ewma": evaluate_predictions(test_samples, ewma_flow, ewma_ret, vol_proxy),
    }

    seeds = (101, 202, 303)
    saved_preds: dict[str, dict[str, np.ndarray]] = {}
    pred_flows_by_model: dict[str, np.ndarray] = {}

    for kind in ("orderflow_mlp", "orderflow_transformer", "liquidity_coupled_wm"):
        t_m = time.time()
        flow_preds_seeds, ret_preds_seeds, crash_preds_seeds = [], [], []
        n_params = 0
        for sd in seeds:
            model = train_forecaster(kind, train_t, val_t, seed=sd, epochs=40)
            n_params = sum(p.numel() for p in model.parameters())
            with torch.no_grad():
                pf_asinh, pr_bps, pc_logit = model(test_t["micro"], test_t["seq"], test_t["bins"])
                flow_preds_seeds.append(torch.sinh(torch.clamp(pf_asinh, -6.0, 6.0)).cpu().numpy() * 50.0)
                ret_preds_seeds.append(pr_bps.cpu().numpy())
                crash_preds_seeds.append(torch.sigmoid(pc_logit).cpu().numpy())

        mean_pf = np.mean(flow_preds_seeds, axis=0)
        mean_pr = np.mean(ret_preds_seeds, axis=0)
        mean_pc = np.mean(crash_preds_seeds, axis=0)
        saved_preds[kind] = {"flow": mean_pf, "ret": mean_pr, "crash": mean_pc}
        pred_flows_by_model[kind] = mean_pf

        eval_res = evaluate_predictions(test_samples, mean_pf, mean_pr, mean_pc)
        eval_res["params"] = n_params
        eval_res["train_seconds"] = round(time.time() - t_m, 2)
        results_models[kind] = eval_res
        bt = eval_res["trading_backtest_h50"]
        print(
            f"[{kind}] ({eval_res['train_seconds']}s) | H=25 DirAcc="
            f"{eval_res['by_horizon']['h25']['directional_acc_active_pct']:.1f}%, "
            f"H=50 NetPnL=${bt['net_pnl_usd']:,.2f} (Gross=${bt['gross_pnl_usd']:,.2f}, WinRate={bt['win_rate_pct']:.1f}%)"
        )

    # Save continuous out-of-sample trajectories across Aug 5 crash + Oct 2024
    h25_idx = FORECAST_HORIZONS.index(25)
    h50_idx = FORECAST_HORIZONS.index(50)
    crash_trajectory = []
    for idx in range(0, n_crash_test, 20):
        s = test_samples[idx]
        p_spot = s.spot_eth_usd
        crash_trajectory.append(
            {
                "step": idx,
                "block_number": s.block_number,
                "spot_eth_usd": round(p_spot, 2),
                "true_future_eth_usd_h25": round(float(s.future_eth_usd[h25_idx]), 2),
                "true_future_eth_usd_h50": round(float(s.future_eth_usd[h50_idx]), 2),
                "pred_wm_eth_usd_h25": round(
                    float(p_spot * math.exp(saved_preds["liquidity_coupled_wm"]["ret"][idx, h25_idx] * 1e-4)), 2
                ),
                "pred_wm_eth_usd_h50": round(
                    float(p_spot * math.exp(saved_preds["liquidity_coupled_wm"]["ret"][idx, h50_idx] * 1e-4)), 2
                ),
                "pred_mlp_eth_usd_h50": round(
                    float(p_spot * math.exp(saved_preds["orderflow_mlp"]["ret"][idx, h50_idx] * 1e-4)), 2
                ),
                "pred_transformer_eth_usd_h50": round(
                    float(p_spot * math.exp(saved_preds["orderflow_transformer"]["ret"][idx, h50_idx] * 1e-4)), 2
                ),
                "true_net_flow_eth_h50": round(float(s.future_net_flow_eth[h50_idx]), 2),
                "pred_wm_net_flow_eth_h50": round(float(saved_preds["liquidity_coupled_wm"]["flow"][idx, h50_idx]), 2),
                "pred_mlp_net_flow_eth_h50": round(float(saved_preds["orderflow_mlp"]["flow"][idx, h50_idx]), 2),
                "wm_crash_prob": round(float(saved_preds["liquidity_coupled_wm"]["crash"][idx]), 4),
            }
        )

    exp1_payload = {
        "dataset_summary": {
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "test_crash_aug5_samples": n_crash_test,
            "test_autumn_oct_samples": len(octo),
            "horizons_swaps": list(FORECAST_HORIZONS),
            "seeds": list(seeds),
        },
        "models": results_models,
        "crash_aug5_out_of_sample_trajectory": crash_trajectory,
    }
    _atomic_save_json(out_dir / "eth_price_forecast_results.json", exp1_payload)
    print(f"Saved Experiment 1 to {out_dir / 'eth_price_forecast_results.json'}")

    # Experiment 2: Flow-Conditioned Price Path Forecasting
    print("Running Experiment 2: Flow-Conditioned ETH Price Path Forecasting...")
    run_flow_conditioned_experiment(
        train_t, val_t, test_t, test_samples, pred_flows_by_model, out_dir / "flow_conditioned_forecast.json"
    )
    print(f"Saved Experiment 2 to {out_dir / 'flow_conditioned_forecast.json'}")

    # Experiment 3: Pretrained Qwen2.5 LoRA/QLoRA Price & Order-Flow Forecasters
    if not args.skip_llm:
        print("Running Experiment 3: Qwen2.5 LoRA/QLoRA Price & Order-Flow Forecaster Ladder...")
        run_llm_forecaster_ladder(
            train_t, val_t, test_t, test_samples, out_dir / "llm_price_forecaster.json"
        )
        print(f"Saved Experiment 3 to {out_dir / 'llm_price_forecaster.json'}")

    print(f"All forecasting suite experiments completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
