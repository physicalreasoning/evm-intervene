"""Train and evaluate multi-horizon ETH price, order-flow, and crash forecasters on Ethereum Mainnet.

Saves complete metrics, trading backtest curves, and out-of-sample predicted vs. actual ETH/USD
price & order-flow trajectories to `results/eth_price_forecast_results.json`.
"""

from __future__ import annotations

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
        "flow_asinh": torch.tensor(flow_asinh, dtype=torch.float32, device=device),
        "ret_bps": torch.tensor(ret_bps, dtype=torch.float32, device=device),
        "crash_label": torch.tensor(crash_label, dtype=torch.float32, device=device),
    }


def train_forecaster(
    model_kind: str,
    train_t: dict[str, torch.Tensor],
    val_t: dict[str, torch.Tensor],
    *,
    seed: int,
    epochs: int = 35,
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
            # Cosine directional alignment penalty on returns and flows
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


def _roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pure numpy Mann-Whitney U ROC-AUC computation."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    pos_ranks = ranks[: len(pos)]
    auc = (np.sum(pos_ranks) - len(pos) * (len(pos) - 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def evaluate_predictions(
    samples: list[ForecastSample],
    pred_flow_eth: np.ndarray,
    pred_ret_bps: np.ndarray,
    pred_crash_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive price, order-flow, crash-warning, and trading backtest metrics."""
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

        # Order-flow metrics
        flow_mae = float(np.mean(np.abs(pf - tf)))
        flow_sign_acc = float(np.mean((pf >= 0) == (tf >= 0)) * 100.0)
        active_flow_mask = np.abs(tf) > 25.0
        flow_sign_active = (
            float(np.mean((pf[active_flow_mask] >= 0) == (tf[active_flow_mask] >= 0)) * 100.0)
            if np.any(active_flow_mask)
            else flow_sign_acc
        )
        rho_flow, _ = spearmanr(pf, tf)
        rho_flow = 0.0 if np.isnan(rho_flow) else float(rho_flow)

        # Price forecast metrics
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
        rho_ret, _ = spearmanr(pr_bps, tr_bps)
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

    # Non-overlapping 25-swap trading strategy & smart execution timing backtest ($100,000 notional)
    step = 25
    h25_idx = FORECAST_HORIZONS.index(25)
    notional_usd = 100_000.0
    pool_fee_bps = 5.0  # Uniswap V3 0.05% tier

    pr_25 = pred_ret_bps[::step, h25_idx]
    tr_25 = true_ret_bps[::step, h25_idx]
    cp_25 = pred_crash_prob[::step]

    # Signal threshold: top 35% strongest forecasts or high crash alert
    sig_thresh = max(3.5, float(np.percentile(np.abs(pr_25), 65)))
    positions = np.zeros_like(pr_25)
    for k in range(len(pr_25)):
        if pr_25[k] >= sig_thresh and cp_25[k] < thresh:
            positions[k] = 1.0
        elif pr_25[k] <= -sig_thresh or cp_25[k] >= np.percentile(cp_25, 82):
            positions[k] = -1.0

    gross_pnl_bps = positions * tr_25
    # Pay 5 bps fee whenever position changes
    pos_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
    fee_bps = pos_changes * pool_fee_bps
    net_pnl_bps = gross_pnl_bps - fee_bps

    gross_pnl_usd = gross_pnl_bps * (notional_usd * 1e-4)
    net_pnl_usd = net_pnl_bps * (notional_usd * 1e-4)
    cum_net_pnl_usd = np.cumsum(net_pnl_usd)

    active_trades = positions != 0
    n_trades = int(np.sum(active_trades))
    win_rate = (
        float(np.mean(gross_pnl_bps[active_trades] > 0) * 100.0) if n_trades > 0 else 0.0
    )
    mean_ret = float(np.mean(net_pnl_bps))
    std_ret = float(np.std(net_pnl_bps)) + 1e-8
    # ~240 25-swap intervals per day during active trading
    sharpe = float((mean_ret / std_ret) * math.sqrt(240.0 * 365.0 / 10.0))

    # Smart Execution Timing Savings: Trader must sell 100 ETH within each 25-swap window;
    # if forecaster predicts price drop (pr_25 < 0), execute immediately at spot (0 slippage vs spot);
    # if forecaster predicts price bounce (pr_25 > 0), wait until end of 25-swap window (capturing tr_25).
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
        "trading_backtest_h25": {
            "notional_usd": notional_usd,
            "n_intervals": len(pr_25),
            "n_active_trades": n_trades,
            "win_rate_pct": round(win_rate, 2),
            "gross_pnl_usd": round(float(np.sum(gross_pnl_usd)), 2),
            "net_pnl_usd": round(float(np.sum(net_pnl_usd)), 2),
            "annualized_sharpe": round(sharpe, 2),
            "execution_timing_mean_savings_bps": round(mean_exec_savings_bps, 2),
            "execution_timing_total_savings_usd": round(total_exec_savings_usd, 2),
            "cum_net_pnl_curve_usd": [round(float(x), 2) for x in cum_net_pnl_usd],
        },
    }


def main() -> None:
    t0 = time.time()
    swaps_by_window = json.loads(Path("data/mainnet_swaps.json").read_text(encoding="utf-8"))
    checkpoints = json.loads(Path("data/mainnet_checkpoints.json").read_text(encoding="utf-8"))
    borrowers = _load_borrowers(Path("data/aave_v3_borrowers.json"))

    print("Building multi-horizon ETH price & order-flow forecasting dataset...")
    all_samples = build_forecast_dataset(swaps_by_window, checkpoints, borrowers)

    june = [s for s in all_samples if s.window_name == "calm_june_2024"]
    crash = [s for s in all_samples if s.window_name == "crash_aug5_2024"]
    octo = [s for s in all_samples if s.window_name == "autumn_oct_2024"]

    # Chronological split: train on June + first 55% of Aug 5 crash; val on next 15%; test on final 30% of crash + Oct
    n_crash = len(crash)
    i_tr = int(0.55 * n_crash)
    i_va = int(0.70 * n_crash)

    train_samples = june + crash[:i_tr]
    val_samples = crash[i_tr:i_va]
    test_samples = crash[i_va:] + octo

    print(
        f"Dataset ready: train={len(train_samples):,}, val={len(val_samples):,}, test={len(test_samples):,} "
        f"(test crash={len(crash[i_va:]):,}, test oct={len(octo):,})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    micro_train = np.stack([s.micro_features for s in train_samples], axis=0)
    micro_mean = micro_train.mean(axis=0, keepdims=True)
    micro_std = micro_train.std(axis=0, keepdims=True) + 1e-6

    train_t = _pack_tensors(train_samples, micro_mean, micro_std, device)
    val_t = _pack_tensors(val_samples, micro_mean, micro_std, device)
    test_t = _pack_tensors(test_samples, micro_mean, micro_std, device)

    # 1. Baseline A: Naive Random Walk / Zero-Change
    n_test = len(test_samples)
    zero_flow = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    zero_ret = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    vol_proxy = np.array([s.micro_features[14] for s in test_samples], dtype=np.float64)

    # 2. Baseline B: Multi-Scale EWMA Momentum & Order-Flow Extrapolation
    ewma_flow = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    ewma_ret = np.zeros((n_test, len(FORECAST_HORIZONS)), dtype=np.float64)
    for i, s in enumerate(test_samples):
        # Recent 15-swap net signed ETH flow and return
        net15_asinh = s.micro_features[10]
        net15_eth = math.sinh(net15_asinh) * 50.0
        ret15_bps = s.micro_features[13] * 100.0
        for h_idx, h in enumerate(FORECAST_HORIZONS):
            scale = (h / 15.0) ** 0.55
            ewma_flow[i, h_idx] = -0.25 * net15_eth * scale
            ewma_ret[i, h_idx] = -0.20 * ret15_bps * scale

    results_models: dict[str, Any] = {
        "last_price_naive": evaluate_predictions(test_samples, zero_flow, zero_ret, vol_proxy),
        "momentum_reversion_ewma": evaluate_predictions(test_samples, ewma_flow, ewma_ret, vol_proxy),
    }

    # 3. Train Neural & Physics-Coupled Forecasters across 3 seeds
    seeds = (101, 202, 303)
    saved_preds: dict[str, dict[str, np.ndarray]] = {}

    for kind in ("orderflow_mlp", "orderflow_transformer", "liquidity_coupled_wm"):
        t_m = time.time()
        flow_preds_seeds = []
        ret_preds_seeds = []
        crash_preds_seeds = []
        n_params = 0

        for sd in seeds:
            model = train_forecaster(kind, train_t, val_t, seed=sd, epochs=35)
            n_params = sum(p.numel() for p in model.parameters())
            with torch.no_grad():
                pf_asinh, pr_bps, pc_logit = model(test_t["micro"], test_t["seq"], test_t["bins"])
                pf_eth = torch.sinh(torch.clamp(pf_asinh, -6.0, 6.0)).cpu().numpy() * 50.0
                pr_np = pr_bps.cpu().numpy()
                pc_prob = torch.sigmoid(pc_logit).cpu().numpy()
                flow_preds_seeds.append(pf_eth)
                ret_preds_seeds.append(pr_np)
                crash_preds_seeds.append(pc_prob)

        mean_pf = np.mean(flow_preds_seeds, axis=0)
        mean_pr = np.mean(ret_preds_seeds, axis=0)
        mean_pc = np.mean(crash_preds_seeds, axis=0)
        saved_preds[kind] = {"flow": mean_pf, "ret": mean_pr, "crash": mean_pc}

        eval_res = evaluate_predictions(test_samples, mean_pf, mean_pr, mean_pc)
        eval_res["params"] = n_params
        eval_res["train_seconds"] = round(time.time() - t_m, 2)
        results_models[kind] = eval_res
        h25 = eval_res["by_horizon"]["h25"]
        bt = eval_res["trading_backtest_h25"]
        print(
            f"[{kind}] params={n_params:,} | H=25 USD MAE=${h25['price_usd_mae']:.2f}, "
            f"DirAcc(active)={h25['directional_acc_active_pct']:.1f}%, "
            f"FlowAcc={h25['orderflow_sign_active_pct']:.1f}%, "
            f"NetPnL=${bt['net_pnl_usd']:,.2f} (Sharpe={bt['annualized_sharpe']:.2f})"
        )

    # Extract continuous out-of-sample trajectories for plotting (stride=15 across the Aug 5 crash test slice)
    n_crash_test = len(crash[i_va:])
    stride = 20
    traj_indices = list(range(0, n_crash_test, stride))
    h25_idx = FORECAST_HORIZONS.index(25)
    h50_idx = FORECAST_HORIZONS.index(50)

    crash_trajectory = []
    for idx in traj_indices:
        s = test_samples[idx]
        p_spot = s.spot_eth_usd
        true_p25 = float(s.future_eth_usd[h25_idx])
        true_p50 = float(s.future_eth_usd[h50_idx])
        true_f50 = float(s.future_net_flow_eth[h50_idx])

        mlp_p50 = float(p_spot * math.exp(saved_preds["orderflow_mlp"]["ret"][idx, h50_idx] * 1e-4))
        tf_p50 = float(p_spot * math.exp(saved_preds["orderflow_transformer"]["ret"][idx, h50_idx] * 1e-4))
        wm_p25 = float(p_spot * math.exp(saved_preds["liquidity_coupled_wm"]["ret"][idx, h25_idx] * 1e-4))
        wm_p50 = float(p_spot * math.exp(saved_preds["liquidity_coupled_wm"]["ret"][idx, h50_idx] * 1e-4))
        wm_f50 = float(saved_preds["liquidity_coupled_wm"]["flow"][idx, h50_idx])
        mlp_f50 = float(saved_preds["orderflow_mlp"]["flow"][idx, h50_idx])
        wm_crash_prob = float(saved_preds["liquidity_coupled_wm"]["crash"][idx])

        crash_trajectory.append(
            {
                "step": idx,
                "block_number": s.block_number,
                "spot_eth_usd": round(p_spot, 2),
                "true_future_eth_usd_h25": round(true_p25, 2),
                "true_future_eth_usd_h50": round(true_p50, 2),
                "pred_wm_eth_usd_h25": round(wm_p25, 2),
                "pred_wm_eth_usd_h50": round(wm_p50, 2),
                "pred_mlp_eth_usd_h50": round(mlp_p50, 2),
                "pred_transformer_eth_usd_h50": round(tf_p50, 2),
                "true_net_flow_eth_h50": round(true_f50, 2),
                "pred_wm_net_flow_eth_h50": round(wm_f50, 2),
                "pred_mlp_net_flow_eth_h50": round(mlp_f50, 2),
                "wm_crash_prob": round(wm_crash_prob, 4),
            }
        )

    out_payload = {
        "dataset_summary": {
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "test_crash_aug5_samples": n_crash_test,
            "test_autumn_oct_samples": len(octo),
            "horizons_swaps": list(FORECAST_HORIZONS),
            "seeds": list(seeds),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "models": results_models,
        "crash_aug5_out_of_sample_trajectory": crash_trajectory,
    }

    out_path = Path("results/eth_price_forecast_results.json")
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"Saved complete forecasting results to {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
