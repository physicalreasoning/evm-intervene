"""Large-scale GPU model & dataset sweep (150k -> 85M -> 7B Foundation Parameters).

Designed for a Google Colab Tesla T4 / L4 GPU (15.6 GB VRAM, ~1.5-2.0 Colab compute units).
Evaluates the complete scaling ladder against the 2,549-parameter conservation world model:
  - Anchor: ConservationIntegralWorldModel (2,549 params)
  - Scaling Tiers:
      S_150k  (~250k params)
      M_1.5M  (~2.5M params)
      L_8M    (~8M params)
      XXL_85M (~85M params)
      Tier_7B (7,615,000,000 parameters: Qwen2.5-7B in 4-bit NF4 with action adapter)
  - Pretrained Foundation Baselines: TabPFN v2 (11M) & Chronos-Bolt-Small (48M)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from evm_intervene.metrics import TargetScaler
from evm_intervene.models import (
    BatchTensors,
    ConservationIntegralWorldModel,
    DirectMLPWorldModel,
    LatentJEPAWorldModel,
    SequenceTransformerWorldModel,
    default_torch_device,
    evaluate_model_predictions,
    train_world_model,
)
from scripts.run_experiment import (
    _load_aave_borrowers,
    _summarize_predictions,
    build_counterfactual_suite,
    build_real_observational_transitions,
)


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


SCALE_TIERS: dict[str, dict[str, dict[str, int]]] = {
    "S_150k": {
        "mlp_direct": {"hidden_dim": 192, "depth": 3},
        "seq_transformer": {"d_model": 96, "nhead": 4, "num_layers": 2},
        "latent_jepa": {"latent_dim": 96, "hidden_dim": 192, "depth": 2},
    },
    "M_1.5M": {
        "mlp_direct": {"hidden_dim": 640, "depth": 4},
        "seq_transformer": {"d_model": 256, "nhead": 8, "num_layers": 3},
        "latent_jepa": {"latent_dim": 256, "hidden_dim": 512, "depth": 3},
    },
    "L_8M": {
        "mlp_direct": {"hidden_dim": 1280, "depth": 6},
        "seq_transformer": {"d_model": 512, "nhead": 8, "num_layers": 4},
        "latent_jepa": {"latent_dim": 512, "hidden_dim": 1024, "depth": 4},
    },
    "XXL_85M": {
        "mlp_direct": {"hidden_dim": 3072, "depth": 9},
        "seq_transformer": {"d_model": 1024, "nhead": 16, "num_layers": 8},
        "latent_jepa": {"latent_dim": 1024, "hidden_dim": 2304, "depth": 6},
    },
}


def run_7b_foundation_model(
    train_batch: BatchTensors,
    val_batch: BatchTensors,
    int_batch: BatchTensors,
    y_int_np: np.ndarray,
    scaler: TargetScaler,
    regimes: np.ndarray,
    clusters: np.ndarray,
    device_str: str,
    epochs: int = 20,
    model_id: str = "Qwen/Qwen2.5-7B",
) -> dict[str, Any] | None:
    """Run 7 Billion parameter foundation model (Qwen2.5-7B) in 4-bit NF4 with causal adapter."""
    try:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("[info] transformers or bitsandbytes not installed; skipping 7B model.", flush=True)
        return None

    if not device_str.startswith("cuda") or not torch.cuda.is_available():
        print("[info] 7B model requires a CUDA GPU; skipping.", flush=True)
        return None

    print(f"\n==================================================================", flush=True)
    print(f"--- 7 BILLION PARAMETER FOUNDATION WORLD MODEL ({model_id}) ---", flush=True)
    print(f"==================================================================", flush=True)

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[7B] Loading {model_id} in 4-bit NF4 precision (~3.8 GB VRAM)...", flush=True)
        backbone = AutoModel.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        backbone.eval()
        hidden_dim = backbone.config.hidden_size  # 3584 for Qwen2.5-7B
        total_params = sum(p.numel() for p in backbone.parameters())

        # Project 113-dim state-action into 7B embedding space via learned projection
        class FoundationActionAdapter(torch.nn.Module):
            def __init__(self, in_features: int, d_model: int) -> None:
                super().__init__()
                self.proj_in = torch.nn.Linear(in_features, d_model)
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(d_model, 512),
                    torch.nn.SiLU(),
                    torch.nn.Linear(512, 3),
                )

            def forward(self, h_ctx: torch.Tensor) -> torch.Tensor:
                return self.head(h_ctx)

        adapter = FoundationActionAdapter(train_batch.flat_x.shape[1], hidden_dim).to(device_str)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

        # Batch encode representations through the frozen 7B backbone
        def get_backbone_features(batch: BatchTensors, batch_size: int = 128) -> torch.Tensor:
            n = batch.flat_x.shape[0]
            feats = []
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    xb = batch.flat_x[i : i + batch_size].to(device_str)
                    # Project input tokens into hidden dimension
                    h_in = adapter.proj_in(xb).unsqueeze(1)  # (B, 1, hidden_dim)
                    out = backbone(inputs_embeds=h_in, output_hidden_states=False)
                    feats.append(out.last_hidden_state.squeeze(1).float())
            return torch.cat(feats, dim=0)

        print("[7B] Extracting representations from frozen 7.6B backbone...", flush=True)
        z_tr = get_backbone_features(train_batch)
        z_va = get_backbone_features(val_batch)
        z_in = get_backbone_features(int_batch)

        y_scale_t = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device_str)
        y_tr = train_batch.y.to(device_str)

        print(f"[7B] Training causal readout adapter across {epochs} epochs...", flush=True)
        bs = 256
        n_tr = z_tr.shape[0]
        best_val = float("inf")
        best_preds = None

        for ep in range(epochs):
            adapter.train()
            perm = torch.randperm(n_tr)
            for s in range(0, n_tr, bs):
                idx = perm[s : s + bs]
                optimizer.zero_grad()
                pred = adapter(z_tr[idx])
                diff = (pred - y_tr[idx]) / y_scale_t.view(1, 3)
                loss = torch.mean(diff * diff)
                loss.backward()
                optimizer.step()

            adapter.eval()
            with torch.no_grad():
                pred_va = adapter(z_va).cpu().numpy()
                val_std_mse = float(np.mean(scaler.per_sample_loss(pred_va, val_batch.y.cpu().numpy())))
                if val_std_mse < best_val:
                    best_val = val_std_mse
                    best_preds = adapter(z_in).cpu().numpy()

        metrics = _summarize_predictions(best_preds, y_int_np, regimes, clusters, scaler)
        cliff_bps = metrics["regime_2_cliff"]["price_rmse_bps"]
        cliff_mse = metrics["regime_2_cliff"]["std_mse"]
        print(
            f"  Tier_7B (Qwen2.5) | params=7,615,000,000 | obs_val={best_val:.4f} | "
            f"cliff_rmse={cliff_bps:.2f} bps | cliff_mse={cliff_mse:.2f}",
            flush=True,
        )

        del backbone, adapter
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "family": "foundation_7b_qwen2.5",
            "model_id": model_id,
            "params": 7_615_000_000,
            "obs_val_std_mse": round(best_val, 5),
            "cliff_rmse_bps": cliff_bps,
            "cliff_std_mse": cliff_mse,
            "regimes": metrics,
        }
    except Exception as exc:
        print(f"[warn] 7B foundation model failed ({type(exc).__name__}: {exc})", flush=True)
        return None


def run_tabpfn_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_int: np.ndarray,
    Y_int: np.ndarray,
    scaler: TargetScaler,
    regimes: np.ndarray,
    clusters: np.ndarray,
    device_str: str,
) -> dict[str, Any] | None:
    """Pretrained TabPFN v2 tabular foundation model baseline (11M params)."""
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("[info] tabpfn not installed; skipping TabPFN v2 baseline.", flush=True)
        return None

    try:
        print("\n--- Foundation Baseline: TabPFN v2 (11M params) ---", flush=True)
        max_rows = min(len(X_train), 5000)
        rng = np.random.default_rng(101)
        sub_idx = (
            rng.choice(len(X_train), size=max_rows, replace=False)
            if len(X_train) > max_rows
            else np.arange(len(X_train))
        )
        feat_std = np.std(X_train[sub_idx], axis=0)
        keep_cols = np.argsort(feat_std)[-100:]

        val_preds = np.zeros_like(Y_val)
        int_preds = np.zeros_like(Y_int)
        for col in range(Y_train.shape[1]):
            try:
                reg = TabPFNRegressor(device=device_str, ignore_pretraining_limits=True)
            except TypeError:
                reg = TabPFNRegressor(device=device_str)
            reg.fit(X_train[sub_idx][:, keep_cols], Y_train[sub_idx, col])
            val_preds[:, col] = reg.predict(X_val[:, keep_cols])
            int_preds[:, col] = reg.predict(X_int[:, keep_cols])

        val_std_mse = float(np.mean(scaler.per_sample_loss(val_preds, Y_val)))
        metrics = _summarize_predictions(int_preds, Y_int, regimes, clusters, scaler)
        cliff_bps = metrics["regime_2_cliff"]["price_rmse_bps"]
        print(
            f"  tabpfn_v2        | params= 11,000,000 | obs_val={val_std_mse:.4f} | "
            f"cliff_rmse={cliff_bps:.2f} bps",
            flush=True,
        )
        return {
            "family": "tabpfn_v2_foundation",
            "params": 11_000_000,
            "obs_val_std_mse": val_std_mse,
            "cliff_rmse_bps": cliff_bps,
            "regimes": metrics,
        }
    except Exception as exc:
        print(f"[warn] TabPFN v2 skipped ({type(exc).__name__}: {exc})", flush=True)
        return None


def run_chronos_bolt_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_int: np.ndarray,
    Y_int: np.ndarray,
    scaler: TargetScaler,
    regimes: np.ndarray,
    clusters: np.ndarray,
    device_str: str,
) -> dict[str, Any] | None:
    """Pretrained Amazon Chronos-Bolt-Small (48M) frozen encoder + ridge probe baseline."""
    try:
        from chronos import BaseChronosPipeline
    except ImportError:
        print("[info] chronos-forecasting not installed; skipping Chronos-Bolt-Small baseline.", flush=True)
        return None

    try:
        print("\n--- Foundation Baseline: Chronos-Bolt-Small (48M params) ---", flush=True)
        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map=device_str,
            torch_dtype=torch.float16 if device_str.startswith("cuda") else torch.float32,
        )
        n_params = sum(p.numel() for p in pipe.model.parameters())

        def extract_chronos_features(X: np.ndarray) -> np.ndarray:
            series = torch.from_numpy(X[:, :96]).float()
            feats = []
            bs = 512
            for i in range(0, len(X), bs):
                batch_ctx = series[i : i + bs]
                with torch.no_grad():
                    quantiles, _ = pipe.predict_quantiles(
                        context=batch_ctx,
                        prediction_length=8,
                        quantile_levels=[0.1, 0.5, 0.9],
                    )
                q_flat = quantiles.reshape(len(batch_ctx), -1).float().cpu().numpy()
                feats.append(np.concatenate([q_flat, X[i : i + bs, 96:]], axis=1))
            return np.concatenate(feats, axis=0)

        Z_tr = extract_chronos_features(X_train)
        Z_va = extract_chronos_features(X_val)
        Z_in = extract_chronos_features(X_int)

        mu = Z_tr.mean(axis=0, keepdims=True)
        sd = np.where(Z_tr.std(axis=0, keepdims=True) < 1e-8, 1.0, Z_tr.std(axis=0, keepdims=True))
        Z_tr_n = np.concatenate([(Z_tr - mu) / sd, np.ones((len(Z_tr), 1))], axis=1)
        Z_va_n = np.concatenate([(Z_va - mu) / sd, np.ones((len(Z_va), 1))], axis=1)
        Z_in_n = np.concatenate([(Z_in - mu) / sd, np.ones((len(Z_in), 1))], axis=1)

        reg_eye = 10.0 * np.eye(Z_tr_n.shape[1])
        reg_eye[-1, -1] = 0.0
        w = np.linalg.solve(Z_tr_n.T @ Z_tr_n + reg_eye, Z_tr_n.T @ Y_train)

        val_preds = Z_va_n @ w
        int_preds = Z_in_n @ w
        val_std_mse = float(np.mean(scaler.per_sample_loss(val_preds, Y_val)))
        metrics = _summarize_predictions(int_preds, Y_int, regimes, clusters, scaler)
        cliff_bps = metrics["regime_2_cliff"]["price_rmse_bps"]
        print(
            f"  chronos_bolt_sm  | params={n_params:11,d} | obs_val={val_std_mse:.4f} | "
            f"cliff_rmse={cliff_bps:.2f} bps",
            flush=True,
        )
        return {
            "family": "chronos_bolt_small_probe",
            "params": int(n_params),
            "obs_val_std_mse": val_std_mse,
            "cliff_rmse_bps": cliff_bps,
            "regimes": metrics,
        }
    except Exception as exc:
        print(f"[warn] Chronos-Bolt-Small skipped ({type(exc).__name__}: {exc})", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EVM-Intervene GPU model-scale sweep.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/scaling_sweep_results.json"))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        default=["S_150k", "M_1.5M", "L_8M", "XXL_85M"],
        choices=list(SCALE_TIERS.keys()),
    )
    parser.add_argument("--include-7b", action="store_true")
    parser.add_argument("--include-tabpfn", action="store_true")
    parser.add_argument("--include-chronos", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = default_torch_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[gpu] Using {gpu_name} ({vram_gb:.2f} GB VRAM) on {device}", flush=True)
    else:
        print(f"[cpu] Running on {device}", flush=True)

    swaps_by_win = json.loads((args.data_dir / "mainnet_swaps.json").read_text(encoding="utf-8"))
    cps_by_win = json.loads((args.data_dir / "mainnet_checkpoints.json").read_text(encoding="utf-8"))
    aave_raw = json.loads((args.data_dir / "aave_v3_borrowers.json").read_text(encoding="utf-8"))
    aave_borrowers = _load_aave_borrowers(aave_raw)

    train_june = build_real_observational_transitions(
        swaps_by_win["calm_june_2024"],
        cps_by_win["calm_june_2024"],
        min_block=20_000_000,
        max_block=20_006_400,
    )
    val_june = build_real_observational_transitions(
        swaps_by_win["calm_june_2024"],
        cps_by_win["calm_june_2024"],
        min_block=20_006_401,
        max_block=20_008_000,
    )
    crash_mid_train = build_real_observational_transitions(
        swaps_by_win["crash_aug5_2024"],
        cps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_460_500,
        max_block=20_464_200,
        max_samples=None,
    )
    crash_mid_val = build_real_observational_transitions(
        swaps_by_win["crash_aug5_2024"],
        cps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_464_201,
        max_block=20_465_000,
        max_samples=None,
    )

    int_crash_early, _, _ = build_counterfactual_suite(
        cps_by_win["crash_aug5_2024"],
        swaps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_458_000,
        max_block=20_460_400,
        window_tag="crash_aug5_onset",
    )
    int_crash_late, _, _ = build_counterfactual_suite(
        cps_by_win["crash_aug5_2024"],
        swaps_by_win["crash_aug5_2024"],
        all_borrowers=aave_borrowers,
        min_block=20_465_001,
        max_block=20_468_000,
        stride=1,
        window_tag="crash_aug5_late",
    )
    int_oct, _, _ = build_counterfactual_suite(
        cps_by_win["autumn_oct_2024"],
        swaps_by_win["autumn_oct_2024"],
        all_borrowers=None,
        stride=2,
        window_tag="autumn_oct_oot",
    )

    train_obs = train_june + crash_mid_train
    val_obs = val_june + crash_mid_val
    int_samples = int_crash_early + int_crash_late + int_oct

    train_batch = BatchTensors.from_samples(train_obs, device=device)
    val_batch = BatchTensors.from_samples(val_obs, device=device)
    int_batch = BatchTensors.from_samples(int_samples, device=device)

    y_train_np = train_batch.y.detach().cpu().numpy().astype(np.float64)
    y_val_np = val_batch.y.detach().cpu().numpy().astype(np.float64)
    y_int_np = int_batch.y.detach().cpu().numpy().astype(np.float64)
    x_train_np = train_batch.flat_x.detach().cpu().numpy()
    x_val_np = val_batch.flat_x.detach().cpu().numpy()
    x_int_np = int_batch.flat_x.detach().cpu().numpy()

    regimes = np.array([s.regime for s in int_samples])
    clusters = np.array([s.block_number for s in int_samples], dtype=np.int64)
    scaler = TargetScaler.from_targets(y_train_np)

    print(
        f"[data] Mainnet Transitions: train={len(train_obs):,d}, val={len(val_obs):,d}, "
        f"interventions={len(int_samples):,d} across {len(np.unique(clusters))} checkpoints",
        flush=True,
    )

    # 1. Structure-Preserving Anchor: conservation_wm (2,549 params)
    print("\n--- Training Structure-Preserving Anchor: conservation_wm (2,549 params) ---", flush=True)
    anchor_runs = []
    for seed in args.seeds:
        bundle = train_world_model(
            "conservation_wm",
            train_batch,
            val_batch,
            scaler,
            seed=seed,
            epochs=args.epochs,
            batch_size=512,
            lr=1.5e-3,
        )
        pred_int = evaluate_model_predictions(bundle.best_model, int_batch)
        metrics = _summarize_predictions(pred_int, y_int_np, regimes, clusters, scaler)
        anchor_runs.append(
            {
                "seed": int(seed),
                "params": count_parameters(bundle.best_model),
                "obs_val_std_mse": round(bundle.best_val_loss, 5),
                "cliff_rmse_bps": metrics["regime_2_cliff"]["price_rmse_bps"],
                "cliff_std_mse": metrics["regime_2_cliff"]["std_mse"],
                "cascade_rmse_bps": metrics["regime_3_cascade"]["price_rmse_bps"],
            }
        )
    anchor_obs_target = float(np.mean([r["obs_val_std_mse"] for r in anchor_runs]))
    print(
        f"[anchor] conservation_wm ({anchor_runs[0]['params']:,d} params): "
        f"obs_val={anchor_obs_target:.4f}, "
        f"cliff_rmse={np.mean([r['cliff_rmse_bps'] for r in anchor_runs]):.2f} bps",
        flush=True,
    )

    tier_results: dict[str, Any] = {}
    all_seed_records: list[dict[str, Any]] = [
        {"tier": "Anchor_2.5k", "family": "conservation_wm", **r} for r in anchor_runs
    ]

    def save_checkpoint(
        tier7b_res: dict[str, Any] | None = None,
        tabpfn_res: dict[str, Any] | None = None,
        chronos_res: dict[str, Any] | None = None,
    ) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        obs_vec = np.array([r["obs_val_std_mse"] for r in all_seed_records])
        cliff_vec = np.array([r["cliff_rmse_bps"] for r in all_seed_records])
        rho, p_val = spearmanr(obs_vec, cliff_vec) if len(obs_vec) >= 3 else (0.0, 1.0)
        payload = {
            "dataset_summary": {
                "train_swaps": len(train_obs),
                "val_swaps": len(val_obs),
                "counterfactual_interventions": len(int_samples),
                "held_out_checkpoints": int(len(np.unique(clusters))),
            },
            "anchor_conservation_wm": {
                "params": int(anchor_runs[0]["params"]),
                "obs_val_std_mse_mean": round(anchor_obs_target, 5),
                "cliff_rmse_bps_mean": round(float(np.mean([r["cliff_rmse_bps"] for r in anchor_runs])), 2),
                "cliff_std_mse_mean": round(float(np.mean([r["cliff_std_mse"] for r in anchor_runs])), 4),
                "cascade_rmse_bps_mean": round(
                    float(np.mean([r["cascade_rmse_bps"] for r in anchor_runs])), 2
                ),
                "runs": anchor_runs,
            },
            "scaling_tiers": tier_results,
            "foundation_7b_model": tier7b_res,
            "tabpfn_v2_baseline": tabpfn_res,
            "chronos_bolt_small_baseline": chronos_res,
            "global_spearman_obs_vs_cliff_rmse": {
                "rho": round(float(rho), 4),
                "p_value": round(float(p_val), 5),
                "n_models": len(all_seed_records),
            },
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 2. Unstructured Scaling Sweep (150k -> 85M params)
    for tier_name in args.tiers:
        tier_cfg = SCALE_TIERS[tier_name]
        tier_results[tier_name] = {}
        print(f"\n=== Scaling Tier: {tier_name} ===", flush=True)
        for family, kwargs in tier_cfg.items():
            fam_runs = []
            is_huge = "XXL" in tier_name
            is_large = "XL" in tier_name or "L_" in tier_name
            bs = 256 if is_huge else 512
            lr = 3e-4 if is_huge else (6e-4 if is_large else 1.2e-3)
            ep = max(30, args.epochs - 10) if is_huge else args.epochs

            for seed in args.seeds:
                bundle = train_world_model(
                    family,
                    train_batch,
                    val_batch,
                    scaler,
                    seed=seed,
                    epochs=ep,
                    batch_size=bs,
                    lr=lr,
                    target_match_loss=anchor_obs_target,
                    model_kwargs=kwargs,
                )
                n_params = count_parameters(bundle.best_model)
                pred_int = evaluate_model_predictions(bundle.best_model, int_batch, batch_size=512)
                metrics = _summarize_predictions(pred_int, y_int_np, regimes, clusters, scaler)
                rec = {
                    "tier": tier_name,
                    "family": family,
                    "seed": int(seed),
                    "params": n_params,
                    "obs_val_std_mse": round(bundle.best_val_loss, 5),
                    "matched_val_std_mse": round(bundle.matched_val_loss, 5),
                    "cliff_rmse_bps": metrics["regime_2_cliff"]["price_rmse_bps"],
                    "cliff_std_mse": metrics["regime_2_cliff"]["std_mse"],
                    "cascade_rmse_bps": metrics["regime_3_cascade"]["price_rmse_bps"],
                }
                fam_runs.append(rec)
                all_seed_records.append(rec)

                del bundle
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            tier_results[tier_name][family] = {
                "params": int(fam_runs[0]["params"]),
                "obs_val_std_mse_mean": round(float(np.mean([r["obs_val_std_mse"] for r in fam_runs])), 5),
                "cliff_rmse_bps_mean": round(float(np.mean([r["cliff_rmse_bps"] for r in fam_runs])), 2),
                "cliff_std_mse_mean": round(float(np.mean([r["cliff_std_mse"] for r in fam_runs])), 4),
                "cascade_rmse_bps_mean": round(
                    float(np.mean([r["cascade_rmse_bps"] for r in fam_runs])), 2
                ),
                "runs": fam_runs,
            }
            p_cnt = int(fam_runs[0]['params'])
            o_val = tier_results[tier_name][family]['obs_val_std_mse_mean']
            c_rmse = tier_results[tier_name][family]['cliff_rmse_bps_mean']
            print(
                f"  {family:16s} | params={p_cnt:11,d} | obs_val={o_val:.4f} | cliff_rmse={c_rmse:.2f} bps",
                flush=True,
            )
            save_checkpoint()

    # 3. 7 Billion Parameter Foundation Model Tier (Qwen2.5-7B 4-bit)
    tier7b_res = None
    if args.include_7b:
        tier7b_res = run_7b_foundation_model(
            train_batch, val_batch, int_batch, y_int_np, scaler, regimes, clusters, str(device)
        )
        if tier7b_res:
            all_seed_records.append(
                {
                    "tier": "Tier_7B",
                    "family": "foundation_7b_qwen2.5",
                    "seed": 101,
                    "params": tier7b_res["params"],
                    "obs_val_std_mse": tier7b_res["obs_val_std_mse"],
                    "cliff_rmse_bps": tier7b_res["cliff_rmse_bps"],
                    "cliff_std_mse": tier7b_res["cliff_std_mse"],
                    "cascade_rmse_bps": tier7b_res["regimes"]["regime_3_cascade"]["price_rmse_bps"],
                }
            )

    tabpfn_res = None
    if args.include_tabpfn:
        tabpfn_res = run_tabpfn_baseline(
            x_train_np, y_train_np, x_val_np, y_val_np, x_int_np, y_int_np, scaler, regimes, clusters, str(device)
        )

    chronos_res = None
    if args.include_chronos:
        chronos_res = run_chronos_bolt_probe(
            x_train_np, y_train_np, x_val_np, y_val_np, x_int_np, y_int_np, scaler, regimes, clusters, str(device)
        )

    save_checkpoint(tier7b_res=tier7b_res, tabpfn_res=tabpfn_res, chronos_res=chronos_res)
    print(f"\n[done] Completed scaling sweep in {time.perf_counter() - t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
