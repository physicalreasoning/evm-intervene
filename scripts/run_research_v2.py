"""Follow-up experiments for evm-intervene (Colab GPU).

Three experiments, each writing its own JSON under results/research_v2/:

  ablation  Which part of conservation_wm carries the cliff accuracy?
            Variants: none, no_bin_net, uniform_profile, no_liq_coupling.

  coverage  Is the cliff failure about model class or about training coverage?
            Mix a fraction f of simulated large (cliff-crossing) trades, drawn at
            TRAINING-window checkpoints with action sizes different from the test grid,
            into the observational training set, and re-measure held-out cliff error.

  llm       Pretrained-LLM world models done properly: per-bin tokens projected into the
            LLM embedding space, LoRA adapters (4-bit QLoRA for >=3B), trainable input
            projection and readout head, trained end to end. Includes a randomly
            initialised (no pretraining) control. A run only counts as valid if its
            observational validation loss beats the `copy` baseline.

All experiments share the same splits as scripts/run_scaling_sweep.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import scripts.run_experiment as rx
from evm_intervene.baselines import predict_copy
from evm_intervene.metrics import TargetScaler
from evm_intervene.models import (
    BatchTensors,
    default_torch_device,
    evaluate_model_predictions,
    train_world_model,
)
from evm_intervene.state import NUM_SIDE_BINS

OUT_DIR = Path("results/research_v2")
CLIFF_REGIMES = ("regime_2_cliff", "regime_3_cascade")

M_TIER = {
    "mlp_direct": {"hidden_dim": 640, "depth": 4},
    "seq_transformer": {"d_model": 256, "nhead": 8, "num_layers": 3},
}


# --------------------------------------------------------------------------------------
# Shared data
# --------------------------------------------------------------------------------------
def load_splits(data_dir: Path) -> dict[str, Any]:
    swaps = json.loads((data_dir / "mainnet_swaps.json").read_text(encoding="utf-8"))
    cps = json.loads((data_dir / "mainnet_checkpoints.json").read_text(encoding="utf-8"))
    borrowers = rx._load_aave_borrowers(json.loads((data_dir / "aave_v3_borrowers.json").read_text(encoding="utf-8")))

    obs = rx.build_real_observational_transitions
    train = obs(swaps["calm_june_2024"], cps["calm_june_2024"], min_block=20_000_000, max_block=20_006_400)
    train += obs(
        swaps["crash_aug5_2024"], cps["crash_aug5_2024"], all_borrowers=borrowers,
        min_block=20_460_500, max_block=20_464_200,
    )
    val = obs(swaps["calm_june_2024"], cps["calm_june_2024"], min_block=20_006_401, max_block=20_008_000)
    val += obs(
        swaps["crash_aug5_2024"], cps["crash_aug5_2024"], all_borrowers=borrowers,
        min_block=20_464_201, max_block=20_465_000,
    )

    cf = rx.build_counterfactual_suite
    test = cf(cps["crash_aug5_2024"], swaps["crash_aug5_2024"], all_borrowers=borrowers,
              min_block=20_458_000, max_block=20_460_400, window_tag="crash_onset")[0]
    test += cf(cps["crash_aug5_2024"], swaps["crash_aug5_2024"], all_borrowers=borrowers,
               min_block=20_465_001, max_block=20_468_000, window_tag="crash_late")[0]
    test += cf(cps["autumn_oct_2024"], swaps["autumn_oct_2024"], stride=2, window_tag="oct")[0]

    return {"swaps": swaps, "cps": cps, "borrowers": borrowers, "train": train, "val": val, "test": test}


def build_augmentation_pool(splits: dict[str, Any], seed: int, rounds: int = 6, per_round: int = 12) -> list:
    """Simulated large trades at TRAINING-window checkpoints, with random (non-test-grid) sizes."""
    rng = np.random.default_rng(seed)
    saved = (rx.COUNTERFACTUAL_MULTIPLIERS, rx.FIXED_ETH_SHOCKS)
    pool: list = []
    try:
        for _ in range(rounds):
            mags = np.exp(rng.uniform(math.log(1.2), math.log(10.0), size=per_round))
            signs = rng.choice([-1.0, 1.0], size=per_round)
            rx.COUNTERFACTUAL_MULTIPLIERS = tuple(float(m * s) for m, s in zip(mags, signs, strict=True))
            rx.FIXED_ETH_SHOCKS = ()
            pool += rx.build_counterfactual_suite(
                splits["cps"]["calm_june_2024"], splits["swaps"]["calm_june_2024"],
                min_block=20_000_000, max_block=20_006_400, window_tag="aug_june",
            )[0]
            pool += rx.build_counterfactual_suite(
                splits["cps"]["crash_aug5_2024"], splits["swaps"]["crash_aug5_2024"],
                all_borrowers=splits["borrowers"], min_block=20_460_500, max_block=20_464_200,
                window_tag="aug_crash",
            )[0]
    finally:
        rx.COUNTERFACTUAL_MULTIPLIERS, rx.FIXED_ETH_SHOCKS = saved
    return [s for s in pool if s.regime in CLIFF_REGIMES]


def summarize(preds: np.ndarray, test: list, scaler: TargetScaler) -> dict[str, Any]:
    y = np.stack([s.target for s in test], axis=0)
    regimes = np.array([s.regime for s in test])
    clusters = np.array([s.block_number for s in test])
    return rx._summarize_predictions(preds, y, regimes, clusters, scaler)


def headline(m: dict[str, Any]) -> dict[str, float | None]:
    def g(k: str) -> float | None:
        return m[k]["price_rmse_bps"] if k in m else None

    return {
        "in_tick_bps": g("regime_1_smooth"),
        "cliff_bps": g("regime_2_cliff"),
        "cascade_bps": g("regime_3_cascade"),
        "cliff_std_mse": m["regime_2_cliff"]["std_mse"] if "regime_2_cliff" in m else None,
    }


def save(name: str, payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Experiment 1: ablation of conservation_wm
# --------------------------------------------------------------------------------------
def exp_ablation(splits: dict[str, Any], device: torch.device, seeds: list[int], epochs: int) -> None:
    scaler = TargetScaler.from_targets(np.stack([s.target for s in splits["train"]]))
    tr = BatchTensors.from_samples(splits["train"], device)
    va = BatchTensors.from_samples(splits["val"], device)
    te = BatchTensors.from_samples(splits["test"], device)
    out: dict[str, Any] = {"seeds": seeds, "epochs": epochs, "variants": {}}
    for variant in ("none", "no_bin_net", "uniform_profile", "no_liq_coupling"):
        runs = []
        for seed in seeds:
            print(f"[ablation] {variant} seed={seed}", flush=True)
            b = train_world_model("conservation_wm", tr, va, scaler, seed=seed, epochs=epochs,
                                  batch_size=512, lr=1.5e-3, model_kwargs={"ablation": variant})
            m = summarize(evaluate_model_predictions(b.best_model, te), splits["test"], scaler)
            runs.append({"seed": seed, "obs_val_std_mse": b.best_val_loss, **headline(m)})
            del b
            free(device)
        out["variants"][variant] = {
            "runs": runs,
            "mean": {k: float(np.mean([r[k] for r in runs if r[k] is not None])) for k in runs[0] if k != "seed"},
        }
        mm = out["variants"][variant]["mean"]
        print(f"  => {variant:16s} obs={mm['obs_val_std_mse']:.4f} cliff={mm['cliff_bps']:.2f} bps "
              f"cascade={mm['cascade_bps']:.1f} bps", flush=True)
        save("ablation", out)


# --------------------------------------------------------------------------------------
# Experiment 2: training-coverage sweep
# --------------------------------------------------------------------------------------
def exp_coverage(splits: dict[str, Any], device: torch.device, seeds: list[int], epochs: int,
                 fractions: list[float]) -> None:
    scaler = TargetScaler.from_targets(np.stack([s.target for s in splits["train"]]))
    va = BatchTensors.from_samples(splits["val"], device)
    te = BatchTensors.from_samples(splits["test"], device)
    print("[coverage] building augmentation pool (simulated large trades at training checkpoints)...", flush=True)
    pool = build_augmentation_pool(splits, seed=7)
    print(f"[coverage] pool size = {len(pool):,} cliff/cascade transitions", flush=True)
    n_obs = len(splits["train"])
    out: dict[str, Any] = {"seeds": seeds, "epochs": epochs, "pool_size": len(pool), "n_obs_train": n_obs, "results": {}}

    for frac in fractions:
        n_add = min(len(pool), int(round(frac * n_obs / max(1e-9, 1.0 - frac))))
        key = f"{frac:.3f}"
        out["results"][key] = {"n_added": n_add, "families": {}}
        for fam in ("conservation_wm", "mlp_direct", "seq_transformer"):
            runs = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(pool), size=n_add, replace=False) if n_add else []
                tr = BatchTensors.from_samples(splits["train"] + [pool[i] for i in idx], device)
                print(f"[coverage] frac={frac:.3f} (+{n_add:,}) {fam} seed={seed}", flush=True)
                kw = M_TIER.get(fam)
                lr = 1.5e-3 if fam == "conservation_wm" else 1.2e-3
                b = train_world_model(fam, tr, va, scaler, seed=seed, epochs=epochs, batch_size=512, lr=lr, model_kwargs=kw)
                m = summarize(evaluate_model_predictions(b.best_model, te), splits["test"], scaler)
                runs.append({"seed": seed, "obs_val_std_mse": b.best_val_loss, **headline(m)})
                del b, tr
                free(device)
            mean = {k: float(np.mean([r[k] for r in runs if r[k] is not None])) for k in runs[0] if k != "seed"}
            out["results"][key]["families"][fam] = {"runs": runs, "mean": mean}
            print(f"  => frac={frac:.3f} {fam:16s} obs={mean['obs_val_std_mse']:.4f} cliff={mean['cliff_bps']:.2f} bps",
                  flush=True)
            save("coverage", out)


# --------------------------------------------------------------------------------------
# Experiment 3: pretrained LLM world models (LoRA / QLoRA)
# --------------------------------------------------------------------------------------
class TokenizedStateEncoder(nn.Module):
    """34 tokens: 1 global token, 32 per-bin tokens, 1 action token (last, used for readout)."""

    def __init__(self, d_model: int, stats: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.stats = stats
        self.bin_proj = nn.Linear(5, d_model)
        self.bin_pos = nn.Parameter(torch.randn(2 * NUM_SIDE_BINS, d_model) * 0.02)
        self.glob_proj = nn.Linear(12, d_model)
        self.act_proj = nn.Linear(6, d_model)

    def forward(self, b: BatchTensors) -> torch.Tensor:
        st = self.stats
        bins = torch.stack([b.rel_liq, b.eth_cap, b.unit_cap, b.log_delta, b.liq_supply], dim=-1)
        bins = (torch.asinh(bins) - st["bin_mu"]) / st["bin_sd"]
        glob = (b.globals_x - st["glob_mu"]) / st["glob_sd"]
        a = b.action
        local_cap = torch.where(a >= 0.0, b.eth_cap[:, 0:1], b.eth_cap[:, NUM_SIDE_BINS : NUM_SIDE_BINS + 1])
        act = torch.cat([a / 50.0, torch.asinh(a), torch.asinh(a / 10.0), torch.sign(a),
                         torch.log1p(a.abs()), torch.asinh(a / (local_cap + 1e-3))], dim=-1)
        tok_bins = self.bin_proj(bins) + self.bin_pos
        return torch.cat([self.glob_proj(glob).unsqueeze(1), tok_bins, self.act_proj(act).unsqueeze(1)], dim=1)


class LLMWorldModel(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int, stats: dict[str, torch.Tensor], y_scale: torch.Tensor,
                 compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.backbone = backbone
        self.enc = TokenizedStateEncoder(d_model, stats)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU(), nn.Linear(256, 3))
        self.register_buffer("y_scale", y_scale)
        self.compute_dtype = compute_dtype

    def forward(self, b: BatchTensors) -> torch.Tensor:
        emb = self.enc(b).to(self.compute_dtype)
        h = self.backbone(inputs_embeds=emb).last_hidden_state[:, -1].float()
        return self.head(h) * self.y_scale


def _input_stats(tr: BatchTensors) -> dict[str, torch.Tensor]:
    bins = torch.asinh(torch.stack([tr.rel_liq, tr.eth_cap, tr.unit_cap, tr.log_delta, tr.liq_supply], dim=-1))
    return {
        "bin_mu": bins.mean(dim=(0, 1)), "bin_sd": bins.std(dim=(0, 1)).clamp_min(1e-4),
        "glob_mu": tr.globals_x.mean(0), "glob_sd": tr.globals_x.std(0).clamp_min(1e-4),
    }


def run_llm(model_id: str, splits: dict[str, Any], device: torch.device, *, seed: int, epochs: int,
            batch_size: int, random_init: bool, load_4bit: bool, lora_r: int) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModel, BitsAndBytesConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    cdtype = torch.bfloat16 if bf16 else (torch.float16 if device.type == "cuda" else torch.float32)

    if random_init:
        cfg = AutoConfig.from_pretrained(model_id)
        backbone = AutoModel.from_config(cfg, torch_dtype=cdtype).to(device)
        load_4bit = False
    elif load_4bit:
        from peft import prepare_model_for_kbit_training

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=cdtype, bnb_4bit_use_double_quant=True)
        backbone = AutoModel.from_pretrained(model_id, quantization_config=bnb, torch_dtype=cdtype,
                                             device_map={"": 0})
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
    else:
        backbone = AutoModel.from_pretrained(model_id, torch_dtype=cdtype).to(device)

    n_backbone = sum(p.numel() for p in backbone.parameters())
    if load_4bit:  # 4-bit packs two weights per byte; report logical parameter count
        n_backbone = sum(p.numel() * (2 if p.dtype == torch.uint8 else 1) for p in backbone.parameters())
    backbone = get_peft_model(backbone, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    d_model = backbone.config.hidden_size

    scaler = TargetScaler.from_targets(np.stack([s.target for s in splits["train"]]))
    tr = BatchTensors.from_samples(splits["train"], device)
    va = BatchTensors.from_samples(splits["val"], device)
    te = BatchTensors.from_samples(splits["test"], device)
    y_scale = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
    model = LLMWorldModel(backbone, d_model, _input_stats(tr), y_scale, cdtype)
    model.enc.to(device)
    model.head.to(device)

    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": 2e-4},
        {"params": list(model.enc.parameters()) + list(model.head.parameters()), "lr": 1e-3},
    ], weight_decay=1e-4)
    n_tr = tr.action.shape[0]
    steps = epochs * math.ceil(n_tr / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-4, 1e-3], total_steps=steps, pct_start=0.1)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def predict(b: BatchTensors) -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, b.action.shape[0], 256):
                idx = torch.arange(i, min(i + 256, b.action.shape[0]), device=device)
                with torch.autocast(device_type=device.type, dtype=cdtype, enabled=device.type == "cuda"):
                    outs.append(model(b.slice(idx)).float().cpu().numpy())
        return np.concatenate(outs).astype(np.float64)

    val_y = va.y.cpu().numpy().astype(np.float64)
    best_val, best_state, curve = float("inf"), None, []
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=gen).to(device)
        run_loss = 0.0
        for k, i in enumerate(range(0, n_tr, batch_size)):
            sub = tr.slice(perm[i : i + batch_size])
            with torch.autocast(device_type=device.type, dtype=cdtype, enabled=device.type == "cuda"):
                pred = model(sub)
            loss = torch.mean(((pred.float() - sub.y) / y_scale) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            if k % 25 == 0:
                pct = step / steps
                bar = "=" * int(pct * 25) + "-" * (25 - int(pct * 25))
                sys.stdout.write(f"\r    [{bar}] epoch {ep + 1}/{epochs} step {step}/{steps} "
                                 f"loss={run_loss / (k + 1):.4f} ({time.time() - t0:.0f}s)")
                sys.stdout.flush()
        v = float(np.mean(scaler.per_sample_loss(predict(va), val_y)))
        curve.append(v)
        print(f"\n    epoch {ep + 1}: obs val std-MSE = {v:.4f}", flush=True)
        if v < best_val:
            best_val = v
            best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    if best_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n])
    m = summarize(predict(te), splits["test"], scaler)
    copy_val = float(np.mean(scaler.per_sample_loss(np.stack([predict_copy(s.action_eth) for s in splits["val"]]), val_y)))
    res = {
        "model_id": model_id, "random_init": random_init, "load_4bit": load_4bit, "seed": seed,
        "backbone_params": int(n_backbone), "trainable_params": int(n_trainable), "lora_r": lora_r,
        "epochs": epochs, "obs_val_std_mse": best_val, "copy_obs_val_std_mse": copy_val,
        "valid_beats_copy": bool(best_val < copy_val), "val_curve": curve,
        "train_seconds": round(time.time() - t0, 1), **headline(m), "regimes": m,
    }
    del model, backbone, opt, tr, va, te
    free(device)
    return res


def exp_llm(splits: dict[str, Any], device: torch.device, model_ids: list[str], seed: int, epochs: int,
            batch_size: int, random_init_ids: list[str], lora_r: int) -> None:
    out_path = OUT_DIR / "llm.json"
    out: dict[str, Any] = json.loads(out_path.read_text()) if out_path.exists() else {"runs": []}
    done = {(r["model_id"], r["random_init"]) for r in out["runs"]}
    jobs = [(m, False) for m in model_ids] + [(m, True) for m in random_init_ids]
    for mid, rand in jobs:
        if (mid, rand) in done:
            print(f"[llm] skip {mid} random_init={rand} (already in results)", flush=True)
            continue
        big = any(t in mid for t in ("3B", "7B", "14B", "32B"))
        print(f"\n[llm] {mid} random_init={rand} 4bit={big and not rand}", flush=True)
        try:
            r = run_llm(mid, splits, device, seed=seed, epochs=epochs, batch_size=batch_size,
                        random_init=rand, load_4bit=big, lora_r=lora_r)
        except Exception as exc:  # keep going; record the failure honestly
            print(f"[llm] FAILED {mid}: {type(exc).__name__}: {exc}", flush=True)
            r = {"model_id": mid, "random_init": rand, "error": f"{type(exc).__name__}: {exc}"}
            free(device)
        out["runs"].append(r)
        save("llm", out)
        if "error" not in r:
            print(f"  => {mid} backbone={r['backbone_params']:,} obs={r['obs_val_std_mse']:.4f} "
                  f"(copy={r['copy_obs_val_std_mse']:.4f}, valid={r['valid_beats_copy']}) "
                  f"cliff={r['cliff_bps']:.2f} bps cascade={r['cascade_bps']} bps", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=["ablation", "coverage", "llm"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.15])
    p.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
                                                    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"])
    p.add_argument("--random-init", nargs="*", default=["Qwen/Qwen2.5-0.5B"])
    p.add_argument("--llm-epochs", type=int, default=4)
    p.add_argument("--llm-batch", type=int, default=64)
    p.add_argument("--lora-r", type=int, default=16)
    args = p.parse_args()

    device = default_torch_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[gpu] {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)", flush=True)
    t0 = time.time()
    splits = load_splits(args.data_dir)
    print(f"[data] train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,} "
          f"({time.time() - t0:.0f}s)", flush=True)

    if args.experiment == "ablation":
        exp_ablation(splits, device, args.seeds, args.epochs)
    elif args.experiment == "coverage":
        exp_coverage(splits, device, args.seeds, args.epochs, args.fractions)
    else:
        exp_llm(splits, device, args.models, args.seeds[0], args.llm_epochs, args.llm_batch,
                args.random_init or [], args.lora_r)
    print(f"[done] {args.experiment} in {time.time() - t0:.0f}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
