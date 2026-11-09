"""Bundle the current repository into self-contained notebooks/evm_intervene_colab_scaleup.ipynb."""

from __future__ import annotations

import base64
import io
import json
import pathlib
import tarfile

INCLUDE_PATHS = (
    "pyproject.toml",
    "README.md",
    "FINDINGS.md",
    "src",
    "scripts",
    "tests",
    "data",
    "results",
)


def main() -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=6) as tf:
        for p_str in INCLUDE_PATHS:
            p = pathlib.Path(p_str)
            if p.exists():
                tf.add(
                    str(p),
                    arcname=str(p),
                    filter=lambda ti: None if "__pycache__" in ti.name else ti,
                )
    b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# `evm-intervene`: Heavy Colab Pro Experiment Suite (`rollout`, `planner`, `llm_ladder`)\n",
                    "\n",
                    "**Physical Reasoning Working Group — Answering [`physicalreasoning/forward` Issue #5](https://github.com/physicalreasoning/forward/issues/5)**\n",
                    "\n",
                    "This notebook runs the **Heavy Follow-Up Experiment Suite** on a Google Colab Pro GPU (`L4` or `A100` recommended; also compatible with `T4`):\n",
                    "1. **Experiment 1 — Multi-Step Autoregressive Counterfactual Rollouts (`H = 1, 2, 3, 5, 8, 12` swaps):** Evaluates 8 world-model configurations (`conservation_wm` `none` & `no_bin_net`, `mlp_direct`, `seq_transformer`, `latent_jepa` at `M_1.5M` & `L_8M` scales x 3 seeds) across `248` 12-step stress trajectories (`2,976` sequential transitions).\n",
                    "2. **Experiment 2 — Downstream Risk-Constrained Execution Planner (`B = 25, 50, 100, 200, 400 bps`):** Tests world-model-guided order sizing across `620` execution tasks (`29,760` counterfactual evaluations), measuring budget violation rates (%), excess slippage overshoot (bps), and sizing error vs. the exact EVM oracle.\n",
                    "3. **Experiment 3 — Real Historical Mainnet Forecasting (`forecast.json`):** (A) **50-Swap Open-Loop Real Order-Flow Forecasting** on held-out August 5, 2024 crash and October 2024 Mainnet streams (`H = 1..50` real trades ahead without ground-truth resets), plus (B) **Unconditional 20-Swap Volatility, Excursion & Liquidity Depletion Forecasting** (comparing EWMA persistence, History-Only MLP, and Spatial `TickLens` Forecasters).\n",
                    "4. **Experiment 4 — Full Foundation-Model Scaling Ladder (`0.5B` -> `14B` QLoRA + Random-Init Control):** End-to-end tokenized spatial `TickLens` fine-tuning across `Qwen2.5-0.5B` (pretrained & random-init scratch control), `1.5B`, `3B`, `7B (r=16 & r=64)`, and `14B` QLoRA, evaluated on single-step regimes, 12-step rollouts, 50-swap real Mainnet forecasting, and execution planning with incremental Google Drive checkpointing after every model.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Mount Google Drive (for live crash-proof checkpointing), Setup GPU & Unpack Repo (~20s)\n",
                    "import base64, io, os, subprocess, sys, tarfile, torch\n",
                    "\n",
                    "assert torch.cuda.is_available(), 'Please enable GPU in Colab: Runtime -> Change runtime type -> L4 / A100 / T4 GPU'\n",
                    "gpu_name = torch.cuda.get_device_name(0)\n",
                    "vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
                    "print(f'[gpu] Detected {gpu_name} ({vram_gb:.1f} GB VRAM)')\n",
                    "\n",
                    "DRIVE_OUT_DIR = '/content/drive/MyDrive/evm_intervene_results/heavy_suite'\n",
                    "try:\n",
                    "    from google.colab import drive\n",
                    "    drive.mount('/content/drive', force_remount=False)\n",
                    "    os.makedirs(DRIVE_OUT_DIR, exist_ok=True)\n",
                    "    os.environ['EVM_INTERVENE_OUT_DIR'] = DRIVE_OUT_DIR\n",
                    "    print(f'[drive] Live checkpointing enabled -> {DRIVE_OUT_DIR}')\n",
                    "    print('[drive] If your session ever disconnects, re-running this notebook will automatically resume from Google Drive!')\n",
                    "except Exception as e:\n",
                    "    print(f'[drive] Google Drive mount skipped ({e}); saving locally to results/heavy_suite')\n",
                    "\n",
                    "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)\n",
                    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'accelerate', 'bitsandbytes', 'peft', 'transformers', 'scipy'], check=True)\n",
                    "\n",
                    f"REPO_TAR_XZ_B64 = '{b64_payload}'\n",
                    "os.makedirs('/content/evm-intervene', exist_ok=True)\n",
                    "os.chdir('/content/evm-intervene')\n",
                    "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(REPO_TAR_XZ_B64)), mode='r:xz') as tf:\n",
                    "    tf.extractall('.')\n",
                    "print('[ok] Unpacked repository into /content/evm-intervene')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Heavy Experiments 1 & 2: 12-Step Autoregressive Rollouts + Risk-Constrained Execution Planner (~5 mins)\n",
                    "# Saves after every model to Google Drive (/content/drive/MyDrive/evm_intervene_results/heavy_suite/{rollout,planner}.json)\n",
                    "import os, subprocess, sys\n",
                    "out_dir = os.environ.get('EVM_INTERVENE_OUT_DIR', 'results/heavy_suite')\n",
                    "env = {**os.environ, 'PYTHONPATH': '.:src', 'EVM_INTERVENE_OUT_DIR': out_dir}\n",
                    "subprocess.run([sys.executable, '-u', 'scripts/run_heavy_suite.py', '--which', 'rollout_and_planner', '--epochs', '45', '--out-dir', out_dir], env=env, check=True)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Heavy Experiment 3: Full LLM Scaling Ladder (0.5B Pretrained, 0.5B Random-Init, 1.5B, 3B, 7B r=16/r=64, 14B QLoRA)\n",
                    "# Incrementally checkpoints to Google Drive (/content/drive/MyDrive/evm_intervene_results/heavy_suite/llm_ladder.json) after every model!\n",
                    "import os, subprocess, sys\n",
                    "out_dir = os.environ.get('EVM_INTERVENE_OUT_DIR', 'results/heavy_suite')\n",
                    "env = {**os.environ, 'PYTHONPATH': '.:src', 'EVM_INTERVENE_OUT_DIR': out_dir}\n",
                    "subprocess.run([sys.executable, '-u', 'scripts/run_heavy_suite.py', '--which', 'llm_ladder', '--llm-epochs', '4', '--out-dir', out_dir], env=env, check=True)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 4. Inspect & Download All Heavy Suite JSON Artifacts (Also Saved in Your Google Drive!)\n",
                    "import json, os\n",
                    "out_dir = os.environ.get('EVM_INTERVENE_OUT_DIR', 'results/heavy_suite')\n",
                    "print(f'Reading artifacts from: {out_dir}')\n",
                    "for name in ['rollout', 'planner', 'forecast', 'llm_ladder']:\n",
                    "    p = os.path.join(out_dir, f'{name}.json')\n",
                    "    if os.path.exists(p):\n",
                    "        print(f'\\n================ {name.upper()} ================')\n",
                    "        with open(p) as f:\n",
                    "            print(f.read())\n",
                    "try:\n",
                    "    from google.colab import files\n",
                    "    for name in ['rollout', 'planner', 'forecast', 'llm_ladder']:\n",
                    "        p = os.path.join(out_dir, f'{name}.json')\n",
                    "        if os.path.exists(p):\n",
                    "            files.download(p)\n",
                    "except Exception as e:\n",
                    "    print('Colab download skipped (files are safely in Google Drive):', e)\n",
                ],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = pathlib.Path("notebooks/evm_intervene_colab_scaleup.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
