# ruff: noqa: E501
"""Bundle the repo + cached Mainnet `data/` into self-contained `notebooks/eth_price_orderflow_forecaster_colab.ipynb`."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

INCLUDE_PATHS = (
    "pyproject.toml",
    "src",
    "scripts",
    "data",
)


def build_notebook() -> dict:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=6) as tf:
        for p_str in INCLUDE_PATHS:
            p = Path(p_str)
            if p.exists():
                tf.add(
                    str(p),
                    arcname=str(p),
                    filter=lambda ti: None if "__pycache__" in ti.name else ti,
                )
    b64_payload = base64.b64encode(buf.getvalue()).decode("ascii")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Next-N-Minutes ETH Price, Order-Flow & Flash-Crash Forecaster (Colab Pro GPU Suite)\n",
                "\n",
                "This self-contained notebook unpacks the cached **31,283 real Ethereum Mainnet Uniswap V3 swaps**, **176 `TickLens` liquidity-book snapshots**, and **Aave V3 borrower positions** (`data/mainnet_swaps.json`, `data/mainnet_checkpoints.json`, `data/aave_v3_borrowers.json`) and runs the 4-part **ETH Price & Order-Flow Forecasting Suite**:\n",
                "\n",
                "1. **Experiment 1 — Multi-Horizon Unconditional Order-Flow & ETH Price Forecasting (`H = 10, 25, 50, 100` swaps / `~2.5 to 25 min` ahead):**\n",
                "   - Compares `LastPrice_Naive`, `Momentum_Reversion_EWMA`, `OrderFlowMLPForecaster` (time-series only), `OrderFlowTransformerForecaster` (32-trade attention), and **`LiquidityCoupledWorldForecaster`** (order-flow sequence predictor + vectorized 32-bin `TickLens` liquidity-book price-impact integrator).\n",
                "2. **Experiment 2 — Flow-Conditioned ETH Price Path Forecasting:**\n",
                "   - Evaluates how accurately each architecture translates incoming order flow (both predicted and realized) across the live Uniswap V3 `TickLens` liquidity book into future ETH/USD prices.\n",
                "3. **Experiment 3 — Pretrained `Qwen2.5` (`0.5B`, `1.5B`, `3B`) LoRA/QLoRA Price & Order-Flow Forecasters:**\n",
                "   - Fine-tunes pretrained `Qwen2.5` foundation models end-to-end on tokenized 32-trade order-flow history + 32-bin `TickLens` spatial liquidity tokens.\n",
                "4. **Experiment 4 — Fee-Aware DeFi Trading Strategy (`$100,000` Notional), Smart Execution Timing, & Air-Pocket Crash Early Warning.**\n",
                "\n",
                "**Google Drive Live Checkpointing:** Every experiment saves atomically to `/content/drive/MyDrive/evm_intervene_forecast/` after each model finishes, so no progress is ever lost if your Colab session disconnects."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Mount Google Drive, Unpack Self-Contained Repo + Mainnet Data (`data/*.json`), and Install Dependencies\n",
                "import base64, io, os, subprocess, sys, tarfile, torch\n",
                "from pathlib import Path\n",
                "\n",
                "assert torch.cuda.is_available(), 'Please enable GPU in Colab: Runtime -> Change runtime type -> L4 / A100 / T4 GPU'\n",
                "print(f'[gpu] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)')\n",
                "\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive', force_remount=False)\n",
                "    OUT_DIR = '/content/drive/MyDrive/evm_intervene_forecast'\n",
                "    print(f'[drive] Mounted! Live outputs will be saved to: {OUT_DIR}')\n",
                "except Exception as e:\n",
                "    OUT_DIR = '/content/evm-intervene/results/forecast_suite'\n",
                "    print(f'[drive] Running without Google Drive mount ({e}); saving to: {OUT_DIR}')\n",
                "\n",
                "os.makedirs(OUT_DIR, exist_ok=True)\n",
                "os.environ['OUT_DIR'] = OUT_DIR\n",
                "\n",
                f"REPO_TAR_XZ_B64 = '{b64_payload}'\n",
                "os.makedirs('/content/evm-intervene', exist_ok=True)\n",
                "os.chdir('/content/evm-intervene')\n",
                "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(REPO_TAR_XZ_B64)), mode='r:xz') as tf:\n",
                "    tf.extractall('.')\n",
                "print('[ok] Unpacked code + Mainnet data into /content/evm-intervene:', sorted(os.listdir('data')))\n",
                "\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'accelerate', 'bitsandbytes', 'peft', 'transformers', 'scipy'], check=True)\n",
                "sys.path.insert(0, '/content/evm-intervene/src')\n",
                "print('[ok] Environment ready!')\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Verify Mainnet Data Files & Existing Google Drive Checkpoints\n",
                "import os, glob\n",
                "print('Mainnet data files:', sorted(glob.glob('/content/evm-intervene/data/*.json')))\n",
                "print('Existing checkpoints in Google Drive:', sorted(glob.glob(os.environ['OUT_DIR'] + '/*.json')))\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Run Full ETH Price, Order-Flow & Qwen2.5 LoRA Forecasting Suite (auto-saves to Google Drive)\n",
                "import os, subprocess, sys\n",
                "out_dir = os.environ['OUT_DIR']\n",
                "env = {**os.environ, 'PYTHONPATH': '.:src'}\n",
                "subprocess.run([sys.executable, '-u', 'scripts/run_colab_forecaster_suite.py', '--out-dir', out_dir], env=env, check=True)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Inspect Saved Results & Download JSON Artifacts\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "out_dir = Path(os.environ['OUT_DIR'])\n",
                "for p in sorted(out_dir.glob('*.json')):\n",
                "    print(f'=== {p.name} ({p.stat().st_size:,} bytes) ===')\n",
                "\n",
                "try:\n",
                "    from google.colab import files\n",
                "    for p in sorted(out_dir.glob('*.json')):\n",
                "        files.download(str(p))\n",
                "except Exception as e:\n",
                "    print('Files are safely stored in Google Drive at:', out_dir)\n"
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "L4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build_notebook()
    p1 = Path("notebooks/eth_price_orderflow_forecaster_colab.ipynb")
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Wrote self-contained {p1} ({p1.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
