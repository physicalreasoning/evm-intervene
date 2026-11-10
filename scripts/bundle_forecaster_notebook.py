# ruff: noqa: E501
"""Generate `notebooks/eth_price_orderflow_forecaster_colab.ipynb` with Google Drive live persistence."""

from __future__ import annotations

import json
from pathlib import Path


def build_notebook() -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Next-N-Minutes ETH Price, Order-Flow & Flash-Crash Forecaster (Colab Pro GPU Suite)\n",
                "\n",
                "This notebook runs the complete **Multi-Horizon ETH Price & Order-Flow Forecasting Suite** on **31,283 real Ethereum Mainnet Uniswap V3 swaps** (including the **August 5, 2024 ETH crash** from `$2,748 -> $2,092` and October 2024):\n",
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
                "# Cell 1: Mount Google Drive for live atomic checkpointing & clone/update repository\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    OUT_DIR = '/content/drive/MyDrive/evm_intervene_forecast'\n",
                "    print(f'Google Drive mounted! Live outputs will be saved to: {OUT_DIR}')\n",
                "except Exception as e:\n",
                "    OUT_DIR = '/content/evm-intervene/results/forecast_suite'\n",
                "    print(f'Running without Google Drive mount ({e}); saving to: {OUT_DIR}')\n",
                "\n",
                "os.makedirs(OUT_DIR, exist_ok=True)\n",
                "os.environ['OUT_DIR'] = OUT_DIR\n",
                "\n",
                "if not Path('/content/evm-intervene').exists():\n",
                "    !git clone https://github.com/physicalreasoning/evm-intervene.git /content/evm-intervene\n",
                "%cd /content/evm-intervene\n",
                "!git pull origin main\n",
                "!pip install -q -e . transformers peft accelerate bitsandbytes scipy\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Verify GPU & existing Google Drive checkpoints\n",
                "import torch, os, glob\n",
                "print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU:', torch.cuda.get_device_name(0), f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')\n",
                "existing = sorted(glob.glob(os.environ['OUT_DIR'] + '/*.json'))\n",
                "print('Existing checkpoints in Google Drive:', existing)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Run Full ETH Price, Order-Flow & Qwen2.5 LoRA Forecasting Suite (auto-saves to Google Drive)\n",
                "!python -u scripts/run_colab_forecaster_suite.py --out-dir \"$OUT_DIR\"\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Inspect Saved Results & Download JSON Artifacts\n",
                "import json, os, glob\n",
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
            "kernelspec": {"display_name": "Python 3", " language": "python", "name": "python3"},
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
    print(f"Wrote {p1}")


if __name__ == "__main__":
    main()
