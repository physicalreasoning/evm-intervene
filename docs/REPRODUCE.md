# Reproduction Guide (`docs/REPRODUCE.md`)

Every table and figure in [`README.md`](../README.md) and [`FINDINGS.md`](../FINDINGS.md) is reproduced deterministically from public Ethereum Mainnet archive data with **zero API keys**.

---

## 1. Environment Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra models --group dev --no-config --default-index https://pypi.org/simple
```

---

## 2. Offline Unit Tests & Synthetic Positive Control (No Network Needed)

```bash
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run pytest -q
uv run python scripts/run_positive_control.py
```

- Unit tests (`tests/`): **~7 seconds** on CPU.
- Synthetic positive control (`results/positive_control.json`): **~45 seconds** on CPU.

---

## 3. Full Ethereum Mainnet Archive Benchmark

```bash
# 1. Fetch 31,588 Uniswap v3 swaps, 180 TickLens maps, and 185 Aave v3 positions (~33s)
uv run python scripts/download_data.py

# 2. Verify F_AMM physics against 31,283 real Mainnet Swap logs (<2s)
uv run python scripts/verify_simulator.py

# 3. Train 4 architectures x 5 seeds across primary + crash-exposed arms (~4-6 min CPU / ~1 min GPU)
uv run python scripts/run_experiment.py --epochs 45

# 4. Regenerate editorial SVG figures from results/mainnet_intervention_results.json (<1s)
uv run python scripts/make_figures.py
```

Check `results/data_manifest.json` to verify SHA-256 checksums of all downloaded archive artifacts.
