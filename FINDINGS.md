# Empirical Findings: Counterfactual Intervention Testing for Financial World Models on Ethereum Mainnet

**Target Problem:** [`physicalreasoning/forward` Issue #5 — *Design an intervention test for world models*](https://github.com/physicalreasoning/forward/issues/5)  
**Pool Under Study:** Ethereum Mainnet Uniswap V3 `USDC/WETH` 0.05% (`0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640`) + Aave V3 WETH-collateral borrower positions, June–October 2024 (`calm_june_2024`, `crash_aug5_2024` August 5, 2024 crash window, and `autumn_oct_2024`; `31,283` real swaps, `176` `TickLens` snapshots)

---

## 1. Executive Summary

Standard observational validation loss ($\mathcal{L}_{\text{obs}}$) evaluates a world model's ability to predict next states along the empirical action distribution $P(a_t \mid s_t)$. In concentrated-liquidity automated market makers (Uniswap V3), over `68%` of routine swaps execute entirely inside a single active tick bin (`0` ticks crossed), where price impact follows a smooth local conservation curve $\Delta(1/\sqrt{P}) = \Delta x / L$. Under stress or large counterfactual order sizes $\text{do}(a_t = a^{\text{cf}})$, however, a swap exhausts the active tick's liquidity and crosses discrete `tickSpacing = 10` boundaries, triggering discontinuous jumps in active liquidity $L \leftarrow L \pm \Delta L_i$.

By pairing observational training on historical Mainnet swaps with an exact `uint256` / `Q64.96` EVM counterfactual oracle across held-out historical pool states, we demonstrate four empirical findings:

1. **Observational Loss Does Not Rank Counterfactual Accuracy:** At checkpoints matched within `5%` of observational validation MSE on our crash-exposed benchmark (`train=8,809`, `val=2,378`, `interventions=936`, 5 seeds), `seq_transformer` (`162.00 bps`, `55.04` std-MSE) and `latent_jepa` (`159.66 bps`, `56.73` std-MSE) incur **`20x` higher counterfactual cliff std-MSE** than the `2,549`-parameter `ConservationIntegralWorldModel` (`41.86 bps`, `2.71` std-MSE; paired block-bootstrap gap `+52.28` `[+31.36, +75.91]`, Holm-Bonferroni $p = 0.0012$). Across all 20 runs, observational validation MSE is **negatively correlated** with counterfactual cliff RMSE (Spearman $\rho = -0.535$, $p = 0.0150$).
2. **Brute-Force Parameter Scaling (`184k` $\to$ `170M`) Plateaus on Liquidity Cliffs:** Scaling `mlp_direct`, `seq_transformer`, and `latent_jepa` across four capacity tiers (`S_150k`, `M_1.5M`, `L_8M`, `XXL_85M`, up to `170,363,907` parameters; `train=13,148`, `val=2,755`, `interventions=1,488`, 3 seeds) lowers observational validation loss to `0.1794–0.2360` (vs `0.5136` for `conservation_wm`), yet counterfactual cliff RMSE remains trapped at **`165.09–207.22 bps`** vs **`50.83 bps`** for the `2,549`-parameter conservation model.
3. **Pretrained `3B` and `7B` QLoRA Foundation Models Hit the Exact Same Structural Barrier:** Fine-tuning `Qwen/Qwen2.5-3B` (`3.09B` parameters, `8.02M` trainable) and `Qwen/Qwen2.5-7B` (`7.07B` parameters, `11.23M` trainable) via 4-bit QLoRA with a 34-token spatial `TickLens` state/action encoder achieves strong observational validation loss (`0.2260` and `0.2167`, beating the `0.6834` copy baseline by $>3\times$) and accurate in-tick execution (`1.73` and `2.06 bps`), yet stalls at **`164.50 bps`** (`48.57` std-MSE) and **`174.95 bps`** (`53.01` std-MSE) on Regime II liquidity cliffs.
4. **Ablation and Coverage Diagnostics Isolate Sequential Reserve Integration:**
   - Ablating the learned per-bin residual network (`no_bin_net`, leaving `4` scalar parameters over the raw `TickLens` profile) increases observational validation MSE (`0.5135 -> 1.0472`) while **reducing** counterfactual cliff error (`50.34 -> 34.00 bps`, `2.09` std-MSE), proving that unstructured residual capacity fits routine observational microstructure noise at the expense of out-of-distribution boundary accuracy.
   - Ablating the `TickLens` spatial profile (`uniform_profile`) while keeping the sequential conservation loop degrades cliff RMSE from `50.34 bps` (`3.48` std-MSE) to `133.92 bps` (`27.07` std-MSE), confirming that both the spatial liquidity profile and the sequential conservation integral are necessary.
   - Augmenting training data with `+15%` (`+2,320`) explicit multi-tick cliff swaps improves all architectures, yet unstructured models still exhibit **`5.3x–5.9x` higher cliff RMSE (`133.27–147.32 bps` vs `24.93 bps`) and `30x–37x` higher std-MSE (`31.58–38.80` vs `1.05`)** because each block presents a distinct 32-bin `(ticks, liquidity_net)` step function that cannot be memorized across blocks.

---

## 2. Experimental Design & Verification Gates

*Transparency note:* The protocol specification in [`docs/DESIGN.md`](docs/DESIGN.md) was authored alongside the initial benchmark implementation to lock the evaluation metrics, regime boundaries, and statistical gates prior to running the full multi-seed experiments and scaling sweeps.

### 2.1 State, Action, and Target Representation ([`src/evm_intervene/state.py`](src/evm_intervene/state.py))

Every model family receives the exact same [`EncodedTransition`](src/evm_intervene/state.py#L52-L69) representation:
- **Spatial Liquidity Profile (`32` log-price bins spanning $\pm 17\%$ around spot $P_t$):** `16` downside bins and `16` upside bins derived from the active `TickLens` bitmap, providing relative bin liquidity $L_b / L_t$, fee-adjusted ETH capacity, base unit capacity, signed bin log-price width $\Delta \ln P_b$, and forced liquidation supply inside each bin.
- **Global & Causal History Scalars ($\mathbb{R}^{12}$):** `6` pool-header scalars ($\ln(P_t/3000)$, $\ln(L_t/10^{19})$, in-tick sell/buy capacities, near-spot liquidation supply, minimum downside relative liquidity) concatenated with a `6`-dimensional causal swap-history summary $H_t$ computed strictly from preceding observational swaps.
- **Action $a_t$ (`action_eth`):** Signed swap size in ETH equivalents ($a_t > 0$ sells WETH for USDC; $a_t < 0$ buys WETH with USDC), plus `5` smooth nonlinear transforms for flat tabular/MLP inputs.
- **Target $y_t \in \mathbb{R}^3$:** Post-swap log-price shift $\Delta \ln P$, log-liquidity shift $\Delta \ln L$, and log execution efficiency $\ln(\text{eff})$. Standardized MSE (`std-MSE`) is computed across all 3 target dimensions standardized by the training split's mean and standard deviation.

### 2.2 Gate G0 — Bit-Exact EVM Simulator Verification (`results/simulator_verification.json`)

Before evaluating any learned model, we verify our pure-Python `Q64.96` / `uint256` Uniswap V3 engine ([`src/evm_intervene/simulator.py`](src/evm_intervene/simulator.py)) against `31,283` real historical `Swap` events emitted on Ethereum Mainnet across the June–October 2024 observation windows:

| Metric | All Swaps (`31,283`) | Single-Tick Swaps (`21,500`) | Multi-Tick Swaps (`9,783`) | Gate G0 Threshold | Status |
| :--- | ---: | ---: | ---: | ---: | :---: |
| Median Price Error (bps) | `4.44e-12` | `4.44e-12` | `4.44e-12` | `< 1e-6` | **PASS** |
| 95th Percentile Price Error (bps) | `2.78e-8` | `0.0` | `8.72e-8` | `< 1e-4` | **PASS** |
| Exact Tick Match Rate | `99.94%` | `100.00%` | `99.81%` | `> 99.5%` | **PASS** |
| Exact Liquidity Match Rate | `95.52%` | `100.00%` | `85.67%` | `> 95.0%` | **PASS** |

*(Residual multi-tick liquidity deviations arise only when intra-window `Mint`/`Burn` events occur between the `2,000`-block `TickLens` snapshot intervals; counterfactual interventions execute directly against exact checkpoint snapshots and therefore have zero snapshot drift.)*

---

## 3. Primary Two-Arm Mainnet Benchmark (`results/mainnet_intervention_results.json`)

![Counterfactual Error by Physical Liquidity Regime](assets/intervention_regimes.svg)

We evaluate four model families across `5` random seeds (`42, 43, 44, 45, 46`):
- `mlp_direct`: Residual MLP with layer normalization and SiLU activations mapping $(s_t, a_t) \mapsto \hat{y}_t$.
- `seq_transformer`: Multi-layer self-attention Transformer over tokenized pool header, 32 spatial tick tokens, and action token.
- `latent_jepa`: Joint-Embedding Predictive Architecture encoding $s_t \mapsto z_t$, predicting $\hat{z}_{t+1} = f_\theta(z_t, a_t)$, and decoding $\hat{y}_t$.
- `conservation_wm` ([`ConservationIntegralWorldModel`](src/evm_intervene/models.py)): Hybrid physical-neural model that predicts effective per-bin liquidity adjustments via a small learned module (`2,549` parameters) and integrates swap flow sequentially across tick boundaries using exact $\Delta(\sqrt{P})$ reserve conservation.

### 3.1 Sensitivity Arm — Crash-Exposed Training (`sensitivity_crash_exposed_training`)

In this arm, the training window (`train=8,809`, `val=2,378`) spans into the August 5, 2024 volatility regime so that observational validation losses overlap across architectures, allowing strict 5%-matched checkpoint selection on `39` held-out checkpoints (`936` counterfactual interventions: `225` in-tick, `711` cliff).

| Model | Best obs val std-MSE | Matched obs val std-MSE | In-tick price RMSE (bps) | Cliff price RMSE (bps) | Cliff std-MSE (matched) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `mlp_direct` | 0.2479 | 0.5361 | 3.50 | 162.07 | 56.40 |
| `seq_transformer` | 0.3153 | 0.6131 | 1.95 | 162.00 | 55.04 |
| `latent_jepa` | 0.4382 | 0.6106 | 6.82 | 159.66 | 56.73 |
| `conservation_wm` | 0.6169 | 0.6169 | 0.22 | 41.86 | 2.71 |
| `copy` (obs val only) | 0.8040 | | | | |

| Challenger vs `conservation_wm` | Within 5% obs loss | Cliff std-MSE gap [95% CI] | Holm $p$ |
| :--- | :---: | :---: | ---: |
| `mlp_direct` | no (15.06%) | +53.66 [+31.69, +78.57] | 0.0012 |
| `seq_transformer` | yes (0.61%) | +52.28 [+31.36, +75.91] | 0.0012 |
| `latent_jepa` | yes (1.03%) | +54.07 [+33.07, +77.73] | 0.0012 |

- **Gate G1 (Matched-Loss Counterfactual Separation):** **PASS.** Both `seq_transformer` (`0.61%` obs loss difference) and `latent_jepa` (`1.03%` obs loss difference) are matched within `5%` of `conservation_wm` (`0.6169`), yet exhibit `+52.28` and `+54.07` higher cliff std-MSE ($p = 0.0012$ after Holm-Bonferroni correction across 10,000 block-bootstrap resamples clustered by checkpoint).
- **Gate G2 (Observational vs. Counterfactual Rank Concordance):** **PASS (Negative Concordance).** Across all $n = 20$ trained models, Spearman rank correlation between observational validation MSE and counterfactual cliff price RMSE is $\rho = -0.535$ ($p = 0.0150$, 95% CI $[-0.802, -0.027]$): models with lower observational error systematically suffer higher counterfactual cliff error.

![Observational Validation Loss vs Counterfactual Cliff Error](assets/obs_vs_int_scatter.svg)

### 3.2 Primary Arm — Chronologically Sealed Pre-Crash Training (`primary_chronological_sealed`)

When trained strictly on calm June 2024 pre-crash blocks (`train=4,309`, `val=1,178`) and evaluated on `54` held-out August 5, 2024 crash and October 2024 checkpoints (`1,296` interventions: `278` in-tick, `978` cliff, `40` cascade), unstructured architectures cannot even match `conservation_wm` on out-of-period validation loss (`6.59–7.61` vs `3.69`), and their counterfactual cliff RMSE reaches `218.48–221.24 bps` (`14,696–15,019` std-MSE) compared to `35.91 bps` (`395.89` std-MSE) for `conservation_wm`.

| Model | Best obs val std-MSE | Matched obs val std-MSE | In-tick price RMSE (bps) | Cliff price RMSE (bps) | Cliff std-MSE (matched) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `mlp_direct` | 6.5914 | 6.5914 | 3.56 | 221.24 | 15,018.82 |
| `seq_transformer` | 7.6062 | 7.6062 | 2.51 | 220.46 | 14,893.93 |
| `latent_jepa` | 6.9216 | 6.9216 | 1.86 | 218.48 | 14,695.66 |
| `conservation_wm` | 3.6949 | 3.6949 | 1.12 | 35.91 | 395.89 |
| `copy` (obs val only) | 9.1744 | | | | |

---

## 4. Scaling Laws & Pretrained Foundation Models (`results/scaling_sweep_results.json`, `results/research_v2/llm.json`)

A natural hypothesis is that `150k`-parameter models fail on multi-tick cliffs simply due to insufficient capacity or lack of pretrained representations. We test both hypotheses on the full `13,148`-swap training split (`val=2,755`, `1,488` counterfactual interventions across `62` held-out checkpoints).

![Parameter Scaling and Pretrained QLoRA Evaluation](assets/scaling_sweep.svg)

### 4.1 Scratch Neural Scaling (`184k` $\to$ `170M` Parameters)

| Tier | Model | Params | Obs val std-MSE | Cliff price RMSE (bps) | Cascade price RMSE (bps) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| anchor | `conservation_wm` | 2,549 | 0.5136 | 50.83 | 2,563.9 |
| S_150k | `mlp_direct` | 247,491 | 0.1945 | 207.22 | 4,070.2 |
| S_150k | `seq_transformer` | 183,747 | 0.2164 | 180.99 | 3,965.5 |
| S_150k | `latent_jepa` | 340,227 | 0.3466 | 191.79 | 3,995.2 |
| M_1.5M | `mlp_direct` | 3,367,043 | 0.1858 | 201.19 | 4,092.0 |
| M_1.5M | `seq_transformer` | 1,795,843 | 0.2135 | 172.06 | 3,957.8 |
| M_1.5M | `latent_jepa` | 3,233,091 | 0.3768 | 181.84 | 3,971.0 |
| L_8M | `mlp_direct` | 19,851,523 | 0.1794 | 200.44 | 4,081.4 |
| L_8M | `seq_transformer` | 9,234,435 | 0.2125 | 170.68 | 3,940.0 |
| L_8M | `latent_jepa` | 16,948,035 | 0.3432 | 180.44 | 3,968.8 |
| XXL_85M | `mlp_direct` | 170,363,907 | 0.1881 | 203.61 | 4,085.1 |
| XXL_85M | `seq_transformer` | 70,426,627 | 0.2360 | 165.09 | 3,942.8 |
| XXL_85M | `latent_jepa` | 126,628,163 | 0.3607 | 180.96 | 3,981.8 |

Across all `39` scratch runs (including the `conservation_wm` anchor), Spearman correlation between observational MSE and cliff RMSE is $\rho = -0.512$ ($p = 0.0009$); across the `36` unstructured neural runs alone, $\rho = -0.380$ ($p = 0.0224$). Increasing parameter count by $69,000\times$ relative to `conservation_wm` improves observational MSE by $2.8\times$ (`0.5136 -> 0.1794`) while leaving counterfactual cliff error more than $3.2\times$ worse (`165.09–207.22 bps` vs `50.83 bps`).

### 4.2 End-to-End QLoRA Fine-Tuning of Pretrained `3B` and `7B` LLMs

To ensure a fair test of pretrained foundation models, [`scripts/run_research_v2.py`](scripts/run_research_v2.py) implements a structured 34-token spatial encoder (`TokenizedStateEncoder`: 1 pool-header token + 32 `TickLens` bin tokens with sinusoidal position embeddings + 1 swap-action token) feeding directly into a pretrained `Qwen2.5` causal Transformer backbone with LoRA adapters (`r=16, lora_alpha=32` on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), trained end-to-end with AdamW for 4 epochs (`batch_size=32`, cosine schedule):

| Backbone | Backbone params | Trainable params | Obs val std-MSE | Beats copy (`0.6834`)? | In-tick RMSE (bps) | Cliff RMSE (bps) | Cliff std-MSE | Cascade RMSE (bps) |
| :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-3B` (4-bit QLoRA) | 3,085,938,688 | 8,020,995 | 0.2260 | yes | 1.73 | 164.50 | 48.57 | 3,892.8 |
| `Qwen/Qwen2.5-7B` (4-bit QLoRA) | 7,070,619,136 | 11,226,115 | 0.2167 | yes | 2.06 | 174.95 | 53.01 | 3,951.2 |

Both `Qwen2.5-3B` and `Qwen2.5-7B` converge cleanly on observational validation loss (`0.2260` and `0.2167`, beating the `0.6834` copy baseline by $>3\times$) and achieve `1.73–2.06 bps` RMSE on Regime I in-tick swaps. Yet on Regime II multi-tick cliffs, they plateau at **`164.50 bps`** and **`174.95 bps`** — virtually identical to the `165.09–180.99 bps` plateau of scratch `seq_transformer` models.

---

## 5. Mechanistic Diagnostics: Ablations & Coverage Sweep

![Conservation Ablations and Training Coverage Sweep](assets/ablation_and_coverage.svg)

### 5.1 Structural Ablation of `ConservationIntegralWorldModel` (`results/research_v2/ablation.json`)

To verify which component of `conservation_wm` drives its counterfactual accuracy, we ablate each sub-module across 3 seeds (`101, 202, 303`):

| Variant | Architectural Change | Obs val std-MSE | In-tick RMSE (bps) | Cliff RMSE (bps) | Cliff std-MSE | Cascade RMSE (bps) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `none` | Full `conservation_wm` (`2,549` params) | 0.5135 | 0.43 | 50.34 | 3.48 | 2,544.8 |
| `no_bin_net` | Zero out learned per-bin residual MLP (`4` scalar params) | 1.0472 | 0.58 | **34.00** | **2.09** | **2,191.7** |
| `uniform_profile` | Erase `TickLens` spatial profile (`log_rel = 0`), keep conservation loop | 0.5531 | 0.45 | 133.92 | 27.07 | 3,780.2 |
| `no_liq_coupling` | Disable liquidity-shift coupling ($\beta_{\text{liq}} = 0$) | 0.5133 | 0.41 | 50.27 | 3.47 | 3,212.1 |

Three mechanistic insights emerge:
1. **Why observational MSE and counterfactual cliff error anti-correlate (`no_bin_net` vs `none`):** In historical observational data, `TickLens` snapshots are sampled every `2,000` blocks (`~7.5 hours`), creating mild intra-window liquidity drift. The learned `bin_net` residual MLP reduces observational validation MSE by `51%` (`1.0472 -> 0.5135`) by absorbing this observational drift, but that learned adjustment introduces a `16.34 bps` bias (`34.00 -> 50.34 bps`) when evaluated on exact counterfactual checkpoints.
2. **Both the spatial profile and sequential integral are required (`uniform_profile`):** Keeping the exact sequential reserve conservation loop while replacing the observed `TickLens` liquidity profile with a uniform profile causes cliff RMSE to jump from `50.34 bps` (`3.48` std-MSE) to `133.92 bps` (`27.07` std-MSE).
3. **Liquidity-shift coupling specifically governs deep cascades (`no_liq_coupling`):** Setting $\beta_{\text{liq}} = 0$ leaves in-tick (`0.41 bps`) and Regime II cliff (`50.27 bps`) price error unchanged, but increases Regime III cascade error from `2,544.8 bps` to `3,212.1 bps`.

### 5.2 Training Coverage Sweep (`results/research_v2/coverage.json`)

Is the failure of unstructured neural models purely a training-support (covariate-shift) problem, or an architectural inability to integrate variable step functions across blocks? We augment the `13,148` observational training swaps with `0% (+0)`, `1% (+133)`, `5% (+692)`, and `15% (+2,320)` simulated multi-tick cliff swaps generated at training-window `TickLens` checkpoints (using random shock multipliers disjoint from the test grid):

| Augmented cliff fraction | Added swaps | `mlp_direct` cliff (bps) | `seq_transformer` cliff (bps) | `conservation_wm` cliff (bps) |
| ---: | ---: | ---: | ---: | ---: |
| 0% | +0 | 201.30 (65.67 std-MSE) | 169.52 (48.58 std-MSE) | 50.34 (3.48 std-MSE) |
| 1% | +133 | 148.57 (37.95 std-MSE) | 161.70 (44.74 std-MSE) | 50.99 (3.56 std-MSE) |
| 5% | +692 | 140.38 (34.27 std-MSE) | 156.06 (42.42 std-MSE) | 41.17 (2.46 std-MSE) |
| 15% | +2,320 | 133.27 (31.58 std-MSE) | 147.32 (38.80 std-MSE) | 24.93 (1.05 std-MSE) |

Adding explicit cliff transitions during training reduces `mlp_direct` cliff error by `33.8%` (`201.30 -> 133.27 bps`) and `seq_transformer` cliff error by `13.1%` (`169.52 -> 147.32 bps`), confirming that observational under-coverage of large shocks accounts for part of the error. However, even at `15%` cliff augmentation (`+2,320` explicit multi-tick swaps), unstructured architectures remain **`5.3x–5.9x` worse in cliff RMSE (`30x–37x` worse in std-MSE)** than `conservation_wm` (`24.93 bps`, `1.05` std-MSE), because executing a multi-tick swap requires sequentially depleting a state-dependent sequence of bins $\sum_{k=1}^{m} L_k \Delta(\sqrt{P}_k)$ that varies across time.

---

## 6. Limitations & Scope Conditions

1. **Single-Step vs. Multi-Step Reflexive Feedback:** The primary benchmark evaluates single-step state transitions $s_{t+1} = T(s_t, \text{do}(a_t))$ against exact EVM execution. In live markets, large price dislocations also trigger endogenous arbitrage and LP `Mint`/`Burn` responses in subsequent blocks.
2. **32-Bin Local `TickLens` Horizon (`Regime III` Cascades):** Because the state representation encodes the `32` nearest initialized ticks around the active tick (`16` per side), extreme Regime III vacuum cascades that cross $\ge 96$ ticks exceed the local 16-bin window on one side, causing all models (including `conservation_wm` at `2,191.7–2,544.8 bps`) to incur elevated error once flow exits the observed bin window.
3. **Domain Specificity:** Concentrated-liquidity AMMs have an exact deterministic invariant determined by on-chain contract bytecode. In domains where the underlying conservation law is only approximate (e.g., limit-order books with hidden liquidity or off-chain latency), hybrid physical-neural world models must learn both the conservation structure and stochastic boundary slippage.
