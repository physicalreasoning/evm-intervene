# Pre-Registered Intervention Design (`evm-intervene`)

**Frozen before model training.** Responds directly to [`physicalreasoning/forward` Issue #5](https://github.com/physicalreasoning/forward/issues/5) (*Design an intervention test for world models*) and the open problem at `physicalreasoning.ai/open-problems/#intervention-task`.

---

## 1. The Question

Do learned financial world models with **matched observational validation loss** predict the same outcome when an unseen intervention pushes the market across a structural boundary—or does low observational prediction error mask catastrophic failure under intervention?

In passive financial time series ([`causal-jepa`](https://github.com/physicalreasoning/causal-jepa); *Merchant et al., 2026*), evaluation is confined to Pearl Rung 1 (observational next-step prediction). Testing Pearl Rung 2/3 interventions in traditional limit-order-book markets ([`forward` Task 3](https://github.com/physicalreasoning/forward)) is confounded by unobserved second-order counterparty reactions (*"nobody reacts to our order"*).

---

## 2. Environment: Why EVM Concentrated-Liquidity + Liquidation State Machines Give Exact Counterfactuals

An atomic transaction on an Ethereum Automated Market Maker (**Uniswap v3 USDC/WETH 0.05% pool**, `0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640`) coupled with same-block atomic liquidations on **Aave v3** (`0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2`) executes **single-threaded and atomically** by EVM consensus:

1. **No mid-swap quote cancellation:** Resting concentrated-liquidity ticks $\mathcal{T}_t = \{(i_k, \Delta L_k)\}$ are frozen inside the atomic swap $a_t = \Delta x_{\text{in}}$.
2. **Deterministic tick-cliff mechanics ($\mathcal{F}_{\text{AMM}}$):** Within tick interval $[p_k, p_{k+1}]$, virtual reserves obey $\Delta(1/\sqrt{P}) = \Delta x_{\text{eff}} / L_k$ and $\Delta\sqrt{P} = \Delta y_{\text{eff}} / L_k$. Exhausting tick $k$ crosses boundary $i_k$, discontinuously updating active liquidity $L \leftarrow L \pm \Delta L_{i_k}$ (often dropping $50\%\text{--}95\%$ outside the active range).
3. **Deterministic endogenous liquidation cascade ($\mathcal{F}_{\text{cas}}$):** If post-swap price $P^{(0)}$ pushes any leveraged position $m \in \{1, \dots, M\}$ below Health Factor $\text{HF}_m(P) = \frac{c_m \cdot P \cdot \text{LT}_m}{d_m} < 1$, atomic flash-loan liquidators seize collateral $\Delta x_{\text{liq}, m}(P)$ and sell it into the pool in the same block, driving $P^{(r+1)} < P^{(r)}$ until fixed-point convergence $\mathcal{U}^{(r)} = \emptyset$.

Thus $s_{t+1} = T(s_t, a_t) = (\mathcal{F}_{\text{cas}} \circ \mathcal{F}_{\text{AMM}})(s_t, a_t)$ is **exact, deterministic, and resettable** at any historical Ethereum archive block.

---

## 3. Fairness Constraint & Model Families

**Observability Equivalence (No Information Bottleneck):** Every learned world model receives the **exact same state representation $s_t$** and action $a_t$:
- Current spot price $P_t$, active liquidity $L_t$, and recent observational swap summary $\mathcal{H}_t$;
- Full spatial concentrated-liquidity depth profile across $B_{\text{tick}} = 32$ relative price-tick bins ($\pm 16\%$ around $P_t$);
- Full spatial collateral liquidation supply profile across $B_{\text{liq}} = 16$ downside price-drop bins ($0\%$ to $-16\%$ below $P_t$).

If a model fails under intervention, it fails due to **inductive bias and observational gradient starvation** (routine swaps rarely cross outer bins), *not* missing input features.

### Baselines (Zero Learned Parameters or Untrained Projections)
1. `copy`: Martingale / zero-impact baseline ($\Delta \log \hat{P} = 0$, $\Delta \log \hat{L} = 0$).
2. `random_encoder_ridge`: Untrained random neural projection of $(s_t, a_t)$ + ridge regression fit on $\mathcal{D}_{\text{train}}$ (the `causal-jepa` control).
3. `local_linear`: Exact first-order Kyle's $\lambda$ tangent $\Delta P = -\frac{2(1-\gamma)P_t^{3/2}}{L_t}\Delta x$ at $s_t$.
4. `single_tick_xyk`: Exact $x \cdot y = k$ curve assuming active liquidity $L_t$ extends infinitely without tick crossings or liquidations.
5. `amm_only_exact`: Exact multi-tick $\mathcal{F}_{\text{AMM}}$ without lending liquidations (isolating Regime III cascades).

### Learned World Models (5 Seeds Each, Trained Strictly on Observational Swaps $\mathcal{D}_{\text{train}}$)
1. `mlp_direct`: Residual MLP mapping concatenated $(s_t, a_t) \to \hat{y}_{t+1}$ under observational MSE.
2. `seq_transformer`: Tokenized cross-attention Transformer attending over spatial tick/liquidation tokens, swap history, and action token $a_t$.
3. `latent_jepa`: Joint-Embedding Predictive Architecture encoding $z_t = E_\theta(s_t)$ with VICReg variance-covariance regularization, action-conditioned latent predictor $\hat{z}_{t+1} = F_\psi(z_t, a_t)$ trained against EMA target $E_{\bar{\theta}}(s_{t+1})$, and readout head $D_\omega(\hat{z}_{t+1})$.
4. `conservation_wm`: Structure-preserving neural integral world model that applies a shared per-bin neural encoder $g_\theta$ across tick/liquidation bins to estimate non-negative bin capacity $\hat{L}_\theta(p_b)$ and liquidation density $\hat{q}_\theta(p_b)$, advancing action $a_t$ via differentiable reserve conservation—trained **solely on the exact same observational swaps** $\mathcal{D}_{\text{train}}$.

---

## 4. Target Vector, Intervention Regimes, and Metrics

Target transition observables:
$$y_{t+1} = \Big(\Delta \log P_{t+1}, \; \log(L_{t+1}/L_t), \; \log\big(1 + |\Delta y_{\text{out}, t+1}| / (P_t |a_t| + \epsilon)\big)\Big)$$
Primary loss $\ell(\hat{y}, y)$ is variance-standardized MSE (standardized using **training-split-only** $\Sigma_{\text{train}}$), alongside **Log-Price RMSE in basis points (`bps`)**:
$$\text{RMSE}_{\text{bps}} = 10^4 \sqrt{\mathbb{E}\big[(\log \hat{P}_{t+1} - \log P_{t+1})^2\big]}.$$

For each held-out archive checkpoint state $s_t \in \mathcal{D}_{\text{test}}$, we reset the simulator and evaluate $K = 24$ counterfactual swap interventions $a \in \mathcal{A}_{\text{int}}(s_t)$ stratified into three physical regimes:
- **Regime I (In-Tick Smooth / Convex):** $|a| \in (0.10, 0.85) \cdot \Delta x_{\text{max}}(s_t)$ (stays inside active tick, zero liquidations).
- **Regime II (Tick-Cliff Crossing):** $|a| > \Delta x_{\text{max}}(s_t)$, crossing initialized tick boundaries where active liquidity $L$ jumps, without triggering liquidations.
- **Regime III (Liquidation Cascade):** Downside sell interventions where $P^{(0)}(a) < \max_m P_{\text{crit}, m}$, triggering endogenous Aave v3 collateral liquidations.

---

## 5. Pre-Registered Gates (Frozen Criteria)

- **Gate G0 (Simulator & Pipeline Validity):**
  1. On real historical Ethereum Mainnet `Swap` logs with archive pre-state, `F_AMM` reproduces true on-chain post-swap `sqrtPriceX96` with median error `< 0.05 bps`.
  2. Every learned model family beats `copy` on held-out observational validation loss $\mathcal{L}_{\text{obs}}$.
- **Gate G1 (Intervention Separation Under Matched Observational Loss):**
  Evaluate checkpoints passing the **5% Observational Match Screen**:
  $$\frac{|\mathcal{L}_{\text{obs}}(f_A) - \mathcal{L}_{\text{obs}}(f_B)|}{\min(\mathcal{L}_{\text{obs}}(f_A), \mathcal{L}_{\text{obs}}(f_B))} \le 0.05.$$
  **Criterion:** At least one matched-loss pair exhibits a 95% cluster-bootstrapped confidence interval (resampling held-out temporal block checkpoints, 5,000 replicates, Holm-corrected across pairwise comparisons) on the Regime II+III intervention error gap $\Delta_{\text{int}}(f_A, f_B) = \mathcal{E}_{\text{int}}(f_A) - \mathcal{E}_{\text{int}}(f_B)$ **excluding zero**, with relative separation $\ge 20\%$.
- **Gate G2 (Rank Correlation Between Validation Loss and Intervention Error):**
  Across all $M = 20$ trained checkpoints ($4 \text{ families} \times 5 \text{ seeds}$) and intermediate matched-loss checkpoints, report Spearman rank correlation $\rho(\mathcal{L}_{\text{obs}}, \mathcal{E}_{\text{int}}^{\text{II+III}})$ with 95% bootstrap CI. Test whether lower observational validation loss fails to guarantee lower intervention error ($\rho < 0.50$).

**Negative-Result Commitment:** If unstructured neural models (`mlp_direct`, `seq_transformer`, `latent_jepa`) generalize across Regime II/III cliffs just as accurately as `conservation_wm`, or if observational loss strongly orders intervention accuracy ($\rho \ge 0.80$), we report that refutation of our own position directly in `FINDINGS.md`.
