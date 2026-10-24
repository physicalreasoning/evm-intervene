# Data Sources & Schemas (`docs/DATA.md`)

All raw data is fetched directly from public, unauthenticated Ethereum Mainnet archive JSON-RPC nodes (`https://rpc.mevblocker.io` primary, `https://eth-pokt.nodies.app` fallback) via `scripts/download_data.py`. Raw files live in `data/` (never committed to Git) and are pinned by SHA-256 checksums in [`results/data_manifest.json`](../results/data_manifest.json).

---

## 1. On-Chain Contracts & Public Domain Status

Ethereum Mainnet blockchain logs and state trie snapshots are public ledger records (CC0 / public domain). No proprietary data, paid API keys, or off-chain vendor feeds are used.

| Protocol | Mainnet Contract | Address | Details |
| :--- | :--- | :--- | :--- |
| **Uniswap v3** | USDC/WETH 0.05% Pool | `0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640` | `token0` = USDC (`6` dec), `token1` = WETH (`18` dec), `fee` = `500` pips (`5 bps`), `tickSpacing` = `10` |
| **Uniswap v3** | `TickLens` | `0xbfd8137f7d1516D3ea5cA83523914859ec47F573` | `getPopulatedTicksInWord(address,int16)` (`0x351fb478`), queried over word radius $\pm 2$ ($1,280$ initialized tick slots $\approx \pm 64\%$ price span) |
| **Aave v3** | Lending Pool | `0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2` | `LiquidationCall` logs (`0xe413a321...`) filtered for WETH collateral (`0xC02aaA39...`) + `getUserAccountData(address)` (`0xbf92857c`) archive state |

---

## 2. Canonical Historical Block Windows (2024)

| Window Key | Block Range | Approximate Period | Market Regime |
| :--- | :--- | :--- | :--- |
| `calm_june_2024` | `20,000,000 .. 20,008,000` | Early June 2024 (~1.1 days) | Calm, tight concentrated-liquidity regime |
| `crash_aug5_2024` | `20,458,000 .. 20,468,000` | August 5, 2024 (~1.4 days) | Global Yen-carry unwind & DeFi liquidation cascade |
| `autumn_oct_2024` | `21,000,000 .. 21,008,000` | Mid October 2024 (~1.1 days) | Strictly out-of-time future evaluation window |

---

## 3. Generated Files in `data/`

1. **`data/mainnet_swaps.json`**
   - Dictionary keyed by window name containing chronologically ordered `Swap` logs (`block_number`, `tx_index`, `log_index`, `amount0_raw`, `amount1_raw`, `sqrt_price_x96`, `liquidity`, `tick`).
2. **`data/mainnet_checkpoints.json`**
   - Dictionary keyed by window name containing `60` evenly spaced archive pool snapshots per window (`180` total), each storing exact `slot0` (`sqrt_price_x96`, `tick`), active `liquidity`, and sorted `TickLens` arrays (`ticks`, `liquidity_net`).
3. **`data/aave_v3_borrowers.json`**
   - Pre-crash archive snapshots (`block 20,458,000`) of near-threshold WETH-collateral borrowers (`1.0 < HF <= 1.35`, collateral $\ge \$1,000$, debt $\ge \$500$) identified from Aave v3 `LiquidationCall` events during the August 5, 2024 crash.
