"""Canonical Ethereum Mainnet addresses, ABI selectors, event topics, and benchmark windows."""

from __future__ import annotations

# Free archive-capable Ethereum Mainnet JSON-RPC endpoints (verified zero-auth).
DEFAULT_RPC_URLS: tuple[str, ...] = (
    "https://rpc.mevblocker.io",
    "https://eth-pokt.nodies.app",
)

# Uniswap v3 USDC / WETH 0.05% pool (token0 = USDC [6 dec], token1 = WETH [18 dec]).
UNISWAP_V3_USDC_WETH_POOL = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
UNISWAP_V3_TICK_LENS = "0xbfd8137f7d1516D3ea5cA83523914859ec47F573"
USDC_DECIMALS = 6
WETH_DECIMALS = 18
DECIMAL_SCALE = 10 ** (WETH_DECIMALS - USDC_DECIMALS)  # 1e12
POOL_FEE_PIPS = 500  # 0.05% = 500e-6
POOL_FEE_RATE = POOL_FEE_PIPS / 1_000_000.0
POOL_TICK_SPACING = 10
Q96 = 1 << 96

# Uniswap v3 view selectors & event topics.
SELECTOR_SLOT0 = "0x3850c7bd"
SELECTOR_LIQUIDITY = "0x1a686502"
SELECTOR_POPULATED_TICKS = "0x351fb478"  # getPopulatedTicksInWord(address,int16)
TOPIC_UNISWAP_SWAP = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# Aave v3 Pool on Ethereum Mainnet.
AAVE_V3_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
SELECTOR_USER_ACCOUNT_DATA = "0xbf92857c"  # getUserAccountData(address)
TOPIC_AAVE_LIQUIDATION = "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286"

# Aave v3 WETH liquidation parameters (standard mainnet eMode/collateral config).
AAVE_LIQUIDATION_BONUS = 0.05
AAVE_FULL_LIQ_HF_CUTOFF = 0.95

# Canonical historical Mainnet windows spanning calm, crash (Aug 5, 2024), and out-of-time autumn.
HISTORICAL_WINDOWS: dict[str, tuple[int, int]] = {
    "calm_june_2024": (20_000_000, 20_008_000),
    "crash_aug5_2024": (20_458_000, 20_468_000),
    "autumn_oct_2024": (21_000_000, 21_008_000),
}

# Pre-registered matching & bootstrap constants (docs/DESIGN.md).
MATCHED_LOSS_REL_TOL = 0.05
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
