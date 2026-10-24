"""Zero-auth Ethereum Mainnet archive downloader for `evm-intervene`.

Fetches:
1. Historical Uniswap v3 USDC/WETH 0.05% `Swap` logs across 3 canonical 2024 windows;
2. Exact concentrated-liquidity tick maps (`slot0`, `liquidity`, `TickLens`) at 180 checkpoint blocks;
3. Real Aave v3 `LiquidationCall` accounts and archive `getUserAccountData` collateral/debt books.

Outputs deterministic JSON files into `data/` (gitignored) and writes SHA-256 hashes to
`results/data_manifest.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from evm_intervene.constants import (
    AAVE_V3_POOL,
    DEFAULT_RPC_URLS,
    HISTORICAL_WINDOWS,
    POOL_TICK_SPACING,
    SELECTOR_LIQUIDITY,
    SELECTOR_POPULATED_TICKS,
    SELECTOR_SLOT0,
    SELECTOR_USER_ACCOUNT_DATA,
    TOPIC_AAVE_LIQUIDATION,
    TOPIC_UNISWAP_SWAP,
    UNISWAP_V3_TICK_LENS,
    UNISWAP_V3_USDC_WETH_POOL,
    WETH_ADDRESS,
)
from evm_intervene.simulator import eth_usd_from_raw_sqrt_price, raw_sqrt_price_from_x96


def _to_signed(val: int, bits: int) -> int:
    if val >= (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def _rpc_post(payload: Any, rpc_urls: tuple[str, ...] = DEFAULT_RPC_URLS, timeout: float = 20.0) -> Any:
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for url in rpc_urls:
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "evm-intervene/0.1 (physicalreasoning)",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                if isinstance(parsed, dict) and "error" in parsed:
                    raise RuntimeError(f"RPC error from {url}: {parsed['error']}")
                return parsed
            except Exception as exc:
                last_err = exc
                time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"All archive RPC endpoints failed; last error: {last_err}")


def _encode_int16_256(val: int) -> str:
    if val < 0:
        val = (1 << 256) + val
    return f"{val:064x}"


def _encode_address_256(addr: str) -> str:
    clean = addr.lower().removeprefix("0x")
    return clean.rjust(64, "0")


def fetch_swap_logs(start_block: int, end_block: int, chunk_size: int = 2000) -> list[dict[str, Any]]:
    """Fetch and decode Uniswap v3 Swap events in [start_block, end_block]."""
    swaps: list[dict[str, Any]] = []
    cur = start_block
    req_id = 1
    while cur <= end_block:
        to_blk = min(cur + chunk_size - 1, end_block)
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "eth_getLogs",
            "params": [
                {
                    "address": UNISWAP_V3_USDC_WETH_POOL,
                    "fromBlock": hex(cur),
                    "toBlock": hex(to_blk),
                    "topics": [TOPIC_UNISWAP_SWAP],
                }
            ],
        }
        res = _rpc_post(payload)
        for log in res.get("result", []):
            data_hex = log["data"].removeprefix("0x")
            if len(data_hex) < 64 * 5:
                continue
            words = [int(data_hex[i * 64 : (i + 1) * 64], 16) for i in range(5)]
            amount0 = _to_signed(words[0], 256)
            amount1 = _to_signed(words[1], 256)
            sqrt_price_x96 = words[2]
            liquidity = words[3]
            tick = _to_signed(words[4], 256)
            swaps.append(
                {
                    "block_number": int(log["blockNumber"], 16),
                    "tx_index": int(log["transactionIndex"], 16),
                    "log_index": int(log["logIndex"], 16),
                    "amount0_raw": str(amount0),
                    "amount1_raw": str(amount1),
                    "sqrt_price_x96": str(sqrt_price_x96),
                    "liquidity": str(liquidity),
                    "tick": tick,
                }
            )
        cur = to_blk + 1
        req_id += 1
    swaps.sort(key=lambda x: (x["block_number"], x["tx_index"], x["log_index"]))
    return swaps


def _decode_populated_ticks(hex_str: str) -> list[tuple[int, int]]:
    raw = hex_str.removeprefix("0x")
    if len(raw) < 128:
        return []
    offset_bytes = int(raw[0:64], 16)
    len_pos = offset_bytes * 2
    if len(raw) < len_pos + 64:
        return []
    count = int(raw[len_pos : len_pos + 64], 16)
    out: list[tuple[int, int]] = []
    base = len_pos + 64
    for i in range(count):
        w0 = base + i * 192
        if w0 + 128 > len(raw):
            break
        tick_val = _to_signed(int(raw[w0 : w0 + 64], 16), 256)
        liq_net_val = _to_signed(int(raw[w0 + 64 : w0 + 128], 16), 256)
        out.append((tick_val, liq_net_val))
    return out


def fetch_pool_checkpoints(blocks: list[int], word_radius: int = 2) -> list[dict[str, Any]]:
    """Batch-fetch exact `slot0`, `liquidity`, and `TickLens` populated ticks at archive blocks."""
    # Step 1: batch slot0 + liquidity for all blocks (in chunks of 40 blocks = 80 calls).
    slot_liq: dict[int, tuple[int, int, int]] = {}
    batch_chunk = 35
    for start in range(0, len(blocks), batch_chunk):
        sub_blocks = blocks[start : start + batch_chunk]
        batch_req = []
        for idx, b in enumerate(sub_blocks):
            batch_req.append(
                {
                    "jsonrpc": "2.0",
                    "id": idx * 2 + 1,
                    "method": "eth_call",
                    "params": [{"to": UNISWAP_V3_USDC_WETH_POOL, "data": SELECTOR_SLOT0}, hex(b)],
                }
            )
            batch_req.append(
                {
                    "jsonrpc": "2.0",
                    "id": idx * 2 + 2,
                    "method": "eth_call",
                    "params": [{"to": UNISWAP_V3_USDC_WETH_POOL, "data": SELECTOR_LIQUIDITY}, hex(b)],
                }
            )
        responses = {item["id"]: item.get("result", "0x") for item in _rpc_post(batch_req)}
        for idx, b in enumerate(sub_blocks):
            r_slot = responses[idx * 2 + 1].removeprefix("0x")
            r_liq = responses[idx * 2 + 2].removeprefix("0x")
            sqrt_p = int(r_slot[0:64], 16)
            tick = _to_signed(int(r_slot[64:128], 16), 256)
            liq = int(r_liq[0:64], 16)
            slot_liq[b] = (sqrt_p, tick, liq)

    # Step 2: batch TickLens.getPopulatedTicksInWord across [w0 - word_radius, w0 + word_radius].
    pool_arg = _encode_address_256(UNISWAP_V3_USDC_WETH_POOL)
    ticks_by_block: dict[int, dict[int, int]] = {b: {} for b in blocks}

    lens_calls: list[tuple[int, dict[str, Any]]] = []
    call_id = 1
    for b in blocks:
        _, tick, _ = slot_liq[b]
        compressed = tick // POOL_TICK_SPACING
        w0 = compressed >> 8
        for w in range(w0 - word_radius, w0 + word_radius + 1):
            calldata = SELECTOR_POPULATED_TICKS + pool_arg + _encode_int16_256(w)
            lens_calls.append(
                (
                    b,
                    {
                        "jsonrpc": "2.0",
                        "id": call_id,
                        "method": "eth_call",
                        "params": [{"to": UNISWAP_V3_TICK_LENS, "data": calldata}, hex(b)],
                    },
                )
            )
            call_id += 1

    lens_chunk = 45
    for start in range(0, len(lens_calls), lens_chunk):
        sub = lens_calls[start : start + lens_chunk]
        payload = [req for _, req in sub]
        resp_map = {item["id"]: item.get("result", "0x") for item in _rpc_post(payload)}
        for b, req in sub:
            res_hex = resp_map.get(req["id"], "0x")
            for t_idx, l_net in _decode_populated_ticks(res_hex):
                ticks_by_block[b][t_idx] = l_net

    checkpoints: list[dict[str, Any]] = []
    for b in blocks:
        sqrt_p, tick, liq = slot_liq[b]
        sorted_ticks = sorted(ticks_by_block[b].items(), key=lambda kv: kv[0])
        checkpoints.append(
            {
                "block_number": b,
                "sqrt_price_x96": str(sqrt_p),
                "tick": tick,
                "liquidity": str(liq),
                "ticks": [t for t, _ in sorted_ticks],
                "liquidity_net": [str(ln) for _, ln in sorted_ticks],
            }
        )
    return checkpoints


def fetch_aave_crash_borrowers(
    start_block: int = 20_455_000,
    end_block: int = 20_470_000,
    snapshot_block: int = 20_458_000,
    max_users: int = 240,
) -> list[dict[str, Any]]:
    """Fetch real Aave v3 liquidated borrowers from Aug 5, 2024 crash and archive pre-crash state."""
    users: list[str] = []
    seen: set[str] = set()
    weth_topic = "0x" + _encode_address_256(WETH_ADDRESS)

    cur = start_block
    req_id = 1
    while cur <= end_block:
        to_blk = min(cur + 5000 - 1, end_block)
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "eth_getLogs",
            "params": [
                {
                    "address": AAVE_V3_POOL,
                    "fromBlock": hex(cur),
                    "toBlock": hex(to_blk),
                    "topics": [TOPIC_AAVE_LIQUIDATION, weth_topic],
                }
            ],
        }
        res = _rpc_post(payload)
        for log in res.get("result", []):
            topics = log.get("topics", [])
            if len(topics) >= 4:
                user_addr = "0x" + topics[3][-40:].lower()
                if user_addr not in seen:
                    seen.add(user_addr)
                    users.append(user_addr)
        cur = to_blk + 1
        req_id += 1

    users = users[:max_users]

    # Query pre-crash ETH/USD spot at snapshot_block to convert USD base collateral -> WETH units.
    slot_res = _rpc_post(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": UNISWAP_V3_USDC_WETH_POOL, "data": SELECTOR_SLOT0}, hex(snapshot_block)],
        }
    )["result"].removeprefix("0x")
    sqrt_p_snap = int(slot_res[0:64], 16)
    eth_usd_snap = eth_usd_from_raw_sqrt_price(raw_sqrt_price_from_x96(sqrt_p_snap))

    borrowers: list[dict[str, Any]] = []
    chunk = 50
    for start in range(0, len(users), chunk):
        sub_u = users[start : start + chunk]
        batch = []
        for idx, u in enumerate(sub_u):
            calldata = SELECTOR_USER_ACCOUNT_DATA + _encode_address_256(u)
            batch.append(
                {
                    "jsonrpc": "2.0",
                    "id": idx + 1,
                    "method": "eth_call",
                    "params": [{"to": AAVE_V3_POOL, "data": calldata}, hex(snapshot_block)],
                }
            )
        resp = {item["id"]: item.get("result", "0x") for item in _rpc_post(batch)}
        for idx, u in enumerate(sub_u):
            raw = resp.get(idx + 1, "0x").removeprefix("0x")
            if len(raw) < 64 * 6:
                continue
            words = [int(raw[i * 64 : (i + 1) * 64], 16) for i in range(6)]
            coll_usd = words[0] / 1e8
            debt_usd = words[1] / 1e8
            liq_thresh = words[3] / 10_000.0
            hf = words[5] / 1e18
            if coll_usd >= 1_000.0 and debt_usd >= 500.0 and 0.50 <= liq_thresh <= 0.98 and 1.0 < hf <= 1.35:
                coll_eth = coll_usd / eth_usd_snap
                borrowers.append(
                    {
                        "user": u,
                        "snapshot_block": snapshot_block,
                        "snapshot_eth_usd": round(eth_usd_snap, 6),
                        "collateral_eth": round(coll_eth, 6),
                        "debt_usdc": round(debt_usd, 6),
                        "liquidation_threshold": round(liq_thresh, 6),
                        "health_factor_at_snapshot": round(hf, 6),
                    }
                )
    return borrowers


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("results/data_manifest.json"))
    parser.add_argument("--checkpoints-per-window", type=int, default=60)
    args = parser.parse_args()

    t0 = time.perf_counter()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    all_swaps: dict[str, list[dict[str, Any]]] = {}
    all_checkpoints: dict[str, list[dict[str, Any]]] = {}

    for name, (b_start, b_end) in HISTORICAL_WINDOWS.items():
        print(f"[{name}] Fetching Uniswap v3 Swap logs {b_start}..{b_end}...")
        swaps = fetch_swap_logs(b_start, b_end)
        all_swaps[name] = swaps
        print(f"  -> {len(swaps)} swaps fetched.")

        # Evenly spaced archive checkpoint blocks across the window.
        step = max(1, (b_end - b_start) // args.checkpoints_per_window)
        cp_blocks = [b_start + i * step for i in range(args.checkpoints_per_window)]
        print(f"[{name}] Fetching archive slot0 + liquidity + TickLens at {len(cp_blocks)} blocks...")
        cps = fetch_pool_checkpoints(cp_blocks, word_radius=2)
        all_checkpoints[name] = cps
        print(f"  -> {len(cps)} checkpoints fetched.")

    print("[aave_v3] Fetching Aug 5, 2024 crash LiquidationCall accounts & archive positions...")
    borrowers = fetch_aave_crash_borrowers()
    print(f"  -> {len(borrowers)} near-threshold WETH borrower positions fetched.")

    swaps_path = args.data_dir / "mainnet_swaps.json"
    cps_path = args.data_dir / "mainnet_checkpoints.json"
    aave_path = args.data_dir / "aave_v3_borrowers.json"

    swaps_path.write_text(json.dumps(all_swaps, indent=2), encoding="utf-8")
    cps_path.write_text(json.dumps(all_checkpoints, indent=2), encoding="utf-8")
    aave_path.write_text(json.dumps(borrowers, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    manifest = {
        "pool_address": UNISWAP_V3_USDC_WETH_POOL,
        "aave_v3_pool": AAVE_V3_POOL,
        "windows": {k: list(v) for k, v in HISTORICAL_WINDOWS.items()},
        "counts": {
            "swaps_by_window": {k: len(v) for k, v in all_swaps.items()},
            "total_swaps": sum(len(v) for v in all_swaps.values()),
            "checkpoints_by_window": {k: len(v) for k, v in all_checkpoints.items()},
            "aave_near_threshold_borrowers": len(borrowers),
        },
        "sha256": {
            "mainnet_swaps.json": _sha256_file(swaps_path),
            "mainnet_checkpoints.json": _sha256_file(cps_path),
            "aave_v3_borrowers.json": _sha256_file(aave_path),
        },
        "download_wall_seconds": round(elapsed, 2),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {elapsed:.2f}s. Manifest written to {args.manifest}.")


if __name__ == "__main__":
    main()
