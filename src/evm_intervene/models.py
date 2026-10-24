"""Four learned world-model families trained strictly on observational swaps.

1. `mlp_direct`: Residual LayerNorm-SiLU MLP on flattened (s_t, a_t).
2. `seq_transformer`: Spatial-bin + history + action cross-attention Transformer.
3. `latent_jepa`: Action-conditioned Joint-Embedding Predictive Architecture (VICReg + EMA target)
   with detached MLP readout probe (`causal-jepa` protocol).
4. `conservation_wm`: Learned equivariant per-bin capacity network integrated via differentiable
   reserve conservation and learned liquidation feedback.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evm_intervene.metrics import TargetScaler
from evm_intervene.state import (
    GLOBAL_SCALAR_DIM,
    NUM_SIDE_BINS,
    NUM_TOTAL_BINS,
    EncodedTransition,
    flatten_features,
)


def default_torch_device(preferred: str = "auto") -> torch.device:
    """Resolve compute device, defaulting to CUDA GPU whenever available."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class BatchTensors:
    """Vectorized tensor dataset for fast mini-batch training and inference."""

    flat_x: torch.Tensor  # (N, D_flat)
    rel_liq: torch.Tensor  # (N, 32)
    eth_cap: torch.Tensor  # (N, 32)
    unit_cap: torch.Tensor  # (N, 32)
    log_delta: torch.Tensor  # (N, 32)
    liq_supply: torch.Tensor  # (N, 32)
    globals_x: torch.Tensor  # (N, 12)
    action: torch.Tensor  # (N, 1)
    y: torch.Tensor  # (N, 3)

    @classmethod
    def from_samples(
        cls,
        samples: list[EncodedTransition],
        device: torch.device,
    ) -> BatchTensors:
        flat_x = np.stack([flatten_features(s) for s in samples], axis=0)
        rel_liq = np.stack([s.bin_rel_liquidity for s in samples], axis=0)
        eth_cap = np.stack([s.bin_eth_capacity for s in samples], axis=0)
        unit_cap = np.stack([s.bin_base_unit_capacity for s in samples], axis=0)
        log_delta = np.stack([s.bin_log_delta for s in samples], axis=0)
        liq_supply = np.stack([s.bin_liq_eth_supply for s in samples], axis=0)
        globals_x = np.stack([s.global_scalars for s in samples], axis=0)
        action = np.array([[s.action_eth] for s in samples], dtype=np.float32)
        y = np.stack([s.target for s in samples], axis=0)

        return cls(
            flat_x=torch.as_tensor(flat_x, dtype=torch.float32, device=device),
            rel_liq=torch.as_tensor(rel_liq, dtype=torch.float32, device=device),
            eth_cap=torch.as_tensor(eth_cap, dtype=torch.float32, device=device),
            unit_cap=torch.as_tensor(unit_cap, dtype=torch.float32, device=device),
            log_delta=torch.as_tensor(log_delta, dtype=torch.float32, device=device),
            liq_supply=torch.as_tensor(liq_supply, dtype=torch.float32, device=device),
            globals_x=torch.as_tensor(globals_x, dtype=torch.float32, device=device),
            action=torch.as_tensor(action, dtype=torch.float32, device=device),
            y=torch.as_tensor(y, dtype=torch.float32, device=device),
        )

    def slice(self, idx: torch.Tensor) -> BatchTensors:
        return BatchTensors(
            flat_x=self.flat_x[idx],
            rel_liq=self.rel_liq[idx],
            eth_cap=self.eth_cap[idx],
            unit_cap=self.unit_cap[idx],
            log_delta=self.log_delta[idx],
            liq_supply=self.liq_supply[idx],
            globals_x=self.globals_x[idx],
            action=self.action[idx],
            y=self.y[idx],
        )


def _action_features(action: torch.Tensor, local_cap: torch.Tensor) -> torch.Tensor:
    """Rich multi-scale action encoding including local-capacity ratio a / cap_0."""
    abs_a = torch.abs(action)
    sign_a = torch.where(action >= 0.0, 1.0, -1.0)
    local_ratio = action / (local_cap + 1e-3)
    return torch.cat(
        [
            action / 50.0,
            torch.asinh(action),
            torch.asinh(action / 10.0),
            sign_a,
            torch.log1p(abs_a),
            torch.asinh(local_ratio),
        ],
        dim=-1,
    )


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(self.norm(x)))
        h = self.drop(self.fc2(h))
        return x + h


class DirectMLPWorldModel(nn.Module):
    """Unstructured residual MLP mapping full flattened (s_t, a_t) -> y_{t+1}."""

    def __init__(self, in_dim: int, hidden_dim: int = 192, depth: int = 3) -> None:
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(1, in_dim))
        self.register_buffer("x_std", torch.ones(1, in_dim))
        self.register_buffer("y_scale", torch.ones(1, 3))

        self.in_proj = nn.Linear(in_dim + 6, hidden_dim)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden_dim) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def set_normalizers(self, flat_x: torch.Tensor, y_scale: torch.Tensor) -> None:
        self.x_mean.copy_(flat_x.mean(dim=0, keepdim=True))
        self.x_std.copy_(flat_x.std(dim=0, keepdim=True).clamp(min=1e-5))
        self.y_scale.copy_(y_scale.view(1, 3))

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        local_cap = torch.where(
            batch.action >= 0.0,
            batch.eth_cap[:, 0:1],
            batch.eth_cap[:, NUM_SIDE_BINS : NUM_SIDE_BINS + 1],
        )
        act_f = _action_features(batch.action, local_cap)
        x_norm = (batch.flat_x - self.x_mean) / self.x_std
        h = F.silu(self.in_proj(torch.cat([x_norm, act_f], dim=-1)))
        for blk in self.blocks:
            h = blk(h)
        return self.head(h) * self.y_scale


class _CrossAttentionQueryBlock(nn.Module):
    """Pre-LN multi-head cross-attention block where query tokens attend over all spatial tokens."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout_p = dropout

    def forward(self, q_tok: torch.Tensor, kv_tok: torch.Tensor) -> torch.Tensor:
        bsz, q_len, dim = q_tok.shape
        kv_len = kv_tok.shape[1]
        q = self.q_proj(self.norm_q(q_tok)).view(bsz, q_len, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(self.norm_kv(kv_tok)).view(bsz, kv_len, 2, self.nhead, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, dim)
        q_tok = q_tok + self.out_proj(attn)
        q_tok = q_tok + self.ffn(self.norm2(q_tok))
        return q_tok


class SequenceTransformerWorldModel(nn.Module):
    """Tokenized Transformer attending over all 32 spatial bins + history + action query."""

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 3) -> None:
        super().__init__()
        self.register_buffer("g_mean", torch.zeros(1, GLOBAL_SCALAR_DIM))
        self.g_std: torch.Tensor
        self.register_buffer("g_std", torch.ones(1, GLOBAL_SCALAR_DIM))
        self.register_buffer("y_scale", torch.ones(1, 3))

        self.bin_proj = nn.Linear(5, d_model)
        self.spatial_mix = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.pos_emb = nn.Parameter(torch.randn(1, NUM_TOTAL_BINS, d_model) * 0.02)
        self.global_proj = nn.Linear(GLOBAL_SCALAR_DIM, d_model)
        self.action_proj = nn.Linear(6, d_model)

        self.blocks = nn.ModuleList(
            [_CrossAttentionQueryBlock(d_model=d_model, nhead=nhead) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def set_normalizers(self, globals_x: torch.Tensor, y_scale: torch.Tensor) -> None:
        self.g_mean.copy_(globals_x.mean(dim=0, keepdim=True))
        self.g_std.copy_(globals_x.std(dim=0, keepdim=True).clamp(min=1e-5))
        self.y_scale.copy_(y_scale.view(1, 3))

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        bsz = batch.action.shape[0]
        side_sign = torch.cat(
            [
                -torch.ones(bsz, NUM_SIDE_BINS, 1, device=batch.action.device),
                torch.ones(bsz, NUM_SIDE_BINS, 1, device=batch.action.device),
            ],
            dim=1,
        )
        bin_feats = torch.stack(
            [
                torch.log(batch.rel_liq.clamp(1e-4, 1e4)),
                torch.log1p(batch.eth_cap),
                torch.log1p(batch.liq_supply),
                batch.log_delta * 20.0,
            ],
            dim=-1,
        )
        bin_tokens = self.bin_proj(torch.cat([bin_feats, side_sign], dim=-1))
        bin_tokens = F.gelu(self.spatial_mix(bin_tokens.transpose(1, 2)).transpose(1, 2)) + self.pos_emb

        g_norm = (batch.globals_x - self.g_mean) / self.g_std
        g_tok = self.global_proj(g_norm).unsqueeze(1)

        local_cap = torch.where(
            batch.action >= 0.0,
            batch.eth_cap[:, 0:1],
            batch.eth_cap[:, NUM_SIDE_BINS : NUM_SIDE_BINS + 1],
        )
        a_tok = self.action_proj(_action_features(batch.action, local_cap)).unsqueeze(1)

        q_tokens = torch.cat([a_tok, g_tok], dim=1)  # (B, 2, d_model)
        kv_tokens = torch.cat([q_tokens, bin_tokens], dim=1)  # (B, 34, d_model)
        for blk in self.blocks:
            q_tokens = blk(q_tokens, kv_tokens)
        return self.head(q_tokens[:, 0, :]) * self.y_scale


class LatentJEPAWorldModel(nn.Module):
    """Action-conditioned JEPA with EMA target encoder, VICReg, and detached readout probe."""

    def __init__(self, state_dim: int, latent_dim: int = 128, hidden_dim: int = 192, depth: int = 1) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("s_mean", torch.zeros(1, state_dim))
        self.register_buffer("s_std", torch.ones(1, state_dim))
        self.register_buffer("y_scale", torch.ones(1, 3))

        self.ctx_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            *[_ResidualBlock(hidden_dim) for _ in range(depth)],
            nn.Linear(hidden_dim, latent_dim),
        )
        # Target encoder maps post-transition summary [s_post_approx] -> latent_dim.
        self.tgt_proj_in = nn.Linear(state_dim + 3, state_dim)
        self.tgt_encoder = copy.deepcopy(self.ctx_encoder)
        for p in self.tgt_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            *[_ResidualBlock(hidden_dim) for _ in range(max(depth - 1, 0))],
            nn.Linear(hidden_dim, latent_dim),
        )
        # Detached readout probe (`causal-jepa` protocol: probe never backprops into encoder).
        probe_hidden = max(128, hidden_dim // 2)
        self.probe = nn.Sequential(
            nn.Linear(2 * latent_dim + 6, probe_hidden),
            nn.LayerNorm(probe_hidden),
            nn.SiLU(),
            nn.Linear(probe_hidden, 3),
        )

    def _extract_state_only(self, batch: BatchTensors) -> torch.Tensor:
        # Everything in flat_x except the trailing 5 action features.
        return batch.flat_x[:, :-5]

    def set_normalizers(self, batch: BatchTensors, y_scale: torch.Tensor) -> None:
        s = self._extract_state_only(batch)
        self.s_mean.copy_(s.mean(dim=0, keepdim=True))
        self.s_std.copy_(s.std(dim=0, keepdim=True).clamp(min=1e-5))
        self.y_scale.copy_(y_scale.view(1, 3))

    @torch.no_grad()
    def update_ema(self, tau: float = 0.99) -> None:
        for p_tgt, p_ctx in zip(self.tgt_encoder.parameters(), self.ctx_encoder.parameters(), strict=True):
            p_tgt.data.mul_(tau).add_(p_ctx.data, alpha=1.0 - tau)

    def jepa_and_probe_loss(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor]:
        s_norm = (self._extract_state_only(batch) - self.s_mean) / self.s_std
        local_cap = torch.where(
            batch.action >= 0.0,
            batch.eth_cap[:, 0:1],
            batch.eth_cap[:, NUM_SIDE_BINS : NUM_SIDE_BINS + 1],
        )
        act_f = _action_features(batch.action, local_cap)

        z_ctx = self.ctx_encoder(s_norm)
        z_pred = self.predictor(torch.cat([z_ctx, act_f], dim=-1))

        y_std_true = batch.y / self.y_scale
        s_post_in = self.tgt_proj_in(torch.cat([s_norm, y_std_true], dim=-1))
        with torch.no_grad():
            z_tgt = self.tgt_encoder(s_post_in)

        inv_loss = F.mse_loss(z_pred, z_tgt)

        # VICReg variance + covariance regularization to prevent latent collapse.
        std_z = torch.sqrt(z_ctx.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std_z))
        z_centered = z_ctx - z_ctx.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / max(z_ctx.shape[0] - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag.pow(2).sum()) / self.latent_dim

        ssl_loss = inv_loss + 1.0 * var_loss + 0.04 * cov_loss + 0.01 * self.tgt_proj_in.weight.pow(2).mean()

        # Detached probe loss: strictly evaluates representation without leaking y gradients!
        probe_in = torch.cat([z_ctx.detach(), z_pred.detach(), act_f], dim=-1)
        y_hat = self.probe(probe_in) * self.y_scale
        diff = (y_hat - batch.y) / self.y_scale
        probe_loss = torch.mean(diff * diff)
        return ssl_loss + probe_loss, y_hat

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        s_norm = (self._extract_state_only(batch) - self.s_mean) / self.s_std
        local_cap = torch.where(
            batch.action >= 0.0,
            batch.eth_cap[:, 0:1],
            batch.eth_cap[:, NUM_SIDE_BINS : NUM_SIDE_BINS + 1],
        )
        act_f = _action_features(batch.action, local_cap)
        z_ctx = self.ctx_encoder(s_norm)
        z_pred = self.predictor(torch.cat([z_ctx, act_f], dim=-1))
        return self.probe(torch.cat([z_ctx, z_pred, act_f], dim=-1)) * self.y_scale


class ConservationIntegralWorldModel(nn.Module):
    """Structure-preserving neural world model with equivariant per-bin capacity integration.

    Learns a shared per-bin capacity function g_theta(v_b) and liquidation coupling beta_theta
    solely from observational swaps, advancing action a_t via sequential reserve conservation.
    """

    def __init__(self, hidden_dim: int = 48) -> None:
        super().__init__()
        self.bin_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize small residual around log_rel_liq so initial map is smooth & trainable.
        nn.init.zeros_(self.bin_net[-1].weight)
        nn.init.zeros_(self.bin_net[-1].bias)

        # Learned global capacity scale, liquidation coupling, and fee offset (randomly initialized).
        self.raw_cap_scale = nn.Parameter(torch.tensor(0.45))  # softplus(0.45) ~ 0.94
        self.raw_liq_coupling = nn.Parameter(torch.tensor(-1.5))  # softplus(-1.5) ~ 0.20
        self.raw_fee_bps = nn.Parameter(torch.tensor(1.5))  # softplus(1.5) * 1e-4 ~ 1.7 bps
        self.liq_head_scale = nn.Parameter(torch.tensor(0.85))

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        is_sell = batch.action >= 0.0  # (B, 1)
        abs_a = torch.abs(batch.action).squeeze(-1)  # (B,)

        # Select 16 directional bins: 0..15 for sell, 16..31 for buy.
        rel_liq = torch.where(is_sell, batch.rel_liq[:, :NUM_SIDE_BINS], batch.rel_liq[:, NUM_SIDE_BINS:])
        unit_cap = torch.where(is_sell, batch.unit_cap[:, :NUM_SIDE_BINS], batch.unit_cap[:, NUM_SIDE_BINS:])
        log_delta = torch.where(
            is_sell, batch.log_delta[:, :NUM_SIDE_BINS], batch.log_delta[:, NUM_SIDE_BINS:]
        )
        liq_sup = torch.where(
            is_sell,
            batch.liq_supply[:, :NUM_SIDE_BINS],
            torch.zeros_like(batch.liq_supply[:, :NUM_SIDE_BINS]),
        )

        log_rel = torch.log(rel_liq.clamp(1e-4, 1e4))
        bin_in = torch.stack([log_rel, torch.abs(log_delta) * 20.0], dim=-1)  # (B, 16, 2)
        # Equivariant per-bin log-multiplier prediction across all 16 bins.
        log_mult = (log_rel + self.bin_net(bin_in).squeeze(-1)).clamp(-7.0, 7.0)
        eff_mult = torch.exp(log_mult)  # (B, 16)

        cap_scale = F.softplus(self.raw_cap_scale) + 1e-3
        liq_beta = F.softplus(self.raw_liq_coupling)
        fee_rate = F.softplus(self.raw_fee_bps) * 1e-4

        bin_cap = (cap_scale * unit_cap * eff_mult).clamp(min=1e-6)  # (B, 16)

        rem_total = abs_a
        rem_trader = abs_a
        dlogp = torch.zeros_like(abs_a)
        exec_val = torch.zeros_like(abs_a)
        cum_log_offset = torch.zeros_like(abs_a)
        active_log_liq = torch.zeros_like(abs_a)  # delta log(L_post / L_0) is 0 in bin 0

        sell_mask_1d = is_sell.squeeze(-1).float()

        for j in range(NUM_SIDE_BINS):
            cap_j = bin_cap[:, j]
            dl_j = log_delta[:, j]

            # Trader's own swap execution value (cannot include Stage-2 liquidator proceeds).
            fill_trader_j = torch.clamp(rem_trader / cap_j, 0.0, 1.0)
            mid_trader_log = cum_log_offset + 0.5 * fill_trader_j * dl_j
            pf_trader = torch.where(
                is_sell.squeeze(-1), torch.exp(mid_trader_log), torch.exp(-mid_trader_log)
            )
            exec_val = exec_val + (fill_trader_j * cap_j) * pf_trader
            rem_trader = F.relu(rem_trader - cap_j)

            # Total flow (trader + endogenous liquidations) governs post-state price & liquidity.
            fill_j = torch.clamp(rem_total / cap_j, 0.0, 1.0)
            dlogp = dlogp + fill_j * dl_j
            cum_log_offset = cum_log_offset + dl_j

            rem_next = F.relu(rem_total - cap_j) + sell_mask_1d * fill_j * liq_beta * liq_sup[:, j]

            if j + 1 < NUM_SIDE_BINS:
                entered_next = torch.clamp(rem_next / (0.05 * cap_j + 1e-6), 0.0, 1.0)
                active_log_liq = (1.0 - entered_next) * active_log_liq + entered_next * log_mult[:, j + 1]

            rem_total = rem_next

        # Overflow beyond 17% grid uses outermost bin's learned marginal impact.
        outer_slope = log_delta[:, -1] / bin_cap[:, -1]
        dlogp_extra = rem_total * outer_slope
        dlogp = (dlogp + dlogp_extra).clamp(-1.5, 1.5)

        extra_trader_dlogp = rem_trader * outer_slope
        extra_mid = cum_log_offset + 0.5 * extra_trader_dlogp
        extra_pf = torch.where(is_sell.squeeze(-1), torch.exp(extra_mid), torch.exp(-extra_mid))
        exec_val = exec_val + rem_trader * extra_pf

        eff_ratio = (1.0 - fee_rate) * exec_val / (abs_a + 1e-8)
        eff_ratio = torch.where(abs_a > 1e-8, eff_ratio, torch.ones_like(eff_ratio))
        log_eff = torch.log(eff_ratio.clamp(min=1e-4, max=1.0))

        dlog_liq = self.liq_head_scale * active_log_liq
        dlog_liq = torch.where(abs_a > 1e-8, dlog_liq, torch.zeros_like(dlog_liq))

        return torch.stack([dlogp, dlog_liq, log_eff], dim=-1)


@dataclass
class TrainedModelBundle:
    """Holds both the best-validation checkpoint and a target-matched-loss checkpoint."""

    family: str
    seed: int
    best_model: nn.Module
    matched_model: nn.Module
    best_val_loss: float
    matched_val_loss: float
    val_curve: list[float]


def evaluate_model_predictions(
    model: nn.Module,
    batch: BatchTensors,
    batch_size: int = 1024,
) -> np.ndarray:
    """Deterministic batched forward pass returning numpy predictions of shape (N, 3)."""
    model.eval()
    preds: list[np.ndarray] = []
    n = batch.action.shape[0]
    device = batch.action.device
    use_amp = device.type == "cuda" and not isinstance(model, ConservationIntegralWorldModel)
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_amp else torch.float32)
    )
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = torch.arange(start, end, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(batch.slice(idx))
            preds.append(out.detach().float().cpu().numpy().astype(np.float64))
    return np.concatenate(preds, axis=0)


def train_world_model(
    family: str,
    train_batch: BatchTensors,
    val_batch: BatchTensors,
    scaler: TargetScaler,
    *,
    seed: int,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1.5e-3,
    target_match_loss: float | None = None,
    model_kwargs: dict[str, int] | None = None,
) -> TrainedModelBundle:
    """Train a world model strictly on observational transitions with deterministic seeding."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = train_batch.action.device
    y_scale_t = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
    in_dim = train_batch.flat_x.shape[1]
    kw = model_kwargs or {}

    if family == "mlp_direct":
        model: nn.Module = DirectMLPWorldModel(in_dim=in_dim, **kw).to(device)
        model.set_normalizers(train_batch.flat_x, y_scale_t)
    elif family == "seq_transformer":
        model = SequenceTransformerWorldModel(**kw).to(device)
        model.set_normalizers(train_batch.globals_x, y_scale_t)
    elif family == "latent_jepa":
        model = LatentJEPAWorldModel(state_dim=in_dim - 5, **kw).to(device)
        model.set_normalizers(train_batch, y_scale_t)
    elif family == "conservation_wm":
        model = ConservationIntegralWorldModel(**kw).to(device)
    else:
        raise ValueError(f"Unknown model family: {family}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    use_amp = device.type == "cuda" and family != "conservation_wm"
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_amp else torch.float32)
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    n_train = train_batch.action.shape[0]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    matched_val = float("inf")
    matched_gap = float("inf")
    matched_state = copy.deepcopy(model.state_dict())
    val_curve: list[float] = []

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            sub = train_batch.slice(idx)
            optimizer.zero_grad(set_to_none=True)

            if isinstance(model, LatentJEPAWorldModel):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    loss, _ = model.jepa_and_probe_loss(sub)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                model.update_ema(tau=0.99)
            else:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    pred = model(sub)
                    diff = (pred - sub.y) / y_scale_t.view(1, 3)
                    loss = torch.mean(diff * diff)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

        scheduler.step()

        # Evaluate on observational validation split.
        val_preds = evaluate_model_predictions(model, val_batch)
        val_y = val_batch.y.detach().cpu().numpy().astype(np.float64)
        val_loss = float(np.mean(scaler.per_sample_loss(val_preds, val_y)))
        val_curve.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if target_match_loss is not None:
            gap = abs(val_loss - target_match_loss)
            if gap < matched_gap:
                matched_gap = gap
                matched_val = val_loss
                matched_state = copy.deepcopy(model.state_dict())

        if (_epoch + 1) % 5 == 0 or _epoch + 1 == epochs:
            import sys
            pct = int(((_epoch + 1) / epochs) * 20)
            pbar = "=" * pct + "-" * (20 - pct)
            sys.stdout.write(f"\r    [{pbar}] epoch {_epoch+1:2d}/{epochs:2d} | val_loss={val_loss:.4f} (best={best_val:.4f})")
            sys.stdout.flush()

    print("", flush=True)
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(best_state)
    best_model.eval()

    matched_model = copy.deepcopy(model)
    if target_match_loss is not None:
        matched_model.load_state_dict(matched_state)
    else:
        matched_model.load_state_dict(best_state)
        matched_val = best_val
    matched_model.eval()

    return TrainedModelBundle(
        family=family,
        seed=seed,
        best_model=best_model,
        matched_model=matched_model,
        best_val_loss=best_val,
        matched_val_loss=matched_val,
        val_curve=val_curve,
    )
