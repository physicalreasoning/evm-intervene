"""Evaluation metrics, matched-loss screening, cluster bootstrap CIs, and Holm correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from evm_intervene.constants import BOOTSTRAP_REPLICATES, MATCHED_LOSS_REL_TOL


@dataclass(frozen=True)
class TargetScaler:
    """Training-fold-only variance standardization for 3D transition targets."""

    scale: np.ndarray  # shape (3,)

    @classmethod
    def from_targets(cls, y_train: np.ndarray) -> TargetScaler:
        std = np.std(y_train, axis=0)
        # Floor scale to avoid zero-division on near-constant observational components.
        scale = np.maximum(std, np.array([1e-4, 1e-3, 1e-4], dtype=np.float64))
        return cls(scale=scale)

    def per_sample_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Mean standardized squared error per transition."""
        diff = (y_pred - y_true) / self.scale[None, :]
        return np.mean(diff * diff, axis=1)


def price_rmse_bps(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root-mean-squared log-price error in basis points (10,000 bps = 100%)."""
    err = y_pred[:, 0] - y_true[:, 0]
    return float(1e4 * np.sqrt(np.mean(err * err)))


def liquidity_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root-mean-squared log-liquidity transition error."""
    err = y_pred[:, 1] - y_true[:, 1]
    return float(np.sqrt(np.mean(err * err)))


def passes_matched_loss_screen(
    loss_a: float,
    loss_b: float,
    rel_tol: float = MATCHED_LOSS_REL_TOL,
) -> bool:
    """Check |L_a - L_b| / min(L_a, L_b) <= rel_tol (docs/DESIGN.md Gate G1)."""
    denom = max(min(loss_a, loss_b), 1e-12)
    return abs(loss_a - loss_b) / denom <= rel_tol


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 20261024,
) -> tuple[float, float, float]:
    """Block/cluster bootstrap mean and percentile 95% CI over held-out checkpoint blocks."""
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    n_c = len(unique_clusters)
    point = float(np.mean(values))
    if n_c <= 1:
        return point, point, point

    # Pre-aggregate cluster sums and counts for fast exact cluster resampling.
    c_sums = np.bincount(inverse, weights=values, minlength=n_c)
    c_counts = np.bincount(inverse, minlength=n_c).astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_c, size=(n_boot, n_c))
    boot_means = np.sum(c_sums[idx], axis=1) / np.sum(c_counts[idx], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, float(lo), float(hi)


def cluster_bootstrap_paired_gap(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 20261024,
) -> dict[str, float]:
    """Paired cluster-bootstrap test for intervention gap Delta = E[loss_a - loss_b]."""
    diff = loss_a - loss_b
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    n_c = len(unique_clusters)
    point = float(np.mean(diff))
    mean_b = float(np.mean(loss_b))
    rel_gap = point / max(mean_b, 1e-12)

    if n_c <= 1:
        return {
            "gap": point,
            "ci_low": point,
            "ci_high": point,
            "rel_gap": rel_gap,
            "p_two_sided": 1.0,
        }

    c_sums = np.bincount(inverse, weights=diff, minlength=n_c)
    c_counts = np.bincount(inverse, minlength=n_c).astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_c, size=(n_boot, n_c))
    boot = np.sum(c_sums[idx], axis=1) / np.sum(c_counts[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Two-sided bootstrap sign tail probability with finite-sample +1 floor.
    le_zero = int(np.sum(boot <= 0.0))
    ge_zero = int(np.sum(boot >= 0.0))
    p_two_sided = min(1.0, 2.0 * (min(le_zero, ge_zero) + 1.0) / (n_boot + 1.0))

    return {
        "gap": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "rel_gap": float(rel_gap),
        "p_two_sided": float(p_two_sided),
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni family-wise error adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        val = min(1.0, p_values[orig_idx] * (m - rank))
        running_max = max(running_max, val)
        adjusted[orig_idx] = running_max
    return adjusted.tolist()


def spearman_rank_ci(
    obs_losses: np.ndarray,
    int_losses: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 20261024,
) -> dict[str, float]:
    """Spearman rank correlation rho(L_obs, E_int) with bootstrap 95% CI (Gate G2)."""
    n = len(obs_losses)
    if n < 3:
        return {"rho": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}
    res = spearmanr(obs_losses, int_losses)
    rho = float(res.statistic)
    pval = float(res.pvalue)

    rng = np.random.default_rng(seed)
    boot_rhos: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(obs_losses[idx])) < 2 or len(np.unique(int_losses[idx])) < 2:
            continue
        r_b = float(spearmanr(obs_losses[idx], int_losses[idx]).statistic)
        if not np.isnan(r_b):
            boot_rhos.append(r_b)

    if len(boot_rhos) == 0:
        return {"rho": rho, "p_value": pval, "ci_low": rho, "ci_high": rho}
    lo, hi = np.percentile(boot_rhos, [2.5, 97.5])
    return {"rho": rho, "p_value": pval, "ci_low": float(lo), "ci_high": float(hi)}
