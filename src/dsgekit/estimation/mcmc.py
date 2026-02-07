"""Metropolis-Hastings MCMC for DSGE posterior sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dsgekit.estimation.likelihood import build_objective
from dsgekit.estimation.posterior import (
    build_log_prior_evaluator,
    infer_estimation_problem,
    prepare_initial_guess_and_bounds,
)
from dsgekit.exceptions import EstimationError

if TYPE_CHECKING:
    import pandas as pd

    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


def _resolve_proposal_scale(
    proposal_scale: float | NDArray[np.float64],
    n_params: int,
) -> NDArray[np.float64]:
    """Normalize proposal scale into per-parameter std vector."""
    if np.isscalar(proposal_scale):
        scale = float(proposal_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise EstimationError(f"proposal_scale must be finite and > 0, got {scale}")
        return np.full(n_params, scale, dtype=np.float64)

    arr = np.asarray(proposal_scale, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n_params:
        raise EstimationError(
            f"proposal_scale length must match parameter count ({n_params}), "
            f"got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise EstimationError("proposal_scale vector must be finite and strictly positive")
    return arr


def _clip_to_bounds(
    theta: NDArray[np.float64],
    bounds: list[tuple[float | None, float | None]],
) -> NDArray[np.float64]:
    """Clip theta to finite bounds if present."""
    out = np.asarray(theta, dtype=np.float64).copy()
    for i, (lb, ub) in enumerate(bounds):
        if lb is not None:
            out[i] = max(out[i], float(lb))
        if ub is not None:
            out[i] = min(out[i], float(ub))
    return out


def _in_bounds(
    theta: NDArray[np.float64],
    bounds: list[tuple[float | None, float | None]],
) -> bool:
    """Check if theta satisfies box constraints."""
    for value, (lb, ub) in zip(theta, bounds, strict=True):
        if lb is not None and value < lb:
            return False
        if ub is not None and value > ub:
            return False
    return True


def _autocorrelation_at_lag(
    centered: NDArray[np.float64],
    lag: int,
    variance: float,
) -> float:
    """Estimate lag-k autocorrelation for a centered series."""
    n = centered.shape[0]
    if lag <= 0 or lag >= n:
        raise EstimationError(f"lag must satisfy 1 <= lag < n, got lag={lag}, n={n}")
    cov = float(np.dot(centered[:-lag], centered[lag:]) / (n - lag))
    return cov / variance


def _effective_sample_size_1d(values: NDArray[np.float64]) -> float:
    """Geyer-style ESS estimator with positive-pair truncation."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n <= 2:
        return float(n)

    centered = x - float(x.mean())
    variance = float(np.dot(centered, centered) / n)
    if not np.isfinite(variance) or variance <= 0.0:
        return float(n)

    rho_pair_sum = 0.0
    prev_pair = float("inf")
    max_lag = max(1, n - 1)
    lag = 1
    while lag <= max_lag:
        rho_odd = _autocorrelation_at_lag(centered, lag, variance)
        rho_even = 0.0
        if lag + 1 <= max_lag:
            rho_even = _autocorrelation_at_lag(centered, lag + 1, variance)

        pair = float(rho_odd + rho_even)
        if not np.isfinite(pair) or pair <= 0.0:
            break

        if pair > prev_pair:
            pair = prev_pair
        prev_pair = pair
        rho_pair_sum += pair
        lag += 2

    tau = 1.0 + 2.0 * rho_pair_sum
    if not np.isfinite(tau) or tau <= 0.0:
        return 1.0

    ess = float(n / tau)
    return float(np.clip(ess, 1.0, float(n)))


def _split_rhat(
    samples: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int, str | None]:
    """Compute split-chain R-hat from post burn-in samples."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise EstimationError(f"samples must be 2D [n_samples, n_params], got ndim={arr.ndim}")
    n_samples, n_params = arr.shape
    if n_samples < 4:
        return np.full(n_params, np.nan, dtype=np.float64), 1, "R-hat unavailable: need >= 4 samples."

    n = n_samples // 2
    if n < 2:
        return np.full(n_params, np.nan, dtype=np.float64), 1, "R-hat unavailable: split chains are too short."

    trimmed = arr[: 2 * n, :]
    chains = np.stack((trimmed[:n, :], trimmed[n:, :]), axis=0)  # [2, n, p]

    chain_means = chains.mean(axis=1)  # [m, p]
    chain_vars = chains.var(axis=1, ddof=1)  # [m, p]
    w = chain_vars.mean(axis=0)  # within-chain variance
    b = n * chain_means.var(axis=0, ddof=1)  # between-chain variance
    var_hat = ((n - 1.0) / n) * w + (1.0 / n) * b

    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / w)

    near_constant = (w <= 1e-14) & (b <= 1e-14)
    rhat = np.where(near_constant, 1.0, rhat)
    rhat = np.where(np.isfinite(rhat), rhat, np.nan).astype(np.float64, copy=False)

    note = None
    if 2 * n != n_samples:
        note = "R-hat computed on an even-length prefix (one sample dropped)."
    return rhat, 2, note


@dataclass
class MCMCDiagnostics:
    """MCMC diagnostics over post burn-in samples."""

    param_names: list[str]
    n_samples: int
    n_chains_for_rhat: int
    acceptance_rate: float
    ess: dict[str, float]
    r_hat: dict[str, float]
    notes: list[str]

    def summary(self) -> str:
        lines = [
            "MCMC Diagnostics",
            "=" * 50,
            f"  Saved samples:   {self.n_samples}",
            f"  Acceptance rate: {self.acceptance_rate:.3f}",
            f"  R-hat chains:    {self.n_chains_for_rhat} (split-chain)",
            "",
            f"  {'Parameter':<15} {'ESS':>12} {'R-hat':>12}",
            f"  {'-' * 15} {'-' * 12} {'-' * 12}",
        ]
        for name in self.param_names:
            ess = self.ess[name]
            rhat = self.r_hat[name]
            rhat_str = f"{rhat:12.4f}" if np.isfinite(rhat) else f"{'n/a':>12}"
            lines.append(f"  {name:<15} {ess:12.1f} {rhat_str}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  Note: {note}")
        return "\n".join(lines)


@dataclass
class MCMCResult:
    """Output container for MH posterior sampling."""

    param_names: list[str]
    chain: NDArray[np.float64]
    log_posterior_chain: NDArray[np.float64]
    samples: NDArray[np.float64]
    log_posterior_samples: NDArray[np.float64]
    accepted: int
    acceptance_rate: float
    n_draws: int
    burn_in: int
    thin: int
    seed: int | None
    proposal_scale: NDArray[np.float64]
    _diagnostics_cache: MCMCDiagnostics | None = field(default=None, init=False, repr=False)

    def posterior_mean(self) -> dict[str, float]:
        return {
            name: float(self.samples[:, i].mean())
            for i, name in enumerate(self.param_names)
        }

    def posterior_std(self) -> dict[str, float]:
        return {
            name: float(self.samples[:, i].std(ddof=1))
            for i, name in enumerate(self.param_names)
        }

    def trace_dict(self, *, post_burn: bool = True) -> dict[str, NDArray[np.float64]]:
        """Return parameter traces as ``name -> vector``."""
        arr = self.samples if post_burn else self.chain
        return {
            name: arr[:, i].copy()
            for i, name in enumerate(self.param_names)
        }

    def diagnostics(self) -> MCMCDiagnostics:
        """Compute (and cache) standard MCMC diagnostics."""
        if self._diagnostics_cache is not None:
            return self._diagnostics_cache

        ess_values = {
            name: _effective_sample_size_1d(self.samples[:, i])
            for i, name in enumerate(self.param_names)
        }
        rhat_values, n_chains, rhat_note = _split_rhat(self.samples)
        rhat = {
            name: float(rhat_values[i])
            for i, name in enumerate(self.param_names)
        }
        notes: list[str] = []
        if rhat_note:
            notes.append(rhat_note)

        diag = MCMCDiagnostics(
            param_names=list(self.param_names),
            n_samples=int(self.samples.shape[0]),
            n_chains_for_rhat=n_chains,
            acceptance_rate=float(self.acceptance_rate),
            ess=ess_values,
            r_hat=rhat,
            notes=notes,
        )
        self._diagnostics_cache = diag
        return diag

    def summary(self) -> str:
        diag = self.diagnostics()
        lines = [
            "Metropolis-Hastings MCMC",
            "=" * 50,
            f"  Draws:           {self.n_draws}",
            f"  Burn-in:         {self.burn_in}",
            f"  Thin:            {self.thin}",
            f"  Saved samples:   {self.samples.shape[0]}",
            f"  Acceptance rate: {self.acceptance_rate:.3f}",
            "",
            f"  {'Parameter':<15} {'Mean':>12} {'Std':>12} {'ESS':>12} {'R-hat':>12}",
            f"  {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}",
        ]
        means = self.posterior_mean()
        stds = self.posterior_std()
        for name in self.param_names:
            rhat = diag.r_hat[name]
            rhat_str = f"{rhat:12.4f}" if np.isfinite(rhat) else f"{'n/a':>12}"
            lines.append(
                f"  {name:<15} {means[name]:12.6f} {stds[name]:12.6f} "
                f"{diag.ess[name]:12.1f} {rhat_str}"
            )
        if diag.notes:
            lines.append("")
            for note in diag.notes:
                lines.append(f"  Note: {note}")
        return "\n".join(lines)


def sample_posterior_mh(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    data: pd.DataFrame | NDArray[np.float64],
    observables: list[str],
    param_names: list[str] | None,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    measurement_error: float | NDArray[np.float64] | None = None,
    demean: bool = True,
    prior_weight: float = 1.0,
    require_priors: bool = True,
    n_draws: int = 5000,
    burn_in: int = 1000,
    thin: int = 1,
    proposal_scale: float | NDArray[np.float64] = 0.05,
    seed: int | None = None,
    initial_theta: NDArray[np.float64] | dict[str, float] | None = None,
    penalty: float = 1e10,
) -> MCMCResult:
    """Sample posterior with random-walk Metropolis-Hastings."""
    if n_draws < 1:
        raise EstimationError(f"n_draws must be >= 1, got {n_draws}")
    if burn_in < 0 or burn_in >= n_draws:
        raise EstimationError(f"burn_in must satisfy 0 <= burn_in < n_draws, got {burn_in}")
    if thin < 1:
        raise EstimationError(f"thin must be >= 1, got {thin}")
    if prior_weight <= 0.0:
        raise EstimationError(f"prior_weight must be > 0, got {prior_weight}")

    resolved_names, resolved_bounds = infer_estimation_problem(
        calibration=calibration,
        param_names=param_names,
        bounds=bounds,
    )
    k = len(resolved_names)
    scales = _resolve_proposal_scale(proposal_scale, k)

    x0_base, scipy_bounds = prepare_initial_guess_and_bounds(
        calibration=calibration,
        param_names=resolved_names,
        bounds=resolved_bounds,
    )
    if initial_theta is None:
        x0 = x0_base
    elif isinstance(initial_theta, dict):
        x0 = x0_base.copy()
        for i, name in enumerate(resolved_names):
            if name in initial_theta:
                x0[i] = float(initial_theta[name])
    else:
        arr = np.asarray(initial_theta, dtype=np.float64).reshape(-1)
        if arr.shape[0] != k:
            raise EstimationError(
                f"initial_theta length must be {k}, got {arr.shape[0]}"
            )
        x0 = arr
    x0 = _clip_to_bounds(x0, scipy_bounds)

    nll_objective = build_objective(
        model,
        steady_state,
        calibration,
        data,
        observables,
        resolved_names,
        measurement_error=measurement_error,
        demean=demean,
        penalty=penalty,
    )
    log_prior = build_log_prior_evaluator(
        calibration,
        resolved_names,
        require_priors=require_priors,
    )

    def log_posterior(theta: NDArray[np.float64]) -> float:
        nll = float(nll_objective(theta))
        if not np.isfinite(nll) or nll >= penalty:
            return float("-inf")
        lp = float(log_prior(theta))
        if not np.isfinite(lp):
            return float("-inf")
        return float(-nll + prior_weight * lp)

    rng = np.random.default_rng(seed)
    chain = np.zeros((n_draws, k), dtype=np.float64)
    logp_chain = np.full(n_draws, -np.inf, dtype=np.float64)

    current = x0.copy()
    current_logp = log_posterior(current)
    if not np.isfinite(current_logp):
        raise EstimationError(
            "Initial point has non-finite log posterior; "
            "adjust initial_theta/calibration/prior specifications"
        )

    accepted = 0
    for t in range(n_draws):
        proposal = current + rng.normal(loc=0.0, scale=scales, size=k)
        proposal_logp = (
            log_posterior(proposal) if _in_bounds(proposal, scipy_bounds) else float("-inf")
        )

        if np.isfinite(proposal_logp):
            log_alpha = proposal_logp - current_logp
            if math.log(rng.uniform()) < min(0.0, log_alpha):
                current = proposal
                current_logp = proposal_logp
                accepted += 1

        chain[t, :] = current
        logp_chain[t] = current_logp

    kept_chain = chain[burn_in::thin, :]
    kept_logp = logp_chain[burn_in::thin]
    acceptance_rate = accepted / float(n_draws)

    return MCMCResult(
        param_names=resolved_names,
        chain=chain,
        log_posterior_chain=logp_chain,
        samples=kept_chain,
        log_posterior_samples=kept_logp,
        accepted=accepted,
        acceptance_rate=acceptance_rate,
        n_draws=n_draws,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
        proposal_scale=scales,
    )


def estimate_mcmc(*args, **kwargs) -> MCMCResult:
    """Alias for ``sample_posterior_mh``."""
    return sample_posterior_mh(*args, **kwargs)
