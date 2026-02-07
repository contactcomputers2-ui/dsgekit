"""Marginal data density estimators for Bayesian model comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import special

from dsgekit.estimation.likelihood import build_objective
from dsgekit.estimation.mcmc import MCMCResult
from dsgekit.estimation.mle import MAPResult, estimate_map
from dsgekit.estimation.posterior import (
    build_log_prior_evaluator,
    infer_estimation_problem,
)
from dsgekit.exceptions import EstimationError

if TYPE_CHECKING:
    import pandas as pd

    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


def _resolve_observation_count(data: pd.DataFrame | NDArray[np.float64]) -> int:
    if hasattr(data, "shape"):
        return int(data.shape[0])
    return int(len(data))


def _compute_numerical_hessian(
    objective: callable,
    theta_hat: NDArray[np.float64],
    *,
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Finite-difference Hessian with cached objective evaluations."""
    if eps <= 0.0 or not np.isfinite(eps):
        raise EstimationError(f"hessian eps must be finite and > 0, got {eps}")

    theta = np.asarray(theta_hat, dtype=np.float64).reshape(-1)
    k = int(theta.shape[0])
    hess = np.zeros((k, k), dtype=np.float64)
    eval_cache: dict[bytes, float] = {}

    def eval_point(x: NDArray[np.float64]) -> float:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        key = arr.tobytes()
        if key not in eval_cache:
            eval_cache[key] = float(objective(arr))
        return eval_cache[key]

    f0 = eval_point(theta)

    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k, dtype=np.float64)
            ej = np.zeros(k, dtype=np.float64)
            ei[i] = eps
            ej[j] = eps

            if i == j:
                fpp = eval_point(theta + ei + ei)
                fmm = eval_point(theta - ei - ei)
                hess[i, i] = (fpp - 2.0 * f0 + fmm) / (4.0 * eps * eps)
                continue

            fpp = eval_point(theta + ei + ej)
            fpm = eval_point(theta + ei - ej)
            fmp = eval_point(theta - ei + ej)
            fmm = eval_point(theta - ei - ej)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            hess[j, i] = hess[i, j]

    return hess


def _laplace_log_mdd_from_mode(
    *,
    log_posterior_mode: float,
    hessian: NDArray[np.float64],
) -> tuple[float, float]:
    """Compute Laplace log-MDD from mode value and Hessian."""
    if not np.isfinite(log_posterior_mode):
        raise EstimationError("log_posterior_mode must be finite for Laplace approximation")

    mat = np.asarray(hessian, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise EstimationError("hessian must be a square 2D matrix")

    sign, logdet = np.linalg.slogdet(mat)
    if not np.isfinite(logdet) or sign <= 0.0:
        raise EstimationError(
            "Hessian at MAP is not positive definite; Laplace approximation is invalid"
        )

    k = int(mat.shape[0])
    log_mdd = log_posterior_mode + 0.5 * k * math.log(2.0 * math.pi) - 0.5 * logdet
    return float(log_mdd), float(logdet)


def _harmonic_mean_log_mdd(log_likelihood_samples: NDArray[np.float64]) -> float:
    """Harmonic-mean log-MDD from posterior log-likelihood draws."""
    ll = np.asarray(log_likelihood_samples, dtype=np.float64).reshape(-1)
    finite = ll[np.isfinite(ll)]
    if finite.size == 0:
        raise EstimationError("No finite log-likelihood samples available for harmonic mean")

    log_inv_lik_mean = float(special.logsumexp(-finite) - math.log(float(finite.size)))
    return float(-log_inv_lik_mean)


@dataclass
class MarginalDataDensityResult:
    """Result container for marginal data density estimation."""

    method: str
    log_mdd: float
    n_params: int
    n_obs: int
    n_samples: int | None = None
    log_posterior_mode: float | None = None
    hessian_logdet: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Marginal Data Density",
            "=" * 50,
            f"  Method:          {self.method}",
            f"  Log MDD:         {self.log_mdd:.6f}",
            f"  Parameters:      {self.n_params}",
            f"  Observations:    {self.n_obs}",
        ]
        if self.n_samples is not None:
            lines.append(f"  Samples used:    {self.n_samples}")
        if self.log_posterior_mode is not None:
            lines.append(f"  Log posterior*:  {self.log_posterior_mode:.6f}")
        if self.hessian_logdet is not None:
            lines.append(f"  log|H| at mode:  {self.hessian_logdet:.6f}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  Note: {note}")
        return "\n".join(lines)


def estimate_mdd_laplace(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    data: pd.DataFrame | NDArray[np.float64],
    observables: list[str],
    param_names: list[str] | None,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    map_result: MAPResult | None = None,
    measurement_error: float | NDArray[np.float64] | None = None,
    demean: bool = True,
    prior_weight: float = 1.0,
    require_priors: bool = True,
    penalty: float = 1e10,
    map_method: str = "L-BFGS-B",
    map_options: dict | None = None,
    hessian_eps: float = 1e-4,
) -> MarginalDataDensityResult:
    """Estimate log marginal data density via Laplace approximation."""
    if prior_weight <= 0.0:
        raise EstimationError(f"prior_weight must be > 0, got {prior_weight}")

    resolved_names, resolved_bounds = infer_estimation_problem(
        calibration=calibration,
        param_names=param_names,
        bounds=bounds,
    )

    notes: list[str] = []
    if map_result is None:
        map_result = estimate_map(
            model,
            steady_state,
            calibration,
            data,
            observables,
            resolved_names,
            bounds=resolved_bounds,
            measurement_error=measurement_error,
            demean=demean,
            method=map_method,
            options=map_options,
            compute_se=False,
            prior_weight=prior_weight,
            require_priors=require_priors,
            penalty=penalty,
        )
    else:
        notes.append("Using externally supplied MAP result.")

    if not map_result.success:
        raise EstimationError(
            f"MAP optimization did not converge; cannot compute Laplace MDD ({map_result.message})"
        )

    theta_hat = np.array(
        [float(map_result.parameters[name]) for name in resolved_names],
        dtype=np.float64,
    )

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

    def neg_log_posterior(theta: NDArray[np.float64]) -> float:
        nll = float(nll_objective(theta))
        if not np.isfinite(nll) or nll >= penalty:
            return float(penalty)
        lp = float(log_prior(theta))
        if not np.isfinite(lp):
            return float(penalty)
        return float(nll - prior_weight * lp)

    hessian = _compute_numerical_hessian(neg_log_posterior, theta_hat, eps=hessian_eps)
    log_mdd, hessian_logdet = _laplace_log_mdd_from_mode(
        log_posterior_mode=float(map_result.log_posterior),
        hessian=hessian,
    )

    return MarginalDataDensityResult(
        method="laplace",
        log_mdd=log_mdd,
        n_params=len(resolved_names),
        n_obs=_resolve_observation_count(data),
        log_posterior_mode=float(map_result.log_posterior),
        hessian_logdet=hessian_logdet,
        notes=notes,
    )


def estimate_mdd_harmonic_mean(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    data: pd.DataFrame | NDArray[np.float64],
    observables: list[str],
    param_names: list[str] | None,
    mcmc_result: MCMCResult,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    measurement_error: float | NDArray[np.float64] | None = None,
    demean: bool = True,
    penalty: float = 1e10,
    max_samples: int | None = None,
    seed: int | None = None,
) -> MarginalDataDensityResult:
    """Estimate log marginal data density with harmonic mean estimator.

    Notes:
        This estimator is known to be high-variance in practice.
        It is provided mainly for parity/diagnostics against Laplace.
    """
    resolved_names, _ = infer_estimation_problem(
        calibration=calibration,
        param_names=param_names,
        bounds=bounds,
    )
    if list(mcmc_result.param_names) != list(resolved_names):
        raise EstimationError(
            "mcmc_result.param_names must match resolved param_names order for harmonic mean"
        )

    samples = np.asarray(mcmc_result.samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise EstimationError("mcmc_result.samples must be a non-empty 2D array")

    notes: list[str] = []
    if max_samples is not None:
        if max_samples < 1:
            raise EstimationError(f"max_samples must be >= 1, got {max_samples}")
        if max_samples < samples.shape[0]:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(samples.shape[0], size=max_samples, replace=False))
            samples = samples[idx, :]
            notes.append(
                f"Subsampled posterior draws from {mcmc_result.samples.shape[0]} to {samples.shape[0]}."
            )

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

    log_likelihood = np.full(samples.shape[0], -np.inf, dtype=np.float64)
    for i, theta in enumerate(samples):
        nll = float(nll_objective(theta))
        if np.isfinite(nll) and nll < penalty:
            log_likelihood[i] = -nll

    finite_mask = np.isfinite(log_likelihood)
    finite_count = int(np.sum(finite_mask))
    if finite_count == 0:
        raise EstimationError(
            "No finite log-likelihood evaluations from posterior draws; harmonic mean failed"
        )
    if finite_count < log_likelihood.shape[0]:
        notes.append(
            f"Dropped {log_likelihood.shape[0] - finite_count} non-finite likelihood draws."
        )

    log_mdd = _harmonic_mean_log_mdd(log_likelihood[finite_mask])
    notes.append("Harmonic mean can be unstable; interpret with caution.")

    return MarginalDataDensityResult(
        method="harmonic_mean",
        log_mdd=log_mdd,
        n_params=len(resolved_names),
        n_obs=_resolve_observation_count(data),
        n_samples=finite_count,
        notes=notes,
    )

