"""Maximum likelihood estimation for DSGE models.

Uses ``scipy.optimize.minimize`` to find the parameter vector that
maximises the Kalman-filter log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from dsgekit.estimation.likelihood import build_objective
from dsgekit.estimation.posterior import (
    build_log_prior_evaluator,
    infer_estimation_problem,
    prepare_initial_guess_and_bounds,
)

if TYPE_CHECKING:
    import pandas as pd

    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MLEResult:
    """Results from maximum likelihood estimation.

    Attributes:
        parameters: Estimated parameter values.
        log_likelihood: Log-likelihood at the optimum.
        std_errors: Approximate standard errors (Hessian-based), or
            None if computation failed.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        n_obs: Number of observation periods.
        n_params: Number of estimated parameters.
        success: Whether the optimiser converged.
        message: Optimiser status message.
        n_iterations: Number of optimiser iterations.
    """

    parameters: dict[str, float]
    log_likelihood: float
    std_errors: dict[str, float] | None
    aic: float
    bic: float
    n_obs: int
    n_params: int
    success: bool
    message: str
    n_iterations: int

    def summary(self) -> str:
        """Human-readable summary of MLE results."""
        lines = [
            "Maximum Likelihood Estimation",
            "=" * 50,
            f"  Converged:      {self.success}",
            f"  Log-likelihood: {self.log_likelihood:.4f}",
            f"  AIC:            {self.aic:.4f}",
            f"  BIC:            {self.bic:.4f}",
            f"  Observations:   {self.n_obs}",
            f"  Parameters:     {self.n_params}",
            f"  Iterations:     {self.n_iterations}",
            "",
            f"  {'Parameter':<15} {'Estimate':>12} {'Std.Err':>12}",
            f"  {'-' * 15} {'-' * 12} {'-' * 12}",
        ]
        for name, value in self.parameters.items():
            se = self.std_errors.get(name, None) if self.std_errors else None
            se_str = f"{se:12.6f}" if se is not None else "         n/a"
            lines.append(f"  {name:<15} {value:12.6f} {se_str}")
        return "\n".join(lines)


@dataclass
class MAPResult:
    """Results from posterior mode (MAP) optimization."""

    parameters: dict[str, float]
    log_likelihood: float
    log_prior: float
    log_posterior: float
    std_errors: dict[str, float] | None
    aic: float
    bic: float
    n_obs: int
    n_params: int
    success: bool
    message: str
    n_iterations: int

    def summary(self) -> str:
        lines = [
            "Maximum A Posteriori (MAP)",
            "=" * 50,
            f"  Converged:      {self.success}",
            f"  Log-posterior:  {self.log_posterior:.4f}",
            f"  Log-likelihood: {self.log_likelihood:.4f}",
            f"  Log-prior:      {self.log_prior:.4f}",
            f"  AIC:            {self.aic:.4f}",
            f"  BIC:            {self.bic:.4f}",
            f"  Observations:   {self.n_obs}",
            f"  Parameters:     {self.n_params}",
            f"  Iterations:     {self.n_iterations}",
            "",
            f"  {'Parameter':<15} {'Estimate':>12} {'Std.Err':>12}",
            f"  {'-' * 15} {'-' * 12} {'-' * 12}",
        ]
        for name, value in self.parameters.items():
            se = self.std_errors.get(name, None) if self.std_errors else None
            se_str = f"{se:12.6f}" if se is not None else "         n/a"
            lines.append(f"  {name:<15} {value:12.6f} {se_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standard errors via numerical Hessian
# ---------------------------------------------------------------------------


def _compute_std_errors(
    objective: callable,
    theta_hat: NDArray[np.float64],
    param_names: list[str],
) -> dict[str, float] | None:
    """Approximate standard errors from the numerical Hessian.

    Computes the Hessian of the *negative* log-likelihood at the MLE
    via finite differences, then inverts to get the covariance matrix.
    Standard errors are the square roots of the diagonal.
    """
    k = len(theta_hat)
    eps = 1e-4

    # Hessian via central finite differences.
    # Reuse objective evaluations across repeated points.
    H = np.zeros((k, k))
    eval_cache: dict[bytes, float] = {}

    def eval_point(theta: NDArray[np.float64]) -> float:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        key = theta_arr.tobytes()
        if key not in eval_cache:
            eval_cache[key] = float(objective(theta_arr))
        return eval_cache[key]

    f0 = eval_point(theta_hat)

    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ei[i] = eps
            ej = np.zeros(k)
            ej[j] = eps

            if i == j:
                fpp = eval_point(theta_hat + ei + ei)
                fmm = eval_point(theta_hat - ei - ei)
                H[i, i] = (fpp - 2.0 * f0 + fmm) / (4.0 * eps * eps)
                continue

            fpp = eval_point(theta_hat + ei + ej)
            fpm = eval_point(theta_hat + ei - ej)
            fmp = eval_point(theta_hat - ei + ej)
            fmm = eval_point(theta_hat - ei - ej)

            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            H[j, i] = H[i, j]

    try:
        cov = np.linalg.inv(H)
        diag = np.diag(cov)
        if np.any(diag < 0):
            return None
        se = np.sqrt(diag)
        return dict(zip(param_names, se, strict=True))
    except np.linalg.LinAlgError:
        return None


def _resolve_observation_count(data: pd.DataFrame | NDArray[np.float64]) -> int:
    if hasattr(data, "shape"):
        return int(data.shape[0])
    return int(len(data))


def _compute_information_criteria(log_likelihood: float, n_params: int, n_obs: int) -> tuple[float, float]:
    aic = -2.0 * log_likelihood + 2.0 * n_params
    bic = -2.0 * log_likelihood + n_params * np.log(n_obs)
    return float(aic), float(bic)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


def estimate_mle(
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
    method: str = "L-BFGS-B",
    options: dict | None = None,
    compute_se: bool = True,
) -> MLEResult:
    """Estimate model parameters by maximum likelihood.

    The steady state is held fixed; for each candidate parameter vector
    the model is re-linearised, solved, converted to state-space form,
    and evaluated via the Kalman filter.

    Args:
        model: ModelIR from ``load_model()``.
        steady_state: Pre-computed steady state (held fixed).
        calibration: Baseline calibration (provides initial values).
        data: Observed data for the Kalman filter.
        observables: Variable names observed in *data*.
        param_names: Names of parameters to estimate.
        bounds: Lower and upper bounds for each parameter.
        measurement_error: Passed through to ``to_state_space()``.
        demean: Subtract steady-state from data before filtering.
        method: ``scipy.optimize.minimize`` method (default L-BFGS-B).
        options: Extra options for the optimiser.
        compute_se: Compute Hessian-based standard errors (default True).

    Returns:
        MLEResult with estimated parameters, log-likelihood, and
        information criteria.
    """
    resolved_names, resolved_bounds = infer_estimation_problem(
        calibration=calibration,
        param_names=param_names,
        bounds=bounds,
    )

    # Build objective
    obj = build_objective(
        model,
        steady_state,
        calibration,
        data,
        observables,
        resolved_names,
        measurement_error=measurement_error,
        demean=demean,
    )

    x0, scipy_bounds = prepare_initial_guess_and_bounds(
        calibration=calibration,
        param_names=resolved_names,
        bounds=resolved_bounds,
    )

    # Optimise
    result = optimize.minimize(
        obj,
        x0,
        method=method,
        bounds=scipy_bounds,
        options=options,
    )

    theta_hat = result.x
    neg_ll = result.fun
    ll = -neg_ll

    n_obs = _resolve_observation_count(data)
    k = len(resolved_names)
    aic, bic = _compute_information_criteria(ll, k, n_obs)

    # Standard errors
    std_errors = None
    if compute_se and result.success:
        std_errors = _compute_std_errors(obj, theta_hat, resolved_names)

    return MLEResult(
        parameters=dict(zip(resolved_names, theta_hat, strict=True)),
        log_likelihood=ll,
        std_errors=std_errors,
        aic=aic,
        bic=bic,
        n_obs=n_obs,
        n_params=k,
        success=result.success,
        message=str(result.message),
        n_iterations=int(getattr(result, "nit", 0)),
    )


def estimate_map(
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
    method: str = "L-BFGS-B",
    options: dict | None = None,
    compute_se: bool = True,
    prior_weight: float = 1.0,
    require_priors: bool = True,
    penalty: float = 1e10,
) -> MAPResult:
    """Estimate posterior mode (MAP): maximize log-likelihood + log-prior."""
    if prior_weight <= 0.0:
        raise ValueError(f"prior_weight must be > 0, got {prior_weight}")

    resolved_names, resolved_bounds = infer_estimation_problem(
        calibration=calibration,
        param_names=param_names,
        bounds=bounds,
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

    def map_objective(theta: NDArray[np.float64]) -> float:
        nll = float(nll_objective(theta))
        if not np.isfinite(nll) or nll >= penalty:
            return penalty
        lp = float(log_prior(theta))
        if not np.isfinite(lp):
            return penalty
        return float(nll - prior_weight * lp)

    x0, scipy_bounds = prepare_initial_guess_and_bounds(
        calibration=calibration,
        param_names=resolved_names,
        bounds=resolved_bounds,
    )

    result = optimize.minimize(
        map_objective,
        x0,
        method=method,
        bounds=scipy_bounds,
        options=options,
    )

    theta_hat = np.asarray(result.x, dtype=np.float64)
    nll_hat = float(nll_objective(theta_hat))
    ll_hat = -nll_hat
    lp_hat = float(log_prior(theta_hat))
    neg_log_post = float(result.fun)
    log_post = -neg_log_post

    n_obs = _resolve_observation_count(data)
    k = len(resolved_names)
    aic, bic = _compute_information_criteria(ll_hat, k, n_obs)

    std_errors = None
    if compute_se and result.success:
        std_errors = _compute_std_errors(map_objective, theta_hat, resolved_names)

    return MAPResult(
        parameters=dict(zip(resolved_names, theta_hat, strict=True)),
        log_likelihood=ll_hat,
        log_prior=lp_hat,
        log_posterior=log_post,
        std_errors=std_errors,
        aic=aic,
        bic=bic,
        n_obs=n_obs,
        n_params=k,
        success=result.success,
        message=str(result.message),
        n_iterations=int(getattr(result, "nit", 0)),
    )
