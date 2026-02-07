"""Posterior-mode helpers for DSGE estimation (MAP)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy import special

from dsgekit.exceptions import EstimationError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dsgekit.model.calibration import Calibration
    from dsgekit.model.priors import PriorSpec


BoundsDict = dict[str, tuple[float | None, float | None]]


def infer_estimation_problem(
    calibration: Calibration,
    param_names: list[str] | None,
    bounds: BoundsDict | None,
) -> tuple[list[str], BoundsDict]:
    """Resolve parameter names and bounds for optimization."""
    if param_names is None:
        if not calibration.estimated_params:
            raise EstimationError(
                "param_names is required when calibration.estimated_params is empty"
            )
        names: list[str] = []
        seen: set[str] = set()
        for ep in calibration.estimated_params:
            if ep.entry_type == "corr":
                raise EstimationError(
                    "Correlation estimated parameters are not supported in "
                    "current MLE/MAP optimization"
                )
            if ep.name in seen:
                raise EstimationError(
                    f"Duplicate estimated parameter name '{ep.name}' in calibration"
                )
            seen.add(ep.name)
            names.append(ep.name)
        param_names = names
    else:
        if len(param_names) == 0:
            raise EstimationError("param_names cannot be empty")
        if len(set(param_names)) != len(param_names):
            raise EstimationError("param_names must be unique")

    inferred_bounds: BoundsDict = {}
    for ep in calibration.estimated_params:
        if ep.entry_type == "corr":
            continue
        inferred_bounds.setdefault(ep.name, (ep.lower_bound, ep.upper_bound))

    merged: BoundsDict = {}
    bounds_input = bounds or {}
    for name in param_names:
        if name in bounds_input:
            merged[name] = bounds_input[name]
        else:
            merged[name] = inferred_bounds.get(name, (None, None))

    return list(param_names), merged


def _resolve_param_slots(
    calibration: Calibration,
    param_names: list[str],
) -> list[tuple[str, str]]:
    """Classify names as parameter or shock stderr slots."""
    slots: list[tuple[str, str]] = []
    for name in param_names:
        if name in calibration.parameters:
            slots.append((name, "param"))
        elif name in calibration.shock_stderr:
            slots.append((name, "shock"))
        else:
            raise EstimationError(
                f"'{name}' not found in calibration.parameters or "
                f"calibration.shock_stderr"
            )
    return slots


def prepare_initial_guess_and_bounds(
    calibration: Calibration,
    param_names: list[str],
    bounds: BoundsDict,
) -> tuple[NDArray[np.float64], list[tuple[float | None, float | None]]]:
    """Create robust initial guess and scipy bounds."""
    slots = _resolve_param_slots(calibration, param_names)

    x0 = np.zeros(len(slots), dtype=np.float64)
    scipy_bounds: list[tuple[float | None, float | None]] = []

    for i, (name, slot_type) in enumerate(slots):
        value = calibration.parameters.get(name, calibration.shock_stderr.get(name))
        if value is None:
            raise EstimationError(f"Missing initial value for '{name}' in calibration")
        v = float(value)

        lb_raw, ub_raw = bounds.get(name, (None, None))
        lb = -np.inf if lb_raw is None else float(lb_raw)
        ub = np.inf if ub_raw is None else float(ub_raw)

        # stderr must stay strictly positive for covariance construction
        if slot_type == "shock":
            lb = max(lb, 1e-12)

        if lb > ub:
            raise EstimationError(
                f"Invalid bounds for '{name}': lower={lb} > upper={ub}"
            )

        if not np.isfinite(v):
            if np.isfinite(lb) and np.isfinite(ub):
                v = 0.5 * (lb + ub)
            elif np.isfinite(lb):
                v = lb + max(1e-6, 1e-4 * max(1.0, abs(lb)))
            elif np.isfinite(ub):
                v = ub - max(1e-6, 1e-4 * max(1.0, abs(ub)))
            else:
                v = 0.0

        v = min(max(v, lb), ub)
        x0[i] = float(v)
        scipy_bounds.append(
            (
                None if not np.isfinite(lb) else float(lb),
                None if not np.isfinite(ub) else float(ub),
            )
        )

    return x0, scipy_bounds


def _logpdf_normal(x: float, mean: float, std: float) -> float:
    z = (x - mean) / std
    return -0.5 * math.log(2.0 * math.pi) - math.log(std) - 0.5 * z * z


def _beta_shape_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    var = std * std
    kappa = mean * (1.0 - mean) / var - 1.0
    if kappa <= 0.0:
        raise EstimationError(
            "Invalid beta prior moments: require std^2 < mean*(1-mean)"
        )
    alpha = mean * kappa
    beta = (1.0 - mean) * kappa
    return alpha, beta


def _logpdf_beta(x: float, mean: float, std: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return -np.inf
    alpha, beta = _beta_shape_from_mean_std(mean, std)
    return (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x) - special.betaln(
        alpha, beta
    )


def _gamma_shape_scale_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    shape = (mean / std) ** 2
    scale = (std * std) / mean
    return shape, scale


def _logpdf_gamma(x: float, mean: float, std: float) -> float:
    if x <= 0.0:
        return -np.inf
    shape, scale = _gamma_shape_scale_from_mean_std(mean, std)
    return (
        (shape - 1.0) * math.log(x)
        - x / scale
        - shape * math.log(scale)
        - special.gammaln(shape)
    )


def _inv_gamma_shape_scale_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    shape = 2.0 + (mean * mean) / (std * std)
    scale = mean * (shape - 1.0)
    return shape, scale


def _logpdf_inv_gamma(x: float, mean: float, std: float) -> float:
    if x <= 0.0:
        return -np.inf
    shape, scale = _inv_gamma_shape_scale_from_mean_std(mean, std)
    return (
        shape * math.log(scale)
        - special.gammaln(shape)
        - (shape + 1.0) * math.log(x)
        - scale / x
    )


def _logpdf_prior(x: float, prior: PriorSpec) -> float:
    dist = prior.distribution
    if dist == "normal_pdf":
        return _logpdf_normal(x, prior.mean, prior.std)
    if dist == "beta_pdf":
        return _logpdf_beta(x, prior.mean, prior.std)
    if dist == "gamma_pdf":
        return _logpdf_gamma(x, prior.mean, prior.std)
    if dist == "inv_gamma_pdf":
        return _logpdf_inv_gamma(x, prior.mean, prior.std)
    raise EstimationError(f"Unsupported prior distribution '{dist}'")


def build_log_prior_evaluator(
    calibration: Calibration,
    param_names: list[str],
    *,
    require_priors: bool = True,
) -> Callable[[NDArray[np.float64]], float]:
    """Build ``theta -> log prior`` callable aligned with param_names."""
    slots = _resolve_param_slots(calibration, param_names)

    ep_lookup: dict[tuple[str, str], PriorSpec | None] = {}
    for ep in calibration.estimated_params:
        if ep.entry_type not in {"param", "stderr"}:
            continue
        key = (ep.entry_type, ep.name)
        if key in ep_lookup:
            raise EstimationError(
                f"Duplicate estimated_params entry for {ep.entry_type}:{ep.name}"
            )
        ep_lookup[key] = ep.prior

    compiled: list[PriorSpec | None] = []
    for name, slot in slots:
        key = ("param", name) if slot == "param" else ("stderr", name)
        prior = ep_lookup.get(key)
        if require_priors and prior is None:
            raise EstimationError(
                f"Missing prior for estimable '{name}' "
                f"(expected estimated_params entry type '{key[0]}')"
            )
        compiled.append(prior)

    def log_prior(theta: NDArray[np.float64]) -> float:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape[0] != len(compiled):
            raise EstimationError(
                f"Expected theta length {len(compiled)}, got {theta_arr.shape[0]}"
            )
        if not np.all(np.isfinite(theta_arr)):
            return float("-inf")

        total = 0.0
        for value, prior in zip(theta_arr, compiled, strict=True):
            if prior is None:
                continue
            lp = _logpdf_prior(float(value), prior)
            if not np.isfinite(lp):
                return float("-inf")
            total += float(lp)
        return float(total)

    return log_prior
