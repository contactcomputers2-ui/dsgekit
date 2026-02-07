"""Likelihood objective function for DSGE estimation.

Builds a callable that maps a parameter vector to the negative
log-likelihood, suitable for use with ``scipy.optimize.minimize``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from dsgekit.exceptions import (
    BlanchardKahnError,
    DSGEKitError,
    EstimationError,
    FilterError,
    SolverError,
)
from dsgekit.filters.kalman import log_likelihood
from dsgekit.solvers import solve_linear
from dsgekit.transforms import linearize, to_state_space

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


def build_objective(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    data: pd.DataFrame | NDArray[np.float64],
    observables: list[str],
    param_names: list[str],
    *,
    measurement_error: float | NDArray[np.float64] | None = None,
    demean: bool = True,
    penalty: float = 1e10,
    cache: bool = True,
    cache_max_size: int = 2048,
) -> Callable[[NDArray[np.float64]], float]:
    """Build a negative-log-likelihood objective function.

    Returns a callable ``f(theta) -> float`` where *theta* is a 1-D
    array of parameter values (in the same order as *param_names*).
    The function returns ``-log L(theta | data)`` so that it can be
    **minimised** by ``scipy.optimize.minimize``.

    If a parameter vector leads to an indeterminate / explosive model
    or a numerical failure in the Kalman filter, the function returns
    *penalty* instead of raising.

    Args:
        model: ModelIR from ``load_model()``.
        steady_state: Pre-computed steady state (held fixed).
        calibration: Baseline calibration (used as template).
        data: Observed data for the Kalman filter.
        observables: Variable names observed in *data*.
        param_names: Names of parameters to estimate.  Each name must
            exist in ``calibration.parameters`` **or**
            ``calibration.shock_stderr``.
        measurement_error: Passed through to ``to_state_space()``.
        demean: Subtract steady-state from data before filtering.
        penalty: Value returned when the model cannot be solved or the
            likelihood cannot be evaluated.
        cache: Enable memoization for repeated ``theta`` evaluations.
        cache_max_size: Maximum number of cached points (LRU eviction).
            Ignored when ``cache=False``.

    Returns:
        Callable that maps ``theta`` (1-D array) to negative
        log-likelihood (scalar float).

    Raises:
        EstimationError: If a name in *param_names* is not found in
            calibration parameters or shock standard errors.
    """
    # Classify each name as "parameter" or "shock_stderr"
    _param_slots: list[tuple[str, str]] = []  # (name, "param"|"shock")
    for name in param_names:
        if name in calibration.parameters:
            _param_slots.append((name, "param"))
        elif name in calibration.shock_stderr:
            _param_slots.append((name, "shock"))
        else:
            raise EstimationError(
                f"'{name}' not found in calibration.parameters or "
                f"calibration.shock_stderr"
            )

    cache_store: OrderedDict[bytes, float] | None = None
    cache_hits = 0
    cache_misses = 0
    if cache:
        if cache_max_size < 1:
            raise EstimationError("cache_max_size must be >= 1 when cache=True")
        cache_store = OrderedDict()

    def _cache_get(key: bytes) -> float | None:
        nonlocal cache_hits
        if cache_store is None:
            return None
        val = cache_store.pop(key, None)
        if val is None:
            return None
        cache_store[key] = val
        cache_hits += 1
        return val

    def _cache_put(key: bytes, value: float) -> None:
        nonlocal cache_misses
        if cache_store is None:
            return
        cache_misses += 1
        cache_store[key] = value
        if len(cache_store) > cache_max_size:
            cache_store.popitem(last=False)

    def objective(theta: NDArray[np.float64]) -> float:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta_arr.shape[0] != len(_param_slots):
            raise EstimationError(
                f"Expected theta with length {len(_param_slots)}, "
                f"got {theta_arr.shape[0]}"
            )
        if not np.all(np.isfinite(theta_arr)):
            return penalty

        key = theta_arr.tobytes()
        cached = _cache_get(key)
        if cached is not None:
            return cached

        # Build a modified calibration
        cal = calibration.copy()
        for (name, slot), value in zip(_param_slots, theta_arr, strict=True):
            if slot == "param":
                cal.set_parameter(name, float(value))
            else:
                cal.shock_stderr[name] = float(value)

        try:
            lin = linearize(model, steady_state, cal)
            sol = solve_linear(lin)
            ss = to_state_space(
                sol, cal, observables, measurement_error=measurement_error
            )
            ll = log_likelihood(ss, data, demean=demean)
            out = -ll
        except (BlanchardKahnError, SolverError, FilterError, DSGEKitError):
            out = penalty

        _cache_put(key, out)
        return out

    def cache_info() -> dict[str, int]:
        return {
            "enabled": int(cache_store is not None),
            "hits": cache_hits,
            "misses": cache_misses,
            "size": len(cache_store) if cache_store is not None else 0,
            "max_size": cache_max_size if cache_store is not None else 0,
        }

    def cache_clear() -> None:
        nonlocal cache_hits, cache_misses
        if cache_store is not None:
            cache_store.clear()
        cache_hits = 0
        cache_misses = 0

    objective.cache_info = cache_info  # type: ignore[attr-defined]
    objective.cache_clear = cache_clear  # type: ignore[attr-defined]

    return objective
