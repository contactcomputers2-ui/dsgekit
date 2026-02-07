"""Pruned third-order stochastic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.solvers.nonlinear.third_order import ThirdOrderSolution


@dataclass
class ThirdOrderSimulationResult:
    """Results from third-order pruned simulation."""

    data: pd.DataFrame
    shocks: pd.DataFrame
    first_order_component: pd.DataFrame
    second_order_component: pd.DataFrame
    third_order_component: pd.DataFrame
    n_periods: int
    seed: int | None = None

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        return self.data[var_name].values


def _coerce_initial_state(
    initial_state: NDArray[np.float64] | None,
    n_vars: int,
) -> NDArray[np.float64]:
    if initial_state is None:
        return np.zeros(n_vars, dtype=np.float64)
    init = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    if init.shape[0] != n_vars:
        raise ValueError(f"initial_state has length {init.shape[0]}, expected {n_vars}")
    return init.copy()


def _coerce_shocks(
    shocks: NDArray[np.float64] | pd.DataFrame,
    expected_names: list[str],
) -> NDArray[np.float64]:
    n_shocks = len(expected_names)
    if isinstance(shocks, pd.DataFrame):
        missing = [name for name in expected_names if name not in shocks.columns]
        if missing:
            raise ValueError(f"Shock DataFrame is missing columns: {missing}")
        return shocks.loc[:, expected_names].to_numpy(dtype=np.float64, copy=True)

    shock_array = np.asarray(shocks, dtype=np.float64)
    if shock_array.ndim != 2:
        raise ValueError(
            f"shocks must be 2D with shape (n_periods, n_shocks), got ndim={shock_array.ndim}"
        )
    if shock_array.shape[1] != n_shocks:
        raise ValueError(
            f"shocks has {shock_array.shape[1]} columns, expected {n_shocks}"
        )
    return shock_array.copy()


def _shock_cholesky(calibration: Calibration, shock_names: list[str]) -> NDArray[np.float64]:
    shock_cov = calibration.shock_cov_matrix(shock_names)
    if np.allclose(shock_cov, 0.0):
        return np.zeros_like(shock_cov)
    return np.linalg.cholesky(shock_cov)


def _simulate_pruned_arrays(
    solution: ThirdOrderSolution,
    shocks: NDArray[np.float64],
    initial_state: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    T = solution.T
    R = solution.R
    n_periods, n_shocks = shocks.shape
    n_vars = T.shape[0]

    if n_shocks != R.shape[1]:
        raise ValueError(f"shocks has {n_shocks} columns, expected {R.shape[1]}")

    x1 = _coerce_initial_state(initial_state, n_vars)
    x2 = np.zeros(n_vars, dtype=np.float64)
    x3 = np.zeros(n_vars, dtype=np.float64)
    zero_shock = np.zeros(n_shocks, dtype=np.float64)

    paths_total = np.zeros((n_periods, n_vars), dtype=np.float64)
    paths_fo = np.zeros((n_periods, n_vars), dtype=np.float64)
    paths_so = np.zeros((n_periods, n_vars), dtype=np.float64)
    paths_to = np.zeros((n_periods, n_vars), dtype=np.float64)

    for t in range(n_periods):
        u_t = shocks[t, :]
        x1_prev = x1.copy()
        x2_prev = x2.copy()

        quad_11 = solution.quadratic_effect(x1_prev, u_t)
        quad_12 = solution.quadratic_cross_effect(x1_prev, u_t, x2_prev, zero_shock)
        cubic_111 = solution.cubic_effect(x1_prev, u_t)

        x1 = T @ x1_prev + R @ u_t
        x2 = T @ x2_prev + quad_11
        x3 = T @ x3 + quad_12 + cubic_111

        y_t = x1 + x2 + x3
        paths_total[t, :] = y_t
        paths_fo[t, :] = x1
        paths_so[t, :] = x2
        paths_to[t, :] = x3

    return paths_total, paths_fo, paths_so, paths_to


def simulate_pruned_third_order_path(
    solution: ThirdOrderSolution,
    shocks: NDArray[np.float64] | pd.DataFrame,
    *,
    initial_state: NDArray[np.float64] | None = None,
) -> ThirdOrderSimulationResult:
    """Run third-order pruned simulation for a user-provided shock trajectory."""
    shock_array = _coerce_shocks(shocks, solution.shock_names)
    paths_total, paths_fo, paths_so, paths_to = _simulate_pruned_arrays(
        solution,
        shock_array,
        initial_state,
    )

    n_periods = shock_array.shape[0]
    index = range(n_periods)
    data_df = pd.DataFrame(paths_total, index=index, columns=solution.var_names)
    data_df.index.name = "period"
    fo_df = pd.DataFrame(paths_fo, index=index, columns=solution.var_names)
    fo_df.index.name = "period"
    so_df = pd.DataFrame(paths_so, index=index, columns=solution.var_names)
    so_df.index.name = "period"
    to_df = pd.DataFrame(paths_to, index=index, columns=solution.var_names)
    to_df.index.name = "period"
    shocks_df = pd.DataFrame(shock_array, index=index, columns=solution.shock_names)
    shocks_df.index.name = "period"

    return ThirdOrderSimulationResult(
        data=data_df,
        shocks=shocks_df,
        first_order_component=fo_df,
        second_order_component=so_df,
        third_order_component=to_df,
        n_periods=n_periods,
        seed=None,
    )


def simulate_pruned_third_order(
    solution: ThirdOrderSolution,
    calibration: Calibration,
    n_periods: int = 100,
    seed: int | None = None,
    initial_state: NDArray[np.float64] | None = None,
    burn_in: int = 0,
) -> ThirdOrderSimulationResult:
    """Run third-order simulation with pruning decomposition.

    Uses:
      x1_t = T x1_{t-1} + R u_t
      x2_t = T x2_{t-1} + 0.5 * G2([x1_{t-1}, u_t], [x1_{t-1}, u_t])
      x3_t = T x3_{t-1}
             + G2([x1_{t-1}, u_t], [x2_{t-1}, 0])
             + (1/6) * G3([x1_{t-1}, u_t], [x1_{t-1}, u_t], [x1_{t-1}, u_t])
      y_t  = x1_t + x2_t + x3_t
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    n_shocks = solution.n_shocks
    total_periods = n_periods + burn_in

    shock_chol = _shock_cholesky(calibration, solution.shock_names)
    z = rng.standard_normal((total_periods, n_shocks))
    shocks = z @ shock_chol.T
    paths_total, paths_fo, paths_so, paths_to = _simulate_pruned_arrays(
        solution,
        shocks,
        initial_state,
    )

    paths_total = paths_total[burn_in:, :]
    paths_fo = paths_fo[burn_in:, :]
    paths_so = paths_so[burn_in:, :]
    paths_to = paths_to[burn_in:, :]
    shocks = shocks[burn_in:, :]

    index = range(n_periods)
    data_df = pd.DataFrame(paths_total, index=index, columns=solution.var_names)
    data_df.index.name = "period"
    fo_df = pd.DataFrame(paths_fo, index=index, columns=solution.var_names)
    fo_df.index.name = "period"
    so_df = pd.DataFrame(paths_so, index=index, columns=solution.var_names)
    so_df.index.name = "period"
    to_df = pd.DataFrame(paths_to, index=index, columns=solution.var_names)
    to_df.index.name = "period"
    shocks_df = pd.DataFrame(shocks, index=index, columns=solution.shock_names)
    shocks_df.index.name = "period"

    return ThirdOrderSimulationResult(
        data=data_df,
        shocks=shocks_df,
        first_order_component=fo_df,
        second_order_component=so_df,
        third_order_component=to_df,
        n_periods=n_periods,
        seed=seed,
    )
