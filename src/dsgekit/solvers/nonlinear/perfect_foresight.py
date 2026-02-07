"""Deterministic perfect-foresight solver for non-linear DSGE models.

Solves for a full trajectory ``y_1, ..., y_T`` under known shock paths and
boundary conditions:

    f(y_{t-1}, y_t, y_{t+1}, u_t; theta) = 0,  t = 1..T

with fixed ``y_0`` (initial) and ``y_{T+1}`` (terminal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from dsgekit.derivatives import DerivativeStack, var_coord
from dsgekit.exceptions import SolverError
from dsgekit.model.equations import EvalContext
from dsgekit.solvers.nonlinear._newton_linear import (
    BlockTridiagonalJacobianBuilder,
    JitBackendMode,
    LinearSolverMode,
    resolve_jit_backend_mode,
    resolve_linear_solver_mode,
    solve_newton_step,
)

if TYPE_CHECKING:
    from dsgekit.model.calibration import Calibration
    from dsgekit.model.ir import ModelIR
    from dsgekit.model.steady_state import SteadyState


@dataclass(slots=True)
class PerfectForesightResult:
    """Output for deterministic perfect-foresight trajectories."""

    path: pd.DataFrame
    shocks: pd.DataFrame
    residuals: pd.DataFrame
    converged: bool
    n_iterations: int
    max_abs_residual: float
    residual_history: list[float] = field(default_factory=list)
    solver_meta: dict[str, float | int | str] = field(default_factory=dict)

    def __getitem__(self, var_name: str) -> NDArray[np.float64]:
        return self.path[var_name].values

    def summary(self) -> str:
        lines = [
            "Perfect Foresight Solver",
            "=" * 50,
            f"  Converged:        {self.converged}",
            f"  Iterations:       {self.n_iterations}",
            f"  Max |residual|:   {self.max_abs_residual:.3e}",
            f"  Periods:          {self.path.shape[0]}",
            f"  Variables:        {self.path.shape[1]}",
            f"  Shocks:           {self.shocks.shape[1]}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class NewsShock:
    """Single deterministic shock event with optional anticipation horizon."""

    shock: str
    impact_period: int
    value: float
    announcement_period: int | None = None

    def __post_init__(self) -> None:
        if self.impact_period < 1:
            raise SolverError(
                f"impact_period must be >= 1, got {self.impact_period}"
            )
        announce = self.announcement
        if announce < 1:
            raise SolverError(
                f"announcement_period must be >= 1, got {announce}"
            )
        if announce > self.impact_period:
            raise SolverError(
                "announcement_period cannot exceed impact_period "
                f"(got {announce} > {self.impact_period})"
            )
        if not np.isfinite(self.value):
            raise SolverError(f"value must be finite, got {self.value}")

    @property
    def announcement(self) -> int:
        """Announcement period (defaults to impact period)."""
        if self.announcement_period is None:
            return self.impact_period
        return self.announcement_period

    @property
    def anticipation_horizon(self) -> int:
        """Number of periods between announcement and impact."""
        return self.impact_period - self.announcement

    @property
    def is_anticipated(self) -> bool:
        """Whether event is announced before impact (news shock)."""
        return self.anticipation_horizon > 0


def unanticipated_shock(
    shock: str,
    *,
    period: int,
    value: float,
) -> NewsShock:
    """Create an unanticipated shock (impact in announcement period)."""
    return NewsShock(
        shock=shock,
        impact_period=int(period),
        value=float(value),
        announcement_period=int(period),
    )


def anticipated_shock(
    shock: str,
    *,
    announcement_period: int,
    horizon: int,
    value: float,
) -> NewsShock:
    """Create an anticipated shock announced at t and realized at t+horizon."""
    if horizon < 1:
        raise SolverError(f"horizon must be >= 1 for anticipated shocks, got {horizon}")
    announce = int(announcement_period)
    return NewsShock(
        shock=shock,
        impact_period=announce + int(horizon),
        value=float(value),
        announcement_period=announce,
    )


def build_news_shock_path(
    *,
    n_periods: int,
    shock_names: list[str],
    events: list[NewsShock] | tuple[NewsShock, ...] | None = None,
) -> NDArray[np.float64]:
    """Build deterministic shock path from anticipated/unanticipated events."""
    if n_periods < 1:
        raise SolverError(f"n_periods must be >= 1, got {n_periods}")

    out = np.zeros((n_periods, len(shock_names)), dtype=np.float64)
    if not events:
        return out

    shock_index = {name: j for j, name in enumerate(shock_names)}
    for event in events:
        j = shock_index.get(event.shock)
        if j is None:
            raise SolverError(f"Unknown shock in event definition: '{event.shock}'")
        if event.impact_period > n_periods:
            raise SolverError(
                f"Shock '{event.shock}' impact period {event.impact_period} is outside "
                f"the simulation horizon (n_periods={n_periods})"
            )
        out[event.impact_period - 1, j] += float(event.value)

    return out


def _resolve_state_vector(
    *,
    value: dict[str, float] | NDArray[np.float64] | None,
    var_names: list[str],
    baseline: NDArray[np.float64],
    label: str,
) -> NDArray[np.float64]:
    """Resolve dict/array state specification into aligned vector."""
    if value is None:
        return baseline.copy()

    if isinstance(value, dict):
        out = baseline.copy()
        unknown = [name for name in value if name not in var_names]
        if unknown:
            raise SolverError(f"{label} contains unknown variables: {unknown}")
        for i, name in enumerate(var_names):
            if name in value:
                out[i] = float(value[name])
        return out

    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != len(var_names):
        raise SolverError(
            f"{label} must have length {len(var_names)}, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise SolverError(f"{label} must be finite")
    return arr.astype(np.float64, copy=True)


def _resolve_shock_path(
    *,
    shocks: dict[str, NDArray[np.float64]] | NDArray[np.float64] | None,
    n_periods: int,
    shock_names: list[str],
) -> NDArray[np.float64]:
    """Resolve deterministic shock path into dense (T x n_shocks) array."""
    n_shocks = len(shock_names)
    if shocks is None:
        return np.zeros((n_periods, n_shocks), dtype=np.float64)

    if isinstance(shocks, dict):
        out = np.zeros((n_periods, n_shocks), dtype=np.float64)
        unknown = [name for name in shocks if name not in shock_names]
        if unknown:
            raise SolverError(f"shocks contains unknown shocks: {unknown}")
        for j, name in enumerate(shock_names):
            if name not in shocks:
                continue
            vec = np.asarray(shocks[name], dtype=np.float64).reshape(-1)
            if vec.shape[0] != n_periods:
                raise SolverError(
                    f"Shock path '{name}' must have length {n_periods}, got {vec.shape[0]}"
                )
            out[:, j] = vec
        return out

    arr = np.asarray(shocks, dtype=np.float64)
    if arr.shape != (n_periods, n_shocks):
        raise SolverError(
            f"shocks array must have shape ({n_periods}, {n_shocks}), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise SolverError("shocks array must be finite")
    return arr.astype(np.float64, copy=True)


def _default_initial_state_from_steady_state(
    *,
    steady_state: SteadyState,
    var_names: list[str],
    baseline: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Initial boundary default: steady-state/initval with histval(0) overrides."""
    out = baseline.copy()
    var_index = {name: i for i, name in enumerate(var_names)}
    for (var_name, timing), value in steady_state.histval.items():
        if timing != 0:
            continue
        idx = var_index.get(var_name)
        if idx is None:
            continue
        out[idx] = float(value)
    return out


def _default_terminal_state_from_steady_state(
    *,
    steady_state: SteadyState,
    var_names: list[str],
    baseline: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Terminal boundary default: steady-state/initval with endval overrides."""
    out = baseline.copy()
    var_index = {name: i for i, name in enumerate(var_names)}
    for var_name, value in steady_state.endval.items():
        idx = var_index.get(var_name)
        if idx is None:
            continue
        out[idx] = float(value)
    return out


def _default_shock_path_from_steady_state(
    *,
    steady_state: SteadyState,
    n_periods: int,
    shock_names: list[str],
) -> NDArray[np.float64]:
    """Default deterministic shock path from steady_state metadata."""
    out = np.zeros((n_periods, len(shock_names)), dtype=np.float64)
    if not steady_state.deterministic_shocks:
        return out

    shock_index = {name: j for j, name in enumerate(shock_names)}
    for (shock_name, period), value in steady_state.deterministic_shocks.items():
        if shock_name not in shock_index:
            raise SolverError(
                f"deterministic_shocks contains unknown shock '{shock_name}'"
            )
        if period < 1 or period > n_periods:
            raise SolverError(
                f"deterministic shock period for '{shock_name}' must be in "
                f"[1, {n_periods}], got {period}"
            )
        out[period - 1, shock_index[shock_name]] = float(value)
    return out


def _resolve_initial_path(
    *,
    initial_path: pd.DataFrame | NDArray[np.float64] | None,
    n_periods: int,
    var_names: list[str],
    x_initial: NDArray[np.float64],
    x_terminal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Initial guess for Newton iterations."""
    n_vars = len(var_names)
    if initial_path is None:
        if n_periods == 1:
            return ((x_initial + x_terminal) * 0.5).reshape(1, -1)
        w = (np.arange(1, n_periods + 1, dtype=np.float64) / (n_periods + 1.0)).reshape(
            -1, 1
        )
        return (1.0 - w) * x_initial[np.newaxis, :] + w * x_terminal[np.newaxis, :]

    if isinstance(initial_path, pd.DataFrame):
        missing = [name for name in var_names if name not in initial_path.columns]
        if missing:
            raise SolverError(
                f"initial_path DataFrame missing variable columns: {missing}"
            )
        arr = initial_path[var_names].values.astype(np.float64)
    else:
        arr = np.asarray(initial_path, dtype=np.float64)

    if arr.shape != (n_periods, n_vars):
        raise SolverError(
            f"initial_path must have shape ({n_periods}, {n_vars}), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise SolverError("initial_path must be finite")
    return arr.copy()


def _set_period_context(
    *,
    context: EvalContext,
    var_names: list[str],
    shock_names: list[str],
    y_prev: NDArray[np.float64],
    y_curr: NDArray[np.float64],
    y_next: NDArray[np.float64],
    u_t: NDArray[np.float64],
) -> None:
    for i, name in enumerate(var_names):
        context.set_variable(name, -1, float(y_prev[i]))
        context.set_variable(name, 0, float(y_curr[i]))
        context.set_variable(name, 1, float(y_next[i]))
    for j, shock in enumerate(shock_names):
        context.set_shock(shock, float(u_t[j]))


def solve_perfect_foresight(
    model: ModelIR,
    steady_state: SteadyState,
    calibration: Calibration,
    *,
    n_periods: int,
    shocks: dict[str, NDArray[np.float64]] | NDArray[np.float64] | None = None,
    initial_state: dict[str, float] | NDArray[np.float64] | None = None,
    terminal_state: dict[str, float] | NDArray[np.float64] | None = None,
    initial_path: pd.DataFrame | NDArray[np.float64] | None = None,
    derivative_backend: str = "numeric",
    eps: float = 1e-6,
    tol: float = 1e-8,
    max_iter: int = 50,
    line_search_max_steps: int = 12,
    line_search_shrink: float = 0.5,
    linear_solver: LinearSolverMode = "auto",
    jit_backend: JitBackendMode = "none",
    sparse_threshold: int = 200,
    raise_on_fail: bool = True,
) -> PerfectForesightResult:
    """Solve deterministic non-linear transition path with perfect foresight."""
    if n_periods < 1:
        raise SolverError(f"n_periods must be >= 1, got {n_periods}")
    if max_iter < 1:
        raise SolverError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0.0 or not np.isfinite(tol):
        raise SolverError(f"tol must be finite and > 0, got {tol}")
    if line_search_max_steps < 1:
        raise SolverError(
            f"line_search_max_steps must be >= 1, got {line_search_max_steps}"
        )
    if not (0.0 < line_search_shrink < 1.0):
        raise SolverError(
            "line_search_shrink must be in (0, 1), "
            f"got {line_search_shrink}"
        )

    model.validate()
    if model.lead_lag.max_lag > 1 or model.lead_lag.max_lead > 1:
        raise SolverError(
            "Perfect foresight solver supports timings up to one lag/lead "
            f"(max_lag={model.lead_lag.max_lag}, max_lead={model.lead_lag.max_lead})."
        )

    n_vars = model.n_variables
    n_eq = model.n_equations
    if n_eq != n_vars:
        raise SolverError(
            "Perfect foresight solver currently requires a square system: "
            f"{n_eq} equations vs {n_vars} variables"
        )
    resolved_linear_solver = resolve_linear_solver_mode(
        linear_solver=linear_solver,
        n_unknowns=n_periods * n_vars,
        sparse_threshold=sparse_threshold,
    )
    resolved_jit_backend = resolve_jit_backend_mode(
        jit_backend=jit_backend,
        linear_solver=resolved_linear_solver,
    )

    var_names = model.variable_names
    shock_names = model.shock_names
    eq_names = [eq.name or f"eq_{i+1}" for i, eq in enumerate(model.equations)]

    steady_vec = np.array(
        [float(steady_state.values.get(name, 0.0)) for name in var_names],
        dtype=np.float64,
    )
    initial_default = _default_initial_state_from_steady_state(
        steady_state=steady_state,
        var_names=var_names,
        baseline=steady_vec,
    )
    terminal_default = _default_terminal_state_from_steady_state(
        steady_state=steady_state,
        var_names=var_names,
        baseline=steady_vec,
    )
    x_initial = _resolve_state_vector(
        value=initial_state,
        var_names=var_names,
        baseline=initial_default,
        label="initial_state",
    )
    x_terminal = _resolve_state_vector(
        value=terminal_state,
        var_names=var_names,
        baseline=terminal_default,
        label="terminal_state",
    )
    if shocks is None:
        shock_path = _default_shock_path_from_steady_state(
            steady_state=steady_state,
            n_periods=n_periods,
            shock_names=shock_names,
        )
    else:
        shock_path = _resolve_shock_path(
            shocks=shocks,
            n_periods=n_periods,
            shock_names=shock_names,
        )
    x_guess = _resolve_initial_path(
        initial_path=initial_path,
        n_periods=n_periods,
        var_names=var_names,
        x_initial=x_initial,
        x_terminal=x_terminal,
    )

    coords = (
        [var_coord(name, -1) for name in var_names]
        + [var_coord(name, 0) for name in var_names]
        + [var_coord(name, 1) for name in var_names]
    )
    deriv_stack = DerivativeStack(derivative_backend, eps=eps)

    context = EvalContext(
        variables={},
        shocks=dict.fromkeys(shock_names, 0.0),
        parameters=calibration.parameters.copy(),
    )

    def evaluate_system(
        x_flat: NDArray[np.float64],
        *,
        compute_jacobian: bool,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64] | sparse.csr_matrix | None,
        int | None,
    ]:
        x_path = np.asarray(x_flat, dtype=np.float64).reshape(n_periods, n_vars)
        residual_vec = np.zeros(n_periods * n_eq, dtype=np.float64)
        jac_builder = None
        if compute_jacobian:
            jac_builder = BlockTridiagonalJacobianBuilder(
                n_periods=n_periods,
                n_eq=n_eq,
                n_vars=n_vars,
                use_sparse=(resolved_linear_solver == "sparse"),
                jit_backend=resolved_jit_backend,
            )

        for t in range(n_periods):
            y_prev = x_initial if t == 0 else x_path[t - 1, :]
            y_curr = x_path[t, :]
            y_next = x_terminal if t == (n_periods - 1) else x_path[t + 1, :]
            _set_period_context(
                context=context,
                var_names=var_names,
                shock_names=shock_names,
                y_prev=y_prev,
                y_curr=y_curr,
                y_next=y_next,
                u_t=shock_path[t, :],
            )

            row_slice = slice(t * n_eq, (t + 1) * n_eq)
            f_t = model.residuals(context)
            if not np.all(np.isfinite(f_t)):
                raise SolverError(
                    f"Non-finite residuals encountered at period {t + 1}"
                )
            residual_vec[row_slice] = f_t

            if jac_builder is not None:
                j_t = deriv_stack.jacobian(model, context, coords)
                if not np.all(np.isfinite(j_t)):
                    raise SolverError(
                        f"Non-finite Jacobian entries at period {t + 1}"
                    )
                a_t = j_t[:, :n_vars]
                b_t = j_t[:, n_vars : 2 * n_vars]
                c_t = j_t[:, 2 * n_vars :]

                jac_builder.add_period_blocks(
                    period=t,
                    a_t=a_t,
                    b_t=b_t,
                    c_t=c_t,
                )

        if jac_builder is None:
            return residual_vec, None, None
        jac_assembly = jac_builder.build()
        return residual_vec, jac_assembly.matrix, jac_assembly.nnz

    x = x_guess.reshape(-1)
    residual_history: list[float] = []
    converged = False
    n_iterations = 0
    stalled = False
    last_jacobian_nnz = 0

    for it in range(1, max_iter + 1):
        n_iterations = it
        try:
            f_val, jac, jac_nnz = evaluate_system(x, compute_jacobian=True)
        except Exception as exc:
            if raise_on_fail:
                raise SolverError(
                    f"Perfect foresight evaluation failed at iteration {it}: {exc}"
                ) from exc
            break

        assert jac is not None  # for type checkers
        if jac_nnz is not None:
            last_jacobian_nnz = jac_nnz
        current_norm = float(np.max(np.abs(f_val)))
        residual_history.append(current_norm)
        if current_norm < tol:
            converged = True
            break

        rhs = -f_val
        delta = solve_newton_step(
            jacobian=jac,
            rhs=rhs,
            solver_mode=resolved_linear_solver,
        )

        if not np.all(np.isfinite(delta)):
            if raise_on_fail:
                raise SolverError(
                    f"Newton step became non-finite at iteration {it}"
                )
            break

        accepted = False
        alpha = 1.0
        for _ in range(line_search_max_steps):
            x_trial = x + alpha * delta
            try:
                f_trial, _, _ = evaluate_system(x_trial, compute_jacobian=False)
            except Exception:
                alpha *= line_search_shrink
                continue

            trial_norm = float(np.max(np.abs(f_trial)))
            if np.isfinite(trial_norm) and trial_norm < current_norm:
                x = x_trial
                accepted = True
                break
            alpha *= line_search_shrink

        if not accepted:
            stalled = True
            break

    f_final, _, _ = evaluate_system(x, compute_jacobian=False)
    max_abs_residual = float(np.max(np.abs(f_final)))
    converged = converged or (max_abs_residual < tol)

    x_final = x.reshape(n_periods, n_vars)
    period_index = pd.RangeIndex(start=1, stop=n_periods + 1, name="period")
    path_df = pd.DataFrame(x_final, index=period_index, columns=var_names)
    shocks_df = pd.DataFrame(shock_path, index=period_index, columns=shock_names)
    residuals_df = pd.DataFrame(
        f_final.reshape(n_periods, n_eq),
        index=period_index,
        columns=eq_names,
    )

    jac_rows = n_periods * n_eq
    jac_cols = n_periods * n_vars
    jac_density = (
        float(last_jacobian_nnz) / float(jac_rows * jac_cols)
        if jac_rows > 0 and jac_cols > 0
        else 0.0
    )

    result = PerfectForesightResult(
        path=path_df,
        shocks=shocks_df,
        residuals=residuals_df,
        converged=converged,
        n_iterations=n_iterations,
        max_abs_residual=max_abs_residual,
        residual_history=residual_history,
        solver_meta={
            "solver": "perfect_foresight_newton",
            "derivative_backend": derivative_backend,
            "eps": float(eps),
            "tol": float(tol),
            "max_iter": int(max_iter),
            "line_search_max_steps": int(line_search_max_steps),
            "line_search_shrink": float(line_search_shrink),
            "linear_solver_requested": str(linear_solver),
            "linear_solver_used": resolved_linear_solver,
            "jit_backend_requested": str(jit_backend),
            "jit_backend_used": resolved_jit_backend,
            "sparse_threshold": int(sparse_threshold),
            "jacobian_nnz_last": int(last_jacobian_nnz),
            "jacobian_density_last": float(jac_density),
            "stalled": int(stalled),
        },
    )

    if not result.converged and raise_on_fail:
        raise SolverError(
            "Perfect foresight solver did not converge "
            f"(iterations={result.n_iterations}, "
            f"max_abs_residual={result.max_abs_residual:.3e})"
        )
    return result
